#!/usr/bin/env python
"""IF-SitePred (Carbery et al. 2024, github.com/annacarbery/binding-sites @fb9d778) wrapper -> common format.

  python baselines/run_ifsitepred.py <structs|dir|@list> -o OUT [--device cuda:0] [-j N_clean_procs]

Pipeline (official): ESM-IF1 encoder embedding per residue (single chain, backbone N/CA/C) -> 40 LightGBM
classifiers; residue is 'binding' only if ALL 40 predict >=0.5 -> pseudo-atom grid (1.5 A) around the chain,
keep grid points 3-6 A from protein atoms, collect points within 4.5 A of each binding residue, keep points hit by
>=3 binding residues, DBSCAN(eps=1.7, min_samples=2) -> clusters ranked by number of (multiplicity-weighted)
points; centre = mean of cluster points. The official code uses PyMOL for the grid steps; here the same logic is
re-implemented with scipy (no PyMOL), verified to reproduce on 12as.
Deviations: official code only processes the first chain (sorted) and reports top-3 sites; we process every chain
(`--first-chain-only` for the official behaviour) and report all clusters, ranked by size (sites from different
chains are merged and ranked together). score = cluster size (weighted point count).
Extra output <stem>.residues.csv: chain,resnum,mean_prob (mean of 40 LightGBM probabilities, per residue).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

TOOL = C.TOOLS / "ifsitepred"
C.reexec_in_venv(TOOL / "venv")
os.environ.setdefault("TORCH_HOME", str(TOOL / "torch_home"))

import warnings
warnings.filterwarnings("ignore")
import numpy as np


def chain_arrays(pdb_path):
    """per chain: list of (chain, resnum, icode), backbone coords L x3x3 (nan if missing), all heavy-atom coords."""
    import gemmi
    st = gemmi.read_structure(str(pdb_path))
    out = []
    for ch in st[0]:
        ids, bb, atoms, res_atoms = [], [], [], []
        for r in ch:
            d = {a.name: a.pos for a in r}
            if "CA" not in d and "N" not in d and "C" not in d:
                continue
            c = np.full((3, 3), np.nan, dtype=np.float32)
            for k, n in enumerate(("N", "CA", "C")):
                if n in d:
                    c[k] = (d[n].x, d[n].y, d[n].z)
            ids.append((ch.name, r.seqid.num, r.seqid.icode.strip()))
            bb.append(c)
            xyz = np.array([(a.pos.x, a.pos.y, a.pos.z) for a in r], dtype=np.float32)
            res_atoms.append(xyz)
        if len(ids) >= 3:
            out.append((ch.name, ids, np.stack(bb), res_atoms))
    return out


def cloud_sites(res_atoms, bind_idx, grid_step=1.5):
    """PyMOL-free re-implementation of get_final_cloud + DBSCAN. Returns list of sites sorted by size."""
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN
    allxyz = np.concatenate(res_atoms)
    lo, hi = allxyz.min(0) - 10, allxyz.max(0) + 10
    axes = [np.arange(lo[i], hi[i], grid_step) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tree = cKDTree(allxyz)
    d, _ = tree.query(grid, distance_upper_bound=6.0)
    keep = (d > 3.0) & (d <= 6.0)       # cloud3 (<=3 A) removed first, then bubble (<=6 A)
    bubble = grid[keep]
    if len(bubble) == 0 or len(bind_idx) == 0:
        return []
    btree = cKDTree(bubble)
    counts = np.zeros(len(bubble), dtype=np.int32)
    contrib = {}
    for ri in bind_idx:
        hit = set()
        for lst in btree.query_ball_point(res_atoms[ri], 4.5):
            hit.update(lst)
        if hit:
            h = np.fromiter(hit, int)
            counts[h] += 1
            contrib[ri] = h
    sel = np.where(counts > 2)[0]
    if len(sel) == 0:
        return []
    pts = bubble[sel]
    w = counts[sel]
    lab = DBSCAN(eps=1.7, min_samples=2).fit(pts, sample_weight=w).labels_
    sites = []
    for l in set(lab) - {-1}:
        m = lab == l
        sel_set = set(sel[m].tolist())
        res = [ri for ri, h in contrib.items() if sel_set.intersection(h.tolist())]
        sites.append(dict(size=float(w[m].sum()), center=(pts[m] * w[m, None]).sum(0) / w[m].sum(),
                          points=pts[m], res=sorted(res)))
    sites.sort(key=lambda s: -s["size"])
    return sites


class Predictor:
    def __init__(self, device):
        import pickle
        import esm
        import torch
        self.torch, self.esm = torch, esm
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model = self.model.eval().to(device)
        self.gbm = [pickle.load(open(TOOL / "src" / "models" / f"lgbm_if1_{i}.pkl", "rb")) for i in range(40)]

    def residue_probs(self, bb):
        torch = self.torch
        with torch.no_grad():
            # = esm.inverse_folding.util.get_encoder_output with tensors moved to the device
            conv = self.esm.inverse_folding.util.CoordBatchConverter(self.alphabet)
            coords, conf, _, _, pad = conv([(bb, None, None)])
            out = self.model.encoder.forward(coords.to(self.device), pad.to(self.device),
                                             conf.to(self.device), return_all_hiddens=False)
            emb = out["encoder_out"][0][1:-1, 0]
        emb = emb.float().cpu().numpy()
        return np.stack([m.predict(emb, num_threads=4) for m in self.gbm])  # 40 x L


def process(pred, pdb_path, stem, out, info, args):
    arrs = chain_arrays(pdb_path)
    arrs.sort(key=lambda a: a[0])
    if args.first_chain_only:
        arrs = arrs[:1]
    sites, resrows = [], []
    for cname, ids, bb, res_atoms in arrs:
        P = pred.residue_probs(bb)
        binding = np.where((np.rint(P) == 1).all(0))[0]
        for i, (c, n, ic) in enumerate(ids):
            resrows.append((c, f"{n}{ic}", float(P[:, i].mean())))
        for s in cloud_sites(res_atoms, binding):
            s["res"] = [C.fmt_res(*ids[i]) for i in s["res"]]
            sites.append(s)
    sites.sort(key=lambda s: -s["size"])
    pockets, points = [], []
    for rk, s in enumerate(sites, 1):
        pockets.append(dict(rank=rk, score=s["size"], center=tuple(s["center"]), residues=s["res"]))
        points += [(rk, *p) for p in s["points"]]
    C.write_pockets(out, stem, pockets, points, info.get("chain_map"))
    cm = info.get("chain_map") or {}
    with open(out / f"{stem}.residues.csv", "w") as f:
        f.write("chain,resnum,mean_prob\n")
        for c, n, p in resrows:
            f.write(f"{cm.get(c, c)},{n},{p:.4f}\n")
    return len(pockets)


def prep_job(job):
    path, args, td = job
    try:
        f, info, tc = C.prepare(path, Path(td), args, prefer="pdb")
        if f.suffix != ".pdb":
            raise RuntimeError("structure not representable as PDB (>99999 atoms / >94 chains)")
        return (C.stem_of(path), f, info, tc, None)
    except Exception as e:
        return (C.stem_of(path), None, {}, 0, str(e)[:300])


def main():
    import tempfile
    ap = C.base_parser(__doc__)
    ap.add_argument("--device", default="cuda:0", help="torch device for ESM-IF1 (use CUDA_VISIBLE_DEVICES to pick a GPU)")
    ap.add_argument("--first-chain-only", action="store_true", help="official behaviour: only first chain")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    with tempfile.TemporaryDirectory(prefix="ifsp_", dir=args.tmp) as td:
        (Path(td) / "in").mkdir()
        prep = C.pmap(prep_job, [(p, args, str(Path(td) / "in")) for p in inputs], args.procs)
        pred = Predictor(args.device)
        rows = []
        for stem, f, info, tc, err in prep:
            if err:
                rows.append(dict(stem=stem, status="error", note=err))
                continue
            t0 = time.time()
            try:
                n = process(pred, f, stem, out, info, args)
                tt = time.time() - t0
                rows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tt, 3),
                                 total_s=round(tc + tt, 3), note=f"{n} pockets", **info))
            except Exception as e:
                import traceback
                rows.append(dict(stem=stem, status="error", note=f"{type(e).__name__}: {e}"[:300]))
                traceback.print_exc()
    C.finish(out, rows)


if __name__ == "__main__":
    main()
