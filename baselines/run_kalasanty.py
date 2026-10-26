#!/usr/bin/env python
"""Kalasanty (Stepniewska-Dziubinska et al., Sci Rep 2020; gitlab.com/cheminfIBB/kalasanty @aa91525, weights
data/model_scpdb2017.hdf) wrapper -> common pocket format.

  python baselines/run_kalasanty.py <structs|dir|@list> -o OUT [--device 0] [--no-tile] [--thr 0.5] [--min-size 50]

Kalasanty = 3D U-Net on an 18-channel atom-feature grid (2 A voxels, box = 72 A centred on the protein centroid)
predicting a pocket-probability map; pockets = connected components of prob > thr (default 0.5) after closing,
border clearing and removal of < 50 A^3; residues = whole residues with an atom within 4.5 A of a pocket voxel.
Modernisation: original needs TF1/Keras 2.2 + OpenBabel 2; here TF 2.15 (tf.keras) + openbabel-wheel 3.1 with
the weights loaded by topology into an identical UNet (see baselines/tools/kalasanty). X11 libs needed by the
OpenBabel wheel are stubbed (tools/kalasanty/shim/lib).
Preprocessing (original expects protonated protein mol2 with charges, no waters/ligands): our gemmi-cleaned PDB is
read by OpenBabel, hydrogens added at pH 7.4 (AddHydrogens(polaronly=False, correctForPH=True)), and the 'partial
charge' feature is set to the formal charge (sc-PDB mol2 files are 'NO_CHARGES' with formal charges only).
Large proteins: the official code only sees the central 72 A box; by default we tile overlapping boxes (36 A stride,
tile borders masked, max-merge) when the protein extent exceeds the box; --no-tile gives the official behaviour.
Score (OUR addition; the tool has no scores): sum of the pocket probability over the pocket's voxels (probability
mass, A^3-weighted: sum(p) * 8). Center = centroid of pocket voxels. Points = voxel centres (rank,x,y,z).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

TOOL = C.TOOLS / "kalasanty"
SHIM = TOOL / "shim"
# OpenBabel wheel needs stub X11 libs on LD_LIBRARY_PATH *before* the process starts -> re-exec with env
_py = TOOL / "venv" / "bin" / "python"
if str(SHIM / "lib") not in os.environ.get("LD_LIBRARY_PATH", "") or Path(sys.prefix).resolve() != (TOOL / "venv").resolve():
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{SHIM / 'lib'}:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{SHIM}:{TOOL / 'src'}"
    os.execve(str(_py), [str(_py), *sys.argv], env)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

SCALE, MAX_DIST = 0.5, 35.0


class Predictor:
    def __init__(self, args):
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
        tf.get_logger().setLevel("ERROR")
        from kalasanty.data import get_box_size
        from kalasanty.net import UNet
        from tfbio.data import Featurizer, make_grid
        self.make_grid = make_grid
        self.feat = Featurizer(save_molecule_codes=False)
        self.box = get_box_size(SCALE, MAX_DIST)  # 36 voxels
        self.net = UNet(box_size=self.box, input_channels=len(self.feat.FEATURE_NAMES), output_channels=1,
                        scale=SCALE, featurizer=self.feat)
        self.net.max_dist = MAX_DIST
        self.net.load_weights(str(TOOL / "src" / "data" / "model_scpdb2017.hdf"))
        self.res = 1.0 / SCALE

    def density(self, coords, feats, tile):
        """global pocket-probability array [X,Y,Z] and its origin (A)."""
        res, box = self.res, self.box
        lo, hi = coords.min(0), coords.max(0)
        centroid = coords.mean(0)
        if not tile or (hi - lo).max() <= 2 * MAX_DIST - 4:
            # official single-box behaviour
            x = self.make_grid(coords - centroid, feats, max_dist=MAX_DIST, grid_resolution=res)
            d = self.net.predict(x, verbose=0)[0, ..., 0]
            return d, centroid - MAX_DIST
        stride = box // 2
        o0 = lo - 6.0
        n = np.ceil((hi + 6.0 - o0) / res).astype(int)
        nt = np.maximum(1, np.ceil((n - box) / stride).astype(int) + 1)
        G = np.zeros(n + box, dtype=np.float32)
        border = 3
        for ix in range(nt[0]):
            for iy in range(nt[1]):
                for iz in range(nt[2]):
                    idx = np.array([ix, iy, iz]) * stride
                    origin = o0 + idx * res
                    centre = origin + MAX_DIST
                    m = ((coords > origin - 2) & (coords < origin + box * res + 2)).all(1)
                    if m.sum() < 5:
                        continue
                    x = self.make_grid(coords[m] - centre, feats[m], max_dist=MAX_DIST, grid_resolution=res)
                    d = self.net.predict(x, verbose=0)[0, ..., 0].copy()
                    mask = np.ones_like(d, bool)
                    for a in range(3):
                        sl = [slice(None)] * 3
                        if idx[a] > 0:
                            sl[a] = slice(0, border); mask[tuple(sl)] = False
                        if ix * 0 == 0 and [ix, iy, iz][a] < nt[a] - 1:
                            sl = [slice(None)] * 3
                            sl[a] = slice(box - border, box); mask[tuple(sl)] = False
                    sub = G[idx[0]:idx[0] + box, idx[1]:idx[1] + box, idx[2]:idx[2] + box]
                    np.maximum(sub, np.where(mask, d, 0), out=sub)
        return G, o0


def load_mol(path):
    from openbabel import pybel
    from openbabel import openbabel as ob
    mol = next(pybel.readfile("pdb", str(path)))
    # residue identity must be recorded BEFORE AddHydrogens (OpenBabel re-perceives/renames chains afterwards)
    ids = []
    for a in mol.atoms:
        r = a.residue.OBResidue
        ids.append((r.GetChain(), r.GetNum(), r.GetInsertionCode().strip("\x00 ")))
    n0 = len(ids)
    mol.OBMol.AddHydrogens(False, True, 7.4)
    for a in mol.atoms:
        a.OBAtom.SetPartialCharge(float(a.formalcharge))
        if a.idx > n0:  # added hydrogen: residue of its heavy neighbour
            nb = next(ob.OBAtomAtomIter(a.OBAtom))
            ids.append(ids[nb.GetIdx() - 1])
    mol.OBMol.SetPartialChargesPerceived()
    return mol, ids


def process(pred, path, stem, out, info, args):
    from scipy.spatial import cKDTree
    mol, res_of = load_mol(path)
    coords, feats = pred.feat.get_features(mol)
    atoms = np.array([a.coords for a in mol.atoms])
    uniq = {k: i for i, k in enumerate(dict.fromkeys(res_of))}
    atom_res = np.array([uniq[k] for k in res_of])
    keys = list(uniq)
    G, origin = pred.density(coords.astype(np.float64), feats, not args.no_tile)
    lab = pred.net.get_pockets_segmentation(G[None, ..., None], threshold=args.thr, min_size=args.min_size)
    step = pred.res
    pockets, points = [], []
    tree = cKDTree(atoms)
    for l in range(1, lab.max() + 1):
        idx = np.argwhere(lab == l)
        if len(idx) == 0:
            continue
        xyz = idx * step + origin
        mass = float(G[tuple(idx.T)].sum()) * step ** 3
        near = set()
        for lst in tree.query_ball_point(xyz, 4.5):
            near.update(lst)
        if not near:
            continue
        rs = sorted(set(atom_res[list(near)].tolist()))
        pockets.append(dict(score=mass, center=tuple(xyz.mean(0)), residues=[C.fmt_res(*keys[i]) for i in rs], xyz=xyz))
    pockets.sort(key=lambda p: -p["score"])
    for rk, p in enumerate(pockets, 1):
        p["rank"] = rk
        points += [(rk, *v) for v in p["xyz"]]
    C.write_pockets(out, stem, pockets, points, info.get("chain_map"))
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
    ap.add_argument("--no-tile", action="store_true", help="official behaviour: only the central 72 A box")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-size", type=float, default=50.0, help="min pocket volume (A^3)")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    with tempfile.TemporaryDirectory(prefix="kal_", dir=args.tmp) as td:
        (Path(td) / "in").mkdir()
        prep = C.pmap(prep_job, [(p, args, str(Path(td) / "in")) for p in inputs], args.procs)
        pred = Predictor(args)
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
                traceback.print_exc()
                rows.append(dict(stem=stem, status="error", note=f"{type(e).__name__}: {e}"[:300]))
    C.finish(out, rows)


if __name__ == "__main__":
    main()
