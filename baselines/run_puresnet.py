#!/usr/bin/env python
"""PUResNetV2.0 (jivankandel/PUResNetV2.0) wrapper -> common pocket format.

Run with the tool venv (torch, openbabel-wheel, scikit-learn, gemmi); the script re-execs itself with the
LD_LIBRARY_PATH stubs needed by the openbabel wheel (libX11/libXrender are absent in the container):
  baselines/tools/puresnet/venv/bin/python baselines/run_puresnet.py <structs|dir|@list> -o OUT -j 8 --device 1

The original code needs MinkowskiEngine (CUDA build, impossible here without nvcc); this wrapper instead uses
baselines/pv2_torch.py, a pure-PyTorch re-implementation of the sparse-conv ops with the *same checkpoint* (weights
load strictly; sanity check: atom-level AUC 0.93 on the 12 example proteins shipped with the repo, up to 0.997 per protein).
Pipeline = repo's `puresnet.predict.save_predictions` mode 'A' (whole structure as one entity):
  parse (repo S_parser; residue features from RCSB CCD ideal SDFs via openbabel, cached in tools/puresnet/residues)
  -> atoms rounded to 1 A voxels (duplicates dropped) -> sparse ResNet -> per-atom probability
  -> atoms with p >= 0.34 -> DBSCAN(eps=6 A, min_samples=5) -> clusters = pockets (clusters of size 1 skipped).
The tool defines no pocket ranking; here score = sum of atom probabilities in the cluster (--score mean|max|sum).
center = mean xyz of the cluster atoms; residues = residues of cluster atoms; points = the cluster atoms' coordinates.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUR = HERE / "tools" / "puresnet"
_stub = str(PUR / "stubs")
if _stub not in os.environ.get("LD_LIBRARY_PATH", ""):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = _stub + ":" + env.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import subprocess
import tempfile
import time

sys.path.insert(0, str(HERE))
import common as C

CCD = PUR / "residues"
STD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
          "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
THRESH = 0.34


def ensure_ccd(names):
    """download <RES>.cif and <RES>_ideal.sdf from RCSB into the residue cache; return set of names that failed"""
    CCD.mkdir(exist_ok=True)
    bad = set()
    for n in names:
        for fn, url in ((f"{n}.cif", f"https://files.rcsb.org/ligands/download/{n}.cif"),
                        (f"{n}_ideal.sdf", f"https://files.rcsb.org/ligands/download/{n}_ideal.sdf")):
            p = CCD / fn
            if p.exists() and p.stat().st_size > 100:
                continue
            tmp = CCD / (fn + f".tmp{os.getpid()}")
            subprocess.run(["curl", "-sfL", "-m", "60", "-o", str(tmp), url], capture_output=True)
            if tmp.exists() and tmp.stat().st_size > 100:
                tmp.rename(p)
            else:
                tmp.unlink(missing_ok=True)
                bad.add(n)
    return bad


def parse_stage(job):
    """clean + featurise one structure (CPU)."""
    path, args, td = job
    stem = C.stem_of(path)
    t0 = time.time()
    try:
        import numpy as np
        wd = Path(td) / stem
        wd.mkdir(parents=True)
        f, info, tc = C.prepare(path, wd, args, prefer="pdb")
        if f.suffix != ".pdb":
            raise RuntimeError("structure not representable as PDB (>99999 atoms or >94 chains)")
        names = {l[17:20].strip() for l in open(f) if l.startswith(("ATOM", "HETATM"))}
        bad = ensure_ccd(sorted(names - STD_AA))
        ensure_ccd(sorted(names & STD_AA))
        if bad:  # drop residues the tool cannot featurise (no CCD entry)
            keep = [l for l in open(f) if not (l.startswith(("ATOM", "HETATM")) and l[17:20].strip() in bad)]
            open(f, "w").writelines(keep)
        from puresnet.sparse import get_data, get_coordinates_features
        vd, chains = get_data(str(f))
        if vd is None:
            raise RuntimeError("tool parser failed (unknown residue / atom error)")
        coords, feats, information = get_coordinates_features(vd, chains)
        # tool convention: voxel = round(xyz - min); keep original xyz in `information`
        return dict(stem=stem, info=info, coords=coords.numpy(), feats=feats.numpy(), information=information,
                    t_clean=tc, t_parse=time.time() - t0 - tc, dropped=sorted(bad), error=None)
    except Exception as e:
        return dict(stem=stem, error=f"{type(e).__name__}: {e}"[:300], t_parse=time.time() - t0)


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--device", type=int, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--score", default="sum", choices=["sum", "mean", "max"])
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    out = Path(args.out).resolve()
    args.out = str(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = [p.resolve() for p in C.todo(C.expand_inputs(args.inputs), out, args.overwrite)]
    if not inputs:
        print("nothing to do")
        return
    CCD.mkdir(exist_ok=True)
    os.chdir(PUR)  # the tool's residue_h module caches CCD files in $PWD/residues at import time
    sys.path.insert(0, str(PUR / "src"))
    import numpy as np
    import torch
    from sklearn.cluster import DBSCAN
    import pv2_torch as P

    inputs.sort(key=lambda p: -p.stat().st_size)
    with tempfile.TemporaryDirectory(prefix="pur_", dir=args.tmp) as td:
        res1 = C.pmap(parse_stage, [(p, args, td) for p in inputs], args.procs)
        t = time.time()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = P.PUResNetV2()
        model.load_state_dict(torch.load(PUR / "src" / "puresnet" / "data" / "model90 70210.pt", map_location="cpu"))
        model.eval().to(dev)
        t_load = time.time() - t
        rows = []
        for r in res1:
            stem = r["stem"]
            if r["error"]:
                rows.append(dict(stem=stem, status="error", note=r["error"], total_s=round(r["t_parse"], 2)))
                continue
            try:
                t2 = time.time()
                coords = r["coords"].astype(np.int64)
                logits = P.forward_atoms(model, coords, r["feats"], dev)
                prob = torch.sigmoid(logits).cpu().numpy()
                if dev == "cuda":
                    torch.cuda.synchronize()
                t_inf = time.time() - t2
                info_arr = r["information"]
                sel = prob >= THRESH
                pockets, points = [], []
                if sel.sum() >= 2:
                    xyz = np.asarray(info_arr[sel][:, -3:], dtype=float)
                    pr = prob[sel]
                    resid = info_arr[sel]
                    lbl = DBSCAN(eps=6, algorithm="kd_tree", leaf_size=100, min_samples=5).fit(xyz).labels_
                    cl = []
                    for a in np.unique(lbl):
                        if a == -1:
                            continue
                        m = lbl == a
                        if m.sum() == 1:
                            continue
                        pp = pr[m]
                        sc = {"sum": pp.sum(), "mean": pp.mean(), "max": pp.max()}[args.score]
                        resi = sorted({C.fmt_res(x[5], x[4][:-1] if x[4][-1:].isalpha() else x[4],
                                                 x[4][-1] if x[4][-1:].isalpha() else "") for x in resid[m]})
                        cl.append((float(sc), xyz[m].mean(0), resi, xyz[m]))
                    cl.sort(key=lambda c: -c[0])
                    for rk, (sc, cen, resi, pts) in enumerate(cl, 1):
                        pockets.append(dict(rank=rk, score=sc, center=tuple(cen), residues=resi))
                        points += [(rk, *map(float, p)) for p in pts]
                C.write_pockets(out, stem, pockets, points, r["info"].get("chain_map"))
                t_post = time.time() - t2 - t_inf
                note = f"{len(pockets)} pockets; parse {r['t_parse']:.2f}s gpu {t_inf:.2f}s post {t_post:.2f}s (model load {t_load:.1f}s once); {len(prob)} atoms"
                if r["dropped"]:
                    note += f"; dropped residues w/o CCD: {','.join(r['dropped'])}"
                rows.append(dict(stem=stem, status="ok", clean_s=round(r["t_clean"], 3),
                                 tool_s=round(r["t_parse"] + t_inf + t_post, 3),
                                 total_s=round(r["t_clean"] + r["t_parse"] + t_inf + t_post, 3), note=note, **r["info"]))
            except Exception as e:
                import traceback
                traceback.print_exc()
                rows.append(dict(stem=stem, status="error", note=f"{type(e).__name__}: {e}"[:300]))
    C.finish(out, rows)


if __name__ == "__main__":
    main()
