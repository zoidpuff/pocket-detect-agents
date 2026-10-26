#!/usr/bin/env python
"""DeepPocket (devalab/DeepPocket, classification/re-ranking stage) wrapper -> common pocket format.

Run with the tool venv (needs molgrid, torch, gemmi):
  CUDA_VISIBLE_DEVICES=1 baselines/tools/deeppocket/venv/bin/python baselines/run_deeppocket.py <structs|dir|@list> -o OUT -j 8 --device 1

Pipeline (identical to DeepPocket predict.py steps 1-5):
  clean (common.py: no waters/ligands/H/altlocs/nucleic; then only the 20 standard amino acids are kept, as in
  DeepPocket's clean_pdb.py) -> fpocket 4.2.3 default parameters (candidate pockets; -j parallel processes) ->
  candidate center = alpha-sphere barycentre weighted by sphere radius (get_centers.py) ->
  libmolgrid 28-type voxelisation (16 A box, 0.5 A) around each center -> 3D-CNN classifier probability.
Rank/score = CNN probability (descending). Residues / points are those of the fpocket candidate (pocketN_atm.pdb /
pocketN_vert.pqr alpha spheres). The DeepPocket *segmentation* stage is NOT run (its checkpoint is not publicly
retrievable; see README) so residues are fpocket's, not the segmentation mask's.
All structures are scored in one process (model loaded once) on --device.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

DP = C.TOOLS / "deeppocket"
FPOCKET = C.TOOLS / "fpocket_src" / "bin" / "fpocket"
CKPT = DP / "weights" / "first_model_fold1_best_test_auc_85001.pth.tar"
STD_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
          "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def std_aa_only(pdb: Path):
    keep = []
    for l in open(pdb):
        if l.startswith(("ATOM", "HETATM")):
            if l[17:20].strip() in STD_AA:
                keep.append("ATOM  " + l[6:])
        elif l.startswith(("TER", "END")):
            keep.append(l)
    open(pdb, "w").writelines(keep)


def fpocket_stage(job):
    path, args, td = job
    stem = C.stem_of(path)
    t0 = time.time()
    try:
        wd = Path(td) / stem
        wd.mkdir(parents=True)
        f, info, tc = C.prepare(path, wd, args, prefer="pdb")
        if f.suffix != ".pdb":
            raise RuntimeError("structure not representable as PDB (>99999 atoms): DeepPocket/fpocket stage needs PDB")
        std_aa_only(f)
        t1 = time.time()
        C.run_cmd([str(FPOCKET), "-f", str(f), "-w", "p"], timeout=args.timeout, cwd=wd)
        t_fp = time.time() - t1
        od = wd / f"{f.stem}_out" / "pockets"
        cands = []
        n = 1
        while (od / f"pocket{n}_vert.pqr").exists():
            xyz, rad, res = [], [], {}
            for l in open(od / f"pocket{n}_vert.pqr"):
                if l.startswith(("ATOM", "HETATM")):
                    xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
                    rad.append(float(l.split()[-1]))
            for l in open(od / f"pocket{n}_atm.pdb"):
                if l.startswith(("ATOM", "HETATM")):
                    res[(l[21], int(l[22:26]), l[26].strip())] = None
            import numpy as np
            xyz = np.array(xyz)
            rad = np.array(rad)
            bary = (xyz.T * rad).T.sum(0) / rad.sum()  # DeepPocket get_centers.py: radius-weighted barycentre
            cands.append(dict(n=n, center=tuple(float(v) for v in bary), xyz=xyz,
                              residues=[C.fmt_res(c, r, i) for (c, r, i) in res]))
            n += 1
        return dict(stem=stem, pdb=str(f), info=info, cands=cands, t_clean=tc, t_fp=t_fp, error=None)
    except Exception as e:
        return dict(stem=stem, error=str(e)[:300], t_fp=time.time() - t0)


def main():
    ap = C.base_parser(__doc__)
    ap.add_argument("--device", type=int, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES before torch import)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-structure fpocket timeout (s)")
    ap.add_argument("--batch", type=int, default=50, help="candidate pockets per CNN batch")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    inputs.sort(key=lambda p: -p.stat().st_size)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import molgrid
    import struct
    sys.path.insert(0, str(DP / "src"))
    from model import Model

    with tempfile.TemporaryDirectory(prefix="dp_", dir=args.tmp) as td:
        # stage 1: clean + fpocket, parallel over CPU processes
        res1 = C.pmap(fpocket_stage, [(p, args, td) for p in inputs], args.procs)
        # stage 2: CNN scoring on GPU, one process
        t = time.time()
        model = Model()
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"])
        dev = "cuda" if torch.cuda.is_available() else None
        if dev is None:
            raise RuntimeError("DeepPocket needs a GPU (libmolgrid gridding on CUDA tensors)")
        model.to(dev).eval()
        t_load = time.time() - t
        typer = molgrid.FileMappedGninaTyper(str(DP / "src" / "gninamap"))
        gmaker = molgrid.GridMaker()
        rows = []
        for r in res1:
            stem = r["stem"]
            if r["error"]:
                rows.append(dict(stem=stem, status="error", note=r["error"], total_s=round(r["t_fp"], 2)))
                continue
            try:
                t2 = time.time()
                cands = r["cands"]
                wd = Path(r["pdb"]).parent
                chain_map = r["info"].get("chain_map")
                if not cands:
                    C.write_pockets(out, stem, [], [], chain_map)
                    rows.append(dict(stem=stem, status="ok", clean_s=round(r["t_clean"], 3), tool_s=round(r["t_fp"], 3),
                                     total_s=round(r["t_clean"] + r["t_fp"], 3), note="0 pockets (fpocket found none)", **r["info"]))
                    continue
                # gninatype of the protein (same as DeepPocket gninatype())
                (wd / "p.types").write_text(r["pdb"] + "\n")
                ep0 = molgrid.ExampleProvider(typer, shuffle=False, default_batch_size=1)
                ep0.populate(str(wd / "p.types"))
                ex = ep0.next()
                coords = ex.coord_sets[0].coords.tonumpy()
                types = np.int_(ex.coord_sets[0].type_index.tonumpy())
                gt = wd / "p.gninatypes"
                with open(gt, "wb") as fo:
                    for i in range(coords.shape[0]):
                        fo.write(struct.pack("fffi", coords[i][0], coords[i][1], coords[i][2], int(types[i])))
                with open(wd / "bary.types", "w") as fo:
                    for c in cands:
                        fo.write(f"{c['n']} {c['center'][0]} {c['center'][1]} {c['center'][2]} {gt}\n")
                bs = min(len(cands), args.batch)
                ep = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0, balanced=False,
                                             iteration_scheme=molgrid.IterationScheme.LargeEpoch, default_batch_size=bs)
                ep.populate(str(wd / "bary.types"))
                dims = gmaker.grid_dimensions(ep.num_types())
                inp = torch.zeros((bs,) + dims, dtype=torch.float32, device=dev)
                lab = torch.zeros((bs, 4), dtype=torch.float32, device=dev)
                probs = []
                nb = (len(cands) + bs - 1) // bs
                with torch.no_grad():
                    for _ in range(nb):
                        batch = ep.next_batch(bs)
                        batch.extract_labels(lab)
                        cen = lab[:, 1:].cpu()
                        for b in range(bs):
                            gmaker.forward(molgrid.float3(*[float(v) for v in cen[b]]), batch[b].coord_sets[0], inp[b])
                        out_ = model(inp[:, :14])  # first 14 channels = protein
                        probs.append(F.softmax(out_, dim=1)[:, 1].cpu())
                probs = torch.cat(probs)[: len(cands)].numpy()
                order = sorted(range(len(cands)), key=lambda i: (-probs[i], cands[i]["n"]))
                pockets, points = [], []
                for rk, i in enumerate(order, 1):
                    c = cands[i]
                    pockets.append(dict(rank=rk, score=float(probs[i]), center=c["center"], residues=c["residues"]))
                    points += [(rk, *map(float, v)) for v in c["xyz"]]
                C.write_pockets(out, stem, pockets, points, chain_map)
                if args.keep_raw:
                    import shutil
                    shutil.copytree(wd, out / "raw" / stem, dirs_exist_ok=True)
                t_cnn = time.time() - t2
                rows.append(dict(stem=stem, status="ok", clean_s=round(r["t_clean"], 3),
                                 tool_s=round(r["t_fp"] + t_cnn, 3), total_s=round(r["t_clean"] + r["t_fp"] + t_cnn, 3),
                                 note=f"{len(cands)} pockets; fpocket {r['t_fp']:.2f}s cnn {t_cnn:.2f}s (model load {t_load:.1f}s once)",
                                 **r["info"]))
            except Exception as e:
                import traceback
                traceback.print_exc()
                rows.append(dict(stem=stem, status="error", note=f"{type(e).__name__}: {e}"[:300]))
    C.finish(out, rows)


if __name__ == "__main__":
    main()
