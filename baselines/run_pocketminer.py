#!/usr/bin/env python
"""PocketMiner (Meller et al., Nat Commun 2023; github.com/Mickdub/gvp branch pocket_pred @187062d) wrapper.

  python baselines/run_pocketminer.py <structs|dir|@list> -o OUT [--thr 0.5] [--link 8.0] [--device 0]

PocketMiner is a GVP-GNN predicting, per residue, the probability that the residue takes part in a CRYPTIC pocket
opening (trained on MD simulation outcomes), it is NOT a general ligand-binding-site predictor.
Input: backbone N,CA,C,O of all protein residues (all chains together in one k-NN graph, k=30) + residue types;
residues missing any backbone atom or with a non-standard/unknown residue type are skipped (reported in note).
Official demo script feeds one whole structure (all chains) -> we do the same, batch size 1, model loaded once.
Pocket extraction (OUR addition, not part of PocketMiner): residues with p >= --thr (default 0.5) are clustered by
single linkage on CA-CA distance <= --link (8 A); pocket score = sum of residue probabilities, centre = mean CA of
cluster residues; residues = cluster residues; no point file. Pockets are ranked by score.
Extra output <stem>.residues.csv: chain,resnum,prob (all residues, raw PocketMiner output).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

TOOL = C.TOOLS / "pocketminer"
C.reexec_in_venv(TOOL / "venv")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

LOOKUP = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
          'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}


def backbone_arrays(path):
    """X [L,4,3] (N,CA,C,O), S [L], residue ids [(chain,num,icode)]; skips incomplete / unknown residues."""
    import gemmi
    st = gemmi.read_structure(str(path))
    X, S, ids, skipped = [], [], [], 0
    for ch in st[0]:
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            one = info.one_letter_code.upper() if info is not None and info.is_amino_acid() else None
            d = {a.name: a.pos for a in r}
            if one not in LOOKUP or not all(n in d for n in ("N", "CA", "C", "O")):
                skipped += 1
                continue
            X.append([[d[n].x, d[n].y, d[n].z] for n in ("N", "CA", "C", "O")])
            S.append(LOOKUP[one])
            ids.append((ch.name, r.seqid.num, r.seqid.icode.strip()))
    return np.array(X, dtype=np.float32), np.array(S, dtype=np.int32), ids, skipped


class Predictor:
    def __init__(self):
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
        sys.path.insert(0, str(TOOL / "src" / "src"))
        from models import MQAModel
        from util import load_checkpoint
        self.tf = tf
        tf.get_logger().setLevel('ERROR')
        self.model = MQAModel(node_features=(8, 50), edge_features=(1, 32), hidden_dim=(16, 100),
                              num_layers=4, dropout=0.1)
        # build variables by one forward pass, then restore (same as official predict_on_xtals)
        load_checkpoint(self.model, tf.keras.optimizers.legacy.Adam(), str(TOOL / "src" / "models" / "pocketminer"))

    def __call__(self, X, S):
        tf = self.tf
        mask = np.ones((1, len(S)), dtype=np.float32)
        p = self.model(X[None], S[None], mask, train=False, res_level=True)
        return np.asarray(p)[0]


def cluster(ca, probs, thr, link):
    from scipy.cluster.hierarchy import fcluster, linkage
    idx = np.where(probs >= thr)[0]
    if len(idx) == 0:
        return []
    if len(idx) == 1:
        labs = np.array([1])
    else:
        labs = fcluster(linkage(ca[idx], method="single"), t=link, criterion="distance")
    out = []
    for l in set(labs):
        m = idx[labs == l]
        out.append(dict(res=m, score=float(probs[m].sum()), center=ca[m].mean(0)))
    out.sort(key=lambda s: -s["score"])
    return out


def process(pred, path, stem, out, info, args):
    X, S, ids, skipped = backbone_arrays(path)
    if len(S) < 5:
        raise RuntimeError("too few complete residues")
    p = pred(X, S)
    cm = info.get("chain_map") or {}
    with open(out / f"{stem}.residues.csv", "w") as f:
        f.write("chain,resnum,prob\n")
        for (c, n, ic), v in zip(ids, p):
            f.write(f"{cm.get(c, c)},{n}{ic},{v:.4f}\n")
    pockets = []
    for rk, s in enumerate(cluster(X[:, 1], p, args.thr, args.link), 1):
        pockets.append(dict(rank=rk, score=s["score"], center=tuple(s["center"]),
                            residues=[C.fmt_res(*ids[i]) for i in s["res"]]))
    C.write_pockets(out, stem, pockets, None, cm)
    return len(pockets), skipped


def prep_job(job):
    path, args, td = job
    try:
        f, info, tc = C.prepare(path, Path(td), args, prefer="pdb")
        return (C.stem_of(path), f, info, tc, None)
    except Exception as e:
        return (C.stem_of(path), None, {}, 0, str(e)[:300])


def main():
    import tempfile
    ap = C.base_parser(__doc__)
    ap.add_argument("--thr", type=float, default=0.5, help="residue probability threshold for pocket extraction")
    ap.add_argument("--link", type=float, default=8.0, help="single-linkage CA-CA distance (A) for clustering")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = C.todo(C.expand_inputs(args.inputs), out, args.overwrite)
    if not inputs:
        print("nothing to do")
        return
    with tempfile.TemporaryDirectory(prefix="pm_", dir=args.tmp) as td:
        (Path(td) / "in").mkdir()
        prep = C.pmap(prep_job, [(p, args, str(Path(td) / "in")) for p in inputs], args.procs)
        pred = Predictor()
        rows = []
        for stem, f, info, tc, err in prep:
            if err:
                rows.append(dict(stem=stem, status="error", note=err))
                continue
            t0 = time.time()
            try:
                n, sk = process(pred, f, stem, out, info, args)
                tt = time.time() - t0
                rows.append(dict(stem=stem, status="ok", clean_s=round(tc, 3), tool_s=round(tt, 3),
                                 total_s=round(tc + tt, 3), note=f"{n} pockets; {sk} residues skipped", **info))
            except Exception as e:
                import traceback
                traceback.print_exc()
                rows.append(dict(stem=stem, status="error", note=f"{type(e).__name__}: {e}"[:300]))
    C.finish(out, rows)


if __name__ == "__main__":
    main()
