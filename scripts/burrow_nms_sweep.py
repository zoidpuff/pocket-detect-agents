#!/usr/bin/env python
"""NMS / output-size sensitivity curve on the BioLiP2 dev set for a stored BURROW run.

For each setting (centre-distance radius, point-overlap threshold, score floor) re-emit the dev
predictions and evaluate LIGYSIS top-N / top-N+2 (DCC 12), DCA4 top-N+2 (closest-by-DCA and any),
P1K and #pockets.  Selection criterion: maximise mean(recall_topn2, recall_ldca_topn2, P1K_1to1) — DCC12 and DCA4
recall and one-to-one pooled precision weighted equally.  (First version used raw P1K; the sweep
showed raw P1K rewards duplicate pockets on the same site (89.5 % at 4 Å NMS), so the one-to-one
variant is used for selection; both are reported.)

  python scripts/burrow_nms_sweep.py burrow_all_s0 --variant all
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import pickle
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from burrow_train_dev import emit  # noqa: E402
from pocketdetect.burrow.model import FormationConfig  # noqa: E402
from pocketdetect.site_eval import evaluate_sites, summarize_sites  # noqa: E402
from pocketdetect.sites import read_sites, sites_by_structure  # noqa: E402

GRID = [dict(nms_dist=d, nms_overlap=o, min_score=ms)
        for d in (4.0, 6.0, 8.0, 10.0, 12.0, 14.0) for o in (0.3,) for ms in (None, 0.05, 0.2)] + \
       [dict(nms_dist=8.0, nms_overlap=o, min_score=None) for o in (0.1, 0.5, None)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--variant", choices=["all", "small"], default="all")
    ap.add_argument("--n-boot", type=int, default=0)
    a = ap.parse_args(argv)
    store = pickle.loads((ROOT / "results/dev/cache/runs" / f"{a.run}.pkl").read_bytes())
    by = sites_by_structure(read_sites(ROOT / f"results/dev/biolip/dev_sites_{a.variant}.tsv"))
    groups = {sid: ss[0].meta.get("group") for sid, ss in by.items()}
    tmp = ROOT / "results/dev/cache/sweep" / a.run
    rows = []
    for g in GRID:
        fc = replace(FormationConfig(), **g)
        if tmp.exists():
            shutil.rmtree(tmp)
        for sid, (light, rs, ss, y) in store.items():
            if sid in by:
                emit(tmp, sid, light, rs, fc)
        r = evaluate_sites(by, str(tmp), groups=groups)
        s = summarize_sites(r, n_boot=a.n_boot).set_index("metric")["value"]
        row = dict(**{k: v for k, v in g.items()}, pockets=r.pockets.groupby("structure_id").size().mean(),
                   **{m: s[m] for m in ("recall_topn", "recall_topn2", "recall_ldca_topn2", "recall_anyp_dca_topn2",
                                        "recall_max", "P1K", "P1K_1to1")})
        row["criterion"] = (row["recall_topn2"] + row["recall_ldca_topn2"] + row["P1K_1to1"]) / 3
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / f"results/dev/biolip/nms_sweep_{a.run}.csv"
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print("best:", df.loc[df.criterion.idxmax()].to_dict())


if __name__ == "__main__":
    main()
