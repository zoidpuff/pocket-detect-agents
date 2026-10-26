#!/usr/bin/env python
"""Run BURROW on AF-dev models under an AF-mode input variant (pocketdetect.burrow.af_variants).

  python scripts/af_dev_predict.py --config prune_soft [--model M.json] [--af-ranker R.json] \
      [--sites results/dev_af/af_dev_sites_all.tsv] [--name NAME] [--jobs 24]

Predictions: results/dev_af/preds/<name>/<structure_id>.csv (+ _stats.csv: removed residues/atoms,
soft atoms, ms).  ``--af-ranker`` re-ranks the candidates with an AF-aware ranker
(pocketdetect.burrow.af_rank) instead of the model's standard ranker.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.af_variants import CONFIGS, af_input  # noqa: E402
from pocketdetect.burrow.model import BurrowModel  # noqa: E402
from pocketdetect.burrow.pipeline import predict_protein, write_prediction  # noqa: E402
from pocketdetect.sites import read_sites  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

STD_MODEL = ROOT / "results/dev/biolip/models/burrow_all_s0.json"
_G = {}


def _job(args):
    sid, path = args
    try:
        t0 = time.perf_counter()
        s = load_structure(path, structure_id=sid, protein_only=True)
        P, w, st = af_input(s, _G["cfg"])
        if _G.get("folds") is not None:  # out-of-fold point scores + standard ranker (AF train supplement)
            from pocketdetect.burrow.af_rank import candidate_table
            from pocketdetect.burrow.model import FormationConfig, score_points
            from pocketdetect.burrow.pipeline import Prediction, finalize
            import numpy as np
            model = _G["model"]
            fb = _G["folds"][_G["fold_of"][sid]]
            pk, ps, score, F, names = candidate_table(P, model, w, score_override=lambda ps: score_points(
                fb, ps.feats, ps.names, model.point_features, model.meta.get("score_floor")))
            afr = _G["af_ranker"]
            if isinstance(afr, list):  # cluster-CV rankers, one per fold
                afr = afr[_G["fold_of"][sid]]
            if len(pk):
                rb = afr.booster() if afr is not None else model.boosters()[1]
                feats = afr.rank_features if afr is not None else model.rank_features
                rs = rb.predict(F[:, [names.index(n) for n in feats]], num_threads=1)
            else:
                rs = np.zeros(0)
            keep = finalize(pk, rs, FormationConfig(**model.formation))
            pred = Prediction(sid, np.array([pk[i].center for i in keep]).reshape(-1, 3), np.asarray(rs)[keep],
                              [pk[i].residues for i in keep], [], {})
        elif _G["af_ranker"] is not None:
            from pocketdetect.burrow.af_rank import predict_protein_af
            pred = predict_protein_af(P, _G["model"], _G["af_ranker"], weights=w)
        else:
            pred = predict_protein(P, _G["model"], weights=w)
        write_prediction(_G["out"], pred, points=False)
        return dict(structure_id=sid, status="ok", n_pockets=len(pred.scores), ms=(time.perf_counter() - t0) * 1e3, **st)
    except Exception as ex:  # record, never abort
        import traceback
        return dict(structure_id=sid, status=f"error: {type(ex).__name__}: {ex} {traceback.format_exc()[-300:]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--const-plddt", type=float, default=None, help="overwrite pLDDT (no-op check on PDB input)")
    ap.add_argument("--model", type=Path, default=STD_MODEL)
    ap.add_argument("--af-ranker", type=Path, default=None)
    ap.add_argument("--sites", type=Path, default=ROOT / "results/dev_af/af_dev_sites_all.tsv")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-root", type=Path, default=ROOT / "results/dev_af/preds")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--oof", action="store_true", help="score points with the cross-fitting fold model of each "
                    "entry's cluster (AF *train* models; needs BURROW_TRAINSET as for af_dev_train_ranker.py)")
    a = ap.parse_args(argv)
    cfg = CONFIGS[a.config]
    if a.const_plddt is not None:
        cfg = replace(cfg, const_plddt=a.const_plddt)
    name = a.name or a.config
    out = a.out_root / name
    out.mkdir(parents=True, exist_ok=True)
    st = read_sites(a.sites).drop_duplicates("structure_id")
    tasks = list(zip(st.structure_id, st.structure_path))
    model = BurrowModel.load(a.model)
    model.boosters()
    afr = None
    if a.af_ranker is not None:
        from pocketdetect.burrow.af_rank import AFRanker
        if "{k}" in str(a.af_ranker):  # per-fold CV rankers (only with --oof)
            assert a.oof, "--af-ranker with {k} needs --oof"
            afr = [AFRanker.load(str(a.af_ranker).replace("{k}", str(k))) for k in range(5)]
        else:
            afr = AFRanker.load(a.af_ranker)
    _G.update(cfg=cfg, model=model, af_ranker=afr, out=out)
    if a.oof:
        sys.path.insert(0, str(ROOT / "scripts"))
        from af_dev_train_ranker import fold_models
        folds, fold_g, _, _ = fold_models(model, a.model, a.jobs)
        grp = dict(zip(st.structure_id, st.group))
        tasks = [t for t in tasks if grp[t[0]] in fold_g]
        _G.update(folds=folds, fold_of={s: fold_g[grp[s]] for s, _ in tasks})
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("fork")) as ex:
        rows = list(ex.map(_job, tasks, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(out / "_stats.csv", index=False)
    bad = df[df.status != "ok"]
    print(f"{name}: {len(df) - len(bad)}/{len(df)} ok in {time.time() - t0:.0f}s -> {out}"
          + (f"\n{bad.head().to_string()}" if len(bad) else ""))


if __name__ == "__main__":
    main()
