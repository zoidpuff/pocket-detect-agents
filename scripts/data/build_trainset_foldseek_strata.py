#!/usr/bin/env python
"""Fold-novelty strata and fold-disjoint ablation subset (DESIGN §3 strata, §9 fold-disjoint ablation).

Foldseek (tools/bin/foldseek) search of every structure-backed test chain against every training/dev
representative chain (protein heavy atoms of the deployed input). Score = max(qTM, tTM) over hits
with E < 1e-3 (alignment-type 2 = 3Di+AA; TM-scores from the alignment).

Outputs
  data/manifests/test_fold_novelty.tsv   id (test_union id), source_set, max_tm_to_train, best_train_entry, stratum
                                         (<0.5 / 0.5-0.7 / >0.7; 'no_hit' counts as <0.5)
  data/processed/train/fold_to_test.tsv  entry_id, split, max_tm_to_test, best_test_id, fold_disjoint (max_tm < 0.5)
  data/processed/train/foldseek_counts.json

Test structures: PDB chains from the local mirror (LIGYSIS, COACH420/HOLO4K entities, CryptoBench,
PocketMiner, dimer set), PocketWatch test ESMFold2 folds, AF-set models if present under
data/processed/af (searched recursively for AF-<acc>-F1-model_v6.cif.gz).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.structure import pdb_entry_path  # noqa: E402

FOLDSEEK = ROOT / "tools" / "bin" / "foldseek"
MAN = ROOT / "data" / "manifests"
PROC = ROOT / "data" / "processed"
FS = PROC / "train" / "work" / "foldseek"


def write_chain(args):
    src, chain, out = args
    try:
        st = gemmi.read_structure(str(src))
        st.setup_entities()
        st.remove_hydrogens()
        st.remove_alternative_conformations()
        st.remove_ligands_and_waters()
        model = st[0]
        for ch in list(model):
            if chain is not None and ch.name != chain:
                model.remove_chain(ch.name)
        for i in range(len(st) - 1, 0, -1):
            del st[i]
        if sum(len(c) for c in st[0]) < 10:
            return out, "too_small"
        st.write_pdb(str(out), gemmi.PdbWriteOptions(minimal=True))
        return out, None
    except Exception as e:  # noqa: BLE001
        return out, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--tmp", default="/tmp/pocketdetect_foldseek")
    ap.add_argument("--reuse-search", action="store_true", help="reuse an existing test_vs_train.m8")
    args = ap.parse_args()
    trd, ted = FS / "train", FS / "test"
    for d in (trd, ted):
        d.mkdir(parents=True, exist_ok=True)
    jobs = []
    entries = []
    for split in ("train", "dev"):
        m = pd.read_csv(PROC / split / "manifest.csv", dtype={"chain": str, "pdb": str}, keep_default_na=False)
        m = m[m.status == "ok"]
        for r in m.itertuples():
            out = trd / f"{r.entry_id}.pdb"
            entries.append((r.entry_id, split))
            if not out.exists():
                jobs.append((r.structure_path, r.chain, out))
    meta = pd.read_csv(MAN / "test_union_meta.tsv", sep="\t", dtype=str, keep_default_na=False)
    pw = ROOT / "data" / "raw" / "pocketwatch" / "current" / "structures"
    af_models = {p.name.split("-")[1]: p for p in (PROC / "af").rglob("AF-*-F1-model_v*.cif.gz")}
    test_ids = {}
    for r in meta.itertuples():
        safe = r.id.replace("|", "__")
        out = ted / f"{safe}.pdb"
        if r.source_set == "pocketwatch_test":
            src, ch = pw / f"{r.uniprot}.cif.gz", None
        elif r.source_set == "af_set":
            src, ch = af_models.get(r.uniprot), None
        else:
            src, ch = pdb_entry_path(r.pdb_id), r.chain
        if src is None or not Path(src).exists():
            continue
        test_ids[safe] = (r.id, r.source_set)
        if not out.exists():
            jobs.append((src, ch, out))
    with ProcessPoolExecutor(args.procs) as ex:
        errs = [(o, e) for o, e in ex.map(write_chain, jobs, chunksize=8) if e]
    tmp = Path(args.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    m8 = FS / "test_vs_train.m8"
    cols = "query,target,evalue,qtmscore,ttmscore,alntmscore,prob"
    if not (args.reuse_search and m8.exists()):
      subprocess.run([str(FOLDSEEK), "easy-search", str(ted), str(trd), str(m8), str(tmp), "-e", "1e-3", "--exhaustive-search", "1",
                    "--format-output", cols, "--threads", str(args.procs), "-v", "1"], check=True)
    h = pd.read_csv(m8, sep="\t", header=None, names=cols.split(","))
    h["query"] = h["query"].str.replace(r"\.pdb$", "", regex=True)
    h["target"] = h["target"].str.replace(r"\.pdb$", "", regex=True)
    h["tm"] = h[["qtmscore", "ttmscore"]].max(axis=1)
    h = h[h.evalue < 1e-3]
    bq = h.sort_values("tm", ascending=False).drop_duplicates("query").set_index("query")
    rows = []
    for safe, (tid, src) in sorted(test_ids.items()):
        tm = float(bq.tm.get(safe, 0.0)) if safe in bq.index else 0.0
        best = bq.target.get(safe, "") if safe in bq.index else ""
        stratum = "<0.5" if tm < 0.5 else ("0.5-0.7" if tm <= 0.7 else ">0.7")
        rows.append(dict(id=tid, source_set=src, max_tm_to_train=round(tm, 4), best_train_entry=best,
                         stratum=stratum if safe in bq.index else "<0.5 (no_hit)"))
    tn = pd.DataFrame(rows)
    tn.to_csv(MAN / "test_fold_novelty.tsv", sep="\t", index=False)
    bt = h.sort_values("tm", ascending=False).drop_duplicates("target").set_index("target")
    fr = []
    for eid, split in entries:
        tm = float(bt.tm.get(eid, 0.0)) if eid in bt.index else 0.0
        fr.append(dict(entry_id=eid, split=split, max_tm_to_test=round(tm, 4),
                       best_test_id=test_ids.get(bt["query"].get(eid, ""), ("",))[0] if eid in bt.index else "",
                       fold_disjoint=tm < 0.5))
    fr = pd.DataFrame(fr)
    fr.to_csv(PROC / "train" / "fold_to_test.tsv", sep="\t", index=False)
    c = dict(n_test_structures=len(test_ids), n_train_dev_structures=len(entries), write_errors=len(errs),
             write_error_examples=[str(e) for e in errs[:10]],
             test_strata=tn.groupby(["source_set", "stratum"]).size().rename("n").reset_index().to_dict("records"),
             fold_disjoint_train_fraction=float(fr.loc[fr.split == "train", "fold_disjoint"].mean()),
             fold_disjoint_train_entries=int(fr.loc[fr.split == "train", "fold_disjoint"].sum()),
             train_tm_ge_0p6_fraction=float((fr.loc[fr.split == "train", "max_tm_to_test"] >= 0.6).mean()))
    json.dump(c, open(PROC / "train" / "foldseek_counts.json", "w"), indent=1)
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
