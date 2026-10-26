#!/usr/bin/env python
"""Verification of the AF train/dev labels (Track C2) and the AF stress set. Exit 1 on failure.

  1. split integrity: AF manifests contain exactly the entries of the current (audit-filtered)
     Track A manifests, same split; no audit-excluded entry; no stale npz; no UniProt shared
     between train and dev, nor with the AF test set
  2. files: every ok entry has its npz and AF model (sha256 == model index)
  3. thresholds: every kept site has RMSD <= 3 A, identity >= 0.75, mapped fraction >= 0.5
  4. independent recomputation on a seeded sample of entries:
     - rigid-body check: intra-ligand pairwise distances identical to Track A's source atoms
     - transferred ligands touch their AF site residues (min heavy-atom distance)
     - residue mapping agrees with SIFTS (author number -> UniProt position)
     - per-residue pLDDT == CA B-factor of the model
  5. AF stress set: 300 LBS-subset included proteins, ranked by frac pLDDT < 50
Writes data/processed/af_train/verify_report.json.
Usage: .venv/bin/python scripts/data/build_af_train_verify.py [--n-sample 150]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from build_af_train_io import load_af_entry, read_af_manifest  # noqa: E402

from pocketdetect.datasets.trainset import load_entry, read_manifest  # noqa: E402

AFT = ROOT / "data/processed/af_train"
AF = ROOT / "data/processed/af"
FAILS, REPORT = [], {}


def check(name, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + msg if msg else ''}")
    REPORT.setdefault("checks", {})[name] = dict(ok=bool(ok), msg=msg)
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()
    info = json.loads((AFT / "build_info.json").read_text())
    mi = pd.read_csv(AFT / "model_index.csv", dtype=str, keep_default_na=False).set_index("acc")

    # ---------------------------------------------------------------- 1. split integrity
    print("== 1. split integrity")
    excl_p = ROOT / "data/manifests/train_dev_audit_exclusions.tsv"
    excl = set(pd.read_csv(excl_p, sep="\t").entry_id) if excl_p.exists() else set()
    accs = {}
    for split in ("train", "dev"):
        ta = read_manifest(split)
        af_all = pd.read_csv(AFT / split / "manifest.csv", dtype=str, keep_default_na=False)
        check(f"{split}: AF manifest entries == Track A manifest", set(af_all.entry_id) == set(ta.entry_id) and len(af_all) == len(ta),
              f"{len(af_all)} vs {len(ta)}")
        check(f"{split}: split column", bool((af_all.split == split).all()))
        check(f"{split}: no audit-excluded entries", not (set(af_all.entry_id) & excl), f"{len(excl)} exclusions")
        sha = hashlib.sha256((ROOT / f"data/processed/{split}/manifest.csv").read_bytes()).hexdigest()
        check(f"{split}: built from the current Track A manifest (sha256)", info["trainset_manifest_sha256"][split] == sha)
        ok = read_af_manifest(split)
        files = {p.stem for p in (AFT / split / "entries").glob("*.npz")}
        check(f"{split}: npz files == ok entries (no stale files)", files == set(ok.entry_id), f"{len(files)} files, {len(ok)} ok")
        accs[split] = set(ok.acc)
        REPORT[split] = dict(info["splits"][split])
    check("no AF accession shared by train and dev", not (accs["train"] & accs["dev"]))
    test_accs = set(pd.read_csv(AF / "af_manifest.csv").acc)
    check("no train/dev AF accession in the AF test set", not ((accs["train"] | accs["dev"]) & test_accs))

    # ---------------------------------------------------------------- 2./3. files and thresholds
    print("== 2./3. files and site thresholds")
    S = pd.read_csv(AFT / "sites.csv")
    K = S[S.kept]
    check("kept sites: RMSD <= 3, identity >= 0.75, mapped >= 0.5",
          bool((K.rmsd <= 3).all() and (K.identity >= 0.75).all() and (K.frac_mapped >= 0.5).all()))
    REPORT["site_rmsd_quantiles"] = {str(q): float(K.rmsd.quantile(q)) for q in (0.5, 0.9, 0.99)}
    REPORT["site_plddt_band"] = pd.cut(K.plddt_mean, [-1, 50, 70, 90, 101], right=False,
                                       labels=["<50", "50-70", "70-90", ">=90"]).value_counts().to_dict()
    print("  kept-site RMSD quantiles", REPORT["site_rmsd_quantiles"], "pLDDT bands", REPORT["site_plddt_band"])
    bad = []
    for split in ("train", "dev"):
        for r in read_af_manifest(split).itertuples():
            mp = AFT / r.model_path
            if not mp.exists() or hashlib.sha256(mp.read_bytes()).hexdigest() != mi.loc[r.acc, "sha256"]:
                bad.append(r.entry_id)
    check("AF models present, sha256 == model index", not bad, str(bad[:5]))

    # ---------------------------------------------------------------- 4. recomputation sample
    print("== 4. independent recomputation (sample)")
    sifts = pd.read_csv(AF / "external/pdb_chain_uniprot.tsv.gz", sep="\t", comment="#", dtype=str)
    rng = np.random.default_rng(args.seed)
    rows = []
    for split in ("train", "dev"):
        ok = read_af_manifest(split)
        ta = read_manifest(split).set_index("entry_id")
        n = min(len(ok), args.n_sample if split == "train" else max(30, args.n_sample // 3))
        for r in ok.iloc[rng.choice(len(ok), n, replace=False)].itertuples():
            e = load_af_entry(r.npz)
            src = load_entry(ta.loc[r.entry_id, "npz"])
            st = gemmi.read_structure(e.meta["model_path"])
            ch = st[0][0]
            bf = np.array([res.find_atom("CA", "*").b_iso for res in ch])
            ratoms = {res.seqid.num: np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res]) for res in ch}
            pl_ok = np.allclose(bf, e.plddt, atol=0.01)
            ptr, sp = e.arrays["lig_ptr"], src.arrays["lig_ptr"]
            # rigid body: ligands are copied in source order per kept site
            src_ligs = [l for s_ in e.arrays["site_src_index"] for l in np.flatnonzero(src.arrays["lig_site"] == s_)]
            dev_rigid = 0.0
            for k, l in enumerate(src_ligs):
                a_ = e.arrays["lig_coords"][ptr[k]:ptr[k + 1]].astype(float)
                b_ = src.arrays["lig_coords"][sp[l]:sp[l + 1]].astype(float)
                if len(a_) != len(b_):
                    dev_rigid = np.inf
                    break
                if len(a_) > 1:
                    dev_rigid = max(dev_rigid, float(np.abs(pdist(a_) - pdist(b_)).max()))
            xyz, site = e.ligand_atoms()
            touch = []
            for si, res in enumerate(e.site_residues()):
                sx = np.vstack([ratoms[int(u)] for u in res])
                touch.append(float(cKDTree(sx).query(xyz[site == si])[0].min()))
            # SIFTS agreement for source site residues (numeric author keys only)
            seg = sifts[(sifts.PDB == r.pdb) & (sifts.CHAIN == r.chain) & (sifts.SP_PRIMARY == r.acc)]
            n_cmp = n_agree = 0
            src_sites = src.site_residues()
            for si, s_ in enumerate(e.arrays["site_src_index"]):
                af_set = set(e.site_residues()[si])
                for key in src_sites[s_]:
                    if not key.lstrip("-").isdigit():
                        continue
                    num = int(key)
                    for g in seg.itertuples():
                        try:
                            pb, pe = int(g.PDB_BEG), int(g.PDB_END)
                        except ValueError:
                            continue
                        if pb <= num <= pe:
                            n_cmp += 1
                            n_agree += str(int(g.SP_BEG) + num - pb) in af_set
                            break
            rows.append(dict(split=split, entry_id=r.entry_id, n_sites=e.n_sites, plddt_ok=pl_ok, rigid_dev=dev_rigid,
                             max_touch=max(touch), sifts_cmp=n_cmp, sifts_agree=n_agree))
    T = pd.DataFrame(rows)
    REPORT["sample"] = dict(n_entries=len(T), sifts_compared=int(T.sifts_cmp.sum()), sifts_agree=int(T.sifts_agree.sum()),
                            max_rigid_dev=float(T.rigid_dev.max()), touch_quantiles={str(q): float(T.max_touch.quantile(q)) for q in (0.5, 0.9, 1.0)})
    print("  ", json.dumps(REPORT["sample"]))
    check("sample: pLDDT == CA B-factor", bool(T.plddt_ok.all()))
    check("sample: ligands moved rigidly (intra-ligand distances within 1e-3 A)", bool(T.rigid_dev.max() < 1e-3))
    check("sample: every transferred site ligand within 5 A of its AF site residues", bool(T.max_touch.max() <= 5.0),
          f"max {T.max_touch.max():.2f}")
    agree = T.sifts_agree.sum() / max(1, T.sifts_cmp.sum())
    check("sample: alignment mapping agrees with SIFTS for >= 98% of site residues", agree >= 0.98,
          f"{T.sifts_agree.sum()}/{T.sifts_cmp.sum()} = {agree:.4f}")

    # ---------------------------------------------------------------- 5. stress set
    print("== 5. AF stress set")
    st_ = pd.read_csv(AF / "af_stress_set.csv")
    man = pd.read_csv(AF / "af_manifest.csv")
    inc = man[(man.status == "included") & man.in_lbs_benchmark.astype(bool)]
    thr = st_.frac_plddt_lt50.min()
    check("stress set: 300 included LBS proteins, sorted, nothing above threshold left out",
          len(st_) == 300 and set(st_.acc) <= set(inc.acc) and st_.frac_plddt_lt50.is_monotonic_decreasing
          and int((inc.frac_plddt_lt50 > thr).sum()) <= 300, f"threshold frac<50 = {thr:.3f}")

    REPORT["n_fail"] = len(FAILS)
    (AFT / "verify_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print(f"\n{len(FAILS)} failed: {FAILS}" if FAILS else "\nALL CHECKS PASSED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
