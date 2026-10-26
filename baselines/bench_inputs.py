#!/usr/bin/env python
"""Build per-benchmark input directories for the baseline runs: baselines/outputs/inputs/<bench>/<structure_id>.<ext>
(symlinks to the benchmark's own structure files, or chain-restricted copies), so that every wrapper writes
<structure_id>.pockets.csv, matching the evaluation harness' prediction naming.

  .venv/bin/python baselines/bench_inputs.py [bench ...]     (default: all)
Benchmarks: ligysis (authors' ligand-free TRANS-frame chains), cryptobench (input_chains only), pocketwatch (ESMFold models),
af_set (AFDB v6 models), coach420_mlig, holo4k_mlig (P2Rank .ds proteins; ligands are stripped by the wrapper cleaning).
Dimer inputs are used in place (data/processed/dimer/inputs/{assembly,chains}).
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import gemmi

ROOT = C.PROJECT
OUT = C.BASELINES / "outputs" / "inputs"
csv.field_size_limit(1 << 30)
LIGYSIS_TRANS = ROOT / "data/raw/ligysis/lbs_comparison_code/LBS-comparison-v2.0.0/bartongroup-LBS-comparison-820714c/DATA/trans_rep_chains"


def sites_rows(name):
    return list(csv.DictReader(open(ROOT / f"data/processed/gt/{name}/sites.tsv"), delimiter="\t"))


def link(src, dst):
    dst = Path(dst)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(os.path.abspath(src), dst)


def uniq(rows):
    seen = {}
    for r in rows:
        seen.setdefault(r["structure_id"], r)
    return list(seen.values())


def ligysis():
    d = OUT / "ligysis"
    d.mkdir(parents=True, exist_ok=True)
    for r in uniq(sites_rows("ligysis_lbs")):
        link(LIGYSIS_TRANS / f"{r['structure_id']}.trans.pdb", d / f"{r['structure_id']}.pdb")
    return d


def cryptobench():
    d = OUT / "cryptobench"
    d.mkdir(parents=True, exist_ok=True)
    for r in uniq(sites_rows("cryptobench_test")):
        st = gemmi.read_structure(r["structure_path"])
        keep = set(r["input_chains"].split("-"))
        m = st[0]
        for i in range(len(m) - 1, -1, -1):
            if m[i].name not in keep:
                del m[i]
        assert len(m) == len(keep), (r["structure_id"], keep)
        fmt_pdb = all(len(c.name) == 1 for c in m)
        if fmt_pdb:
            st.write_pdb(str(d / f"{r['structure_id']}.pdb"))
        else:
            st.make_mmcif_document().write_file(str(d / f"{r['structure_id']}.cif"))
    return d


def link_sites(name, outname, chains_col=None):
    d = OUT / outname
    d.mkdir(parents=True, exist_ok=True)
    for r in uniq(sites_rows(name)):
        p = Path(r["structure_path"])
        ext = ".cif.gz" if p.name.endswith(".cif.gz") else p.suffix
        link(p, d / f"{r['structure_id']}{ext}")
    return d


def ds(name, outname):
    sys.path.insert(0, str(ROOT / "src"))
    from pocketdetect.benchmarks import REGISTRY, read_p2rank_ds
    man = read_p2rank_ds(REGISTRY[name].abspath)
    d = OUT / outname
    d.mkdir(parents=True, exist_ok=True)
    for r in man.itertuples():
        link(r.structure_path, d / r.structure_id)  # structure_id already carries the .pdb extension
    return d


def main(which):
    todo = {"ligysis": ligysis, "cryptobench": cryptobench,
            "pocketwatch": lambda: link_sites("pocketwatch_test", "pocketwatch"),
            "af_set": lambda: link_sites("af_set", "af_set"),
            "coach420_mlig": lambda: ds("coach420_mlig", "coach420_mlig"),
            "holo4k_mlig": lambda: ds("holo4k_mlig", "holo4k_mlig")}
    for k in (which or todo):
        d = todo[k]()
        n = len(list(d.iterdir()))
        bad = [p.name for p in d.iterdir() if p.is_symlink() and not p.exists()]
        print(f"{k}: {n} inputs in {d}; broken links: {len(bad)} {bad[:3]}")


if __name__ == "__main__":
    main(sys.argv[1:])
