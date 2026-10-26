#!/usr/bin/env python
"""Descriptive summary of the BURROW human-proteome run -> results/proteome/HUMAN.md (+ JSON).

Inputs (data/processed/proteome_human/): {naive,afmode}/{pockets.tsv,structures.tsv,timing.csv},
run.log, mem.log, inputs.list, known_pockets_uniprot.tsv (scripts/proteome_known_pockets.py) and the
optional P2Rank subset comparison (p2rank_subset/).

Protein-level view: AFDB splits proteins > 2,700 aa into overlapping 1,400-residue fragments (F1..Fn,
numbered from 1 within each fragment; DBREF gives the UniProt offset).  Fragment pockets are mapped
to UniProt numbering, pooled, and de-duplicated greedily by score (residue Jaccard >= 0.5).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/processed/proteome_human"
OUT = ROOT / "results/proteome"
STEM = re.compile(r"AF-(?P<u>[^-]+)-F(?P<f>\d+)-model_v\d+")
TOPK = (1, 3, 5, 10, 20)


def frag_offsets(paths):
    """UniProt offset (DBREF seqBegin - 1) of every fragment of fragmented proteins."""
    stems = {Path(p).name.split(".")[0]: p for p in paths}
    frag_prot = {STEM.match(s)["u"] for s in stems if STEM.match(s)["f"] != "1"}
    off = {}
    for s, p in stems.items():
        if STEM.match(s)["u"] not in frag_prot:
            continue
        with gzip.open(p, "rt") as fh:
            for l in fh:
                if l.startswith("DBREF"):
                    off[s] = int(l.split()[-2]) - 1      # DBREF ... UNP <acc> <id> <seqBegin> <seqEnd>
                    break
                if l.startswith("ATOM"):
                    break
    return off, frag_prot


def protein_pockets(pk: pd.DataFrame, offsets: dict, k: int = 20):
    """{uniprot: list of (score, mean_plddt, min_plddt, residue-number set)} top-k after fragment merge."""
    out = {}
    pk = pk.assign(u=pk.stem.str.extract(STEM)["u"])
    for u, g in pk.groupby("u", sort=False):
        items = []
        for r in g.itertuples(index=False):
            o = offsets.get(r.stem, 0)
            res = {int(t.split(":")[1].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) + o for t in str(r.residues).split()} \
                if isinstance(r.residues, str) else set()
            items.append((r.score, r.mean_plddt, r.min_plddt, res))
        if g.stem.nunique() > 1:          # fragmented: pool, sort, de-duplicate
            items.sort(key=lambda t: -t[0])
            kept = []
            for it in items:
                if all(len(it[3] & q[3]) / max(1, len(it[3] | q[3])) < 0.5 for q in kept):
                    kept.append(it)
            items = kept
        out[u] = items[:k]
    return out


def proteins_table(st: pd.DataFrame):
    st = st.assign(u=st.stem.str.extract(STEM)["u"])
    agg = st.groupby("u").apply(lambda g: pd.Series(dict(
        n_frag=len(g), n_res=g.n_res.sum(), frac_lt50=(g.frac_plddt_lt50 * g.n_res).sum() / max(1, g.n_res.sum()),
        frac_lt70=(g.frac_plddt_lt70 * g.n_res).sum() / max(1, g.n_res.sum()),
        mean_plddt=(g.mean_plddt * g.n_res).sum() / max(1, g.n_res.sum()), n_pockets=g.n_pockets.sum())),
        include_groups=False)
    return agg


def low_plddt_fracs(prot_pk, sel=None, top=3):
    v50 = v70 = n = 0
    for u, items in prot_pk.items():
        if sel is not None and u not in sel:
            continue
        for s, mp, _, _ in items[:top]:
            if np.isnan(mp):
                continue
            n += 1
            v50 += mp < 50
            v70 += mp < 70
    return dict(n_pockets=n, frac_mean_plddt_lt50=round(v50 / max(n, 1), 4), frac_mean_plddt_lt70=round(v70 / max(n, 1), 4))


def known_recall(prot_pk, known: pd.DataFrame, cov=0.5):
    out = {}
    have = set(prot_pk)
    kn = known[known.uniprot.isin(have)]
    for cat in ["all"] + sorted(kn.category.unique()):
        k = kn if cat == "all" else kn[kn.category == cat]
        hits = {t: 0 for t in TOPK}
        for r in k.itertuples(index=False):
            site = {int(x.split(":")[1]) for x in r.residues.split()}
            pk = prot_pk.get(r.uniprot, [])
            first = next((i for i, it in enumerate(pk) if len(site & it[3]) / len(site) >= cov), None)
            for t in TOPK:
                hits[t] += first is not None and first < t
        out[cat] = dict(n_sites=len(k), n_proteins=int(k.uniprot.nunique()),
                        **{f"recall_top{t}": round(hits[t] / max(len(k), 1), 4) for t in TOPK})
    return out


def parse_runlog():
    w = {}
    f = D / "run.log"
    if f.exists():
        for l in f.read_text().splitlines():
            m = re.match(r"(\w+) wall_s=([\d.]+) load=(.*)", l)
            if m:
                w[m[1]] = dict(wall_s=float(m[2]), load=m[3])
    return w


def main():
    paths = [l.strip() for l in (D / "inputs.list").read_text().splitlines() if l.strip()]
    offsets, frag_prot = frag_offsets(paths)
    known = pd.read_csv(D / "known_pockets_uniprot.tsv", sep="\t")
    walls = parse_runlog()
    R = dict(n_structures=len(paths), n_proteins=len({STEM.match(Path(p).name)["u"] for p in paths}),
             n_fragmented_proteins=len(frag_prot), n_fragments_of_fragmented=sum(1 for s in offsets),
             extraction_s=49.5, modes={})
    tabs = {}
    for mode in ("naive", "afmode"):
        md = D / mode
        if not (md / "pockets.tsv").exists():
            continue
        pk = pd.read_csv(md / "pockets.tsv", sep="\t")
        st = pd.read_csv(md / "structures.tsv", sep="\t")
        tm = pd.read_csv(md / "timing.csv")
        prot_pk = protein_pockets(pk, offsets)
        pt = proteins_table(st)
        tabs[mode] = (prot_pk, pt)
        wall = walls.get(mode, {}).get("wall_s", float("nan"))
        n_ok = int((tm.status == "ok").sum())
        bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
        by_dis = {}
        for lo, hi in bins:
            sel = set(pt.index[(pt.frac_lt50 >= lo) & (pt.frac_lt50 < hi)])
            by_dis[f"[{lo},{min(hi, 1.0)}]"] = dict(n_proteins=len(sel),
                                                     median_pockets_after_nms=float(pt.loc[list(sel), "n_pockets"].median()) if sel else 0,
                                                     **low_plddt_fracs(prot_pk, sel))
        R["modes"][mode] = dict(
            structures_ok=n_ok, structures_error=int((tm.status != "ok").sum()),
            errors=tm[tm.status != "ok"].note.astype(str).str[:80].value_counts().head(5).to_dict(),
            proteins_with_output=len(prot_pk),
            wall_s=wall, load=walls.get(mode, {}).get("load", ""),
            structures_per_min_run=round(60 * n_ok / wall, 1) if wall == wall else None,
            structures_per_min_incl_tar=round(60 * n_ok / (wall + R["extraction_s"]), 1) if wall == wall else None,
            pockets_per_protein_after_nms=dict(median=float(pt.n_pockets.median()), mean=round(float(pt.n_pockets.mean()), 2)),
            proteins_without_pockets=int(sum(1 for u in pt.index if not prot_pk.get(u))),
            top1_low_plddt=low_plddt_fracs(prot_pk, top=1), top3_low_plddt=low_plddt_fracs(prot_pk, top=3),
            by_disorder=by_dis, known_pocket_recall=known_recall(prot_pk, known))
    p2 = D / "p2rank_subset/summary.json"
    if p2.exists():
        R["p2rank_subset"] = json.loads(p2.read_text())
    mem = D / "mem.log"
    if mem.exists():
        pk = [float(l.split("peak=")[1]) for l in mem.read_text().splitlines() if "peak=" in l]
        R["peak_rss_GB"] = max(pk) if pk else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "HUMAN.json").write_text(json.dumps(R, indent=1))
    write_md(R)


def write_md(R):
    L = ["# BURROW on the human proteome (AFDB v6, UP000005640) — descriptive\n",
         f"{R['n_structures']} AFDB models = {R['n_proteins']} UniProt entries; {R['n_fragmented_proteins']} proteins "
         f"> 2,700 aa are split by AFDB into overlapping fragments ({R['n_fragments_of_fragmented']} fragment models), "
         "whose pockets are mapped to UniProt numbering and merged (greedy by score, residue Jaccard >= 0.5). "
         "Model `burrow_all_default_s0` (frozen v1, default preset), fast backend, 3 GPUs (0,1,3) x 8 CPU workers; "
         "outputs `data/processed/proteome_human/` (top-20 pockets per structure; git-ignored). "
         "**naive** = standard mode on the full model; **afmode** = `--af-mode` (pLDDT<50 spaghetti pruning + soft "
         "occlusion). Shared node (see load) — timings are pessimistic. Code: frozen v1 reference (git worktree at "
         "b6a896b) + the fast-module patch of 2026-10-26 22:20 (bounded GPU driver, `--compact`, fp64 thresholds in "
         "GPU AF mode; `src/pocketdetect/burrow/fast/`, file-set md5 4be956ffc31b); run script "
         "`data/processed/proteome_human/run.sh`, summary `scripts/proteome_summary.py`.\n",
         "## Throughput and output\n",
         "| | naive | afmode |", "|---|---|---|"]
    m = R["modes"]
    g = lambda k, f=lambda x: x: " | ".join(str(f(m[x][k])) if x in m else "–" for x in ("naive", "afmode"))
    okerr = " | ".join(f"{m[x]['structures_ok']} / {m[x]['structures_error']}" if x in m else "–"
                       for x in ("naive", "afmode"))
    pp = lambda d: f"{d['median']:.0f} / {d['mean']}"
    L += [f"| structures ok / error | {okerr} |",
          f"| wall-clock run (s) | {g('wall_s')} |",
          f"| structures/min (run only) | {g('structures_per_min_run')} |",
          f"| structures/min incl. tar extraction ({R['extraction_s']} s, once) | {g('structures_per_min_incl_tar')} |",
          f"| node load (1/5/15 min, at end) | {g('load')} |",
          f"| pockets per protein after NMS (median / mean; NMS keeps <= 50 per model, summed over fragments) | {g('pockets_per_protein_after_nms', pp)} |",
          f"| proteins without any pocket | {g('proteins_without_pockets')} |"]
    if R.get("peak_rss_GB"):
        L.append(f"\nPeak RSS of the whole job (driver + 3 GPU processes + 24 workers): {R['peak_rss_GB']:.1f} GB.\n")
    L.append("\n## Hallucination proxy: pockets in low-confidence regions\n")
    L.append("Fraction of top-ranked pockets whose mean residue pLDDT is < 50 / < 70 (pocket residues = residues with a "
             "heavy atom <= 4 Å of a pocket point; pLDDT from the full model).\n")
    L += ["| pockets | naive <50 | naive <70 | afmode <50 | afmode <70 |", "|---|---|---|---|---|"]
    for key, lab in (("top1_low_plddt", "top-1"), ("top3_low_plddt", "top-3")):
        a, b = m.get("naive", {}).get(key, {}), m.get("afmode", {}).get(key, {})
        L.append(f"| {lab} (n = {a.get('n_pockets', '–')} / {b.get('n_pockets', '–')}) | {a.get('frac_mean_plddt_lt50', '–')} | "
                 f"{a.get('frac_mean_plddt_lt70', '–')} | {b.get('frac_mean_plddt_lt50', '–')} | {b.get('frac_mean_plddt_lt70', '–')} |")
    L.append("\n### By protein disorder (fraction of residues with pLDDT < 50)\n")
    L += ["| disorder | proteins | median pockets (naive / af) | top-3 mean pLDDT<50: naive / af | top-3 <70: naive / af |",
          "|---|---|---|---|---|"]
    for b in m.get("naive", {}).get("by_disorder", {}):
        a, c = m["naive"]["by_disorder"][b], m.get("afmode", {}).get("by_disorder", {}).get(b, {})
        L.append(f"| {b} | {a['n_proteins']} | {a['median_pockets_after_nms']:.0f} / {c.get('median_pockets_after_nms', float('nan')):.0f} | "
                 f"{a['frac_mean_plddt_lt50']} / {c.get('frac_mean_plddt_lt50', '–')} | {a['frac_mean_plddt_lt70']} / {c.get('frac_mean_plddt_lt70', '–')} |")
    L.append("\n## Overlap with known (BioLiP2-derived) pockets from HOTPocket\n")
    L.append("Descriptive only: many of these proteins / sites are in our training and test sets. Known sites = HOTPocket "
             "`known_pockets.csv` (BioLiP2 binding-site residues) mapped to UniProt numbering by local alignment "
             "(`scripts/proteome_known_pockets.py`) and de-duplicated per protein (Jaccard >= 0.5). A site counts as "
             "recovered within the top-k if one of the protein's top-k pockets contains >= 50 % of its residues.\n")
    L += ["| site category | sites | proteins | mode | top-1 | top-3 | top-5 | top-10 | top-20 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cat in m.get("naive", {}).get("known_pocket_recall", {}):
        for mode in ("naive", "afmode"):
            k = m.get(mode, {}).get("known_pocket_recall", {}).get(cat)
            if k:
                L.append(f"| {cat} | {k['n_sites']} | {k['n_proteins']} | {mode} | " +
                         " | ".join(str(k[f'recall_top{t}']) for t in TOPK) + " |")
    if "p2rank_subset" in R:
        p = R["p2rank_subset"]
        L.append(f"\n## P2Rank `-c alphafold` vs BURROW on {p['n']} random proteins\n")
        L.append(p.get("note", ""))
        L += ["", "| method | wall s | structures/min | top-3 mean pLDDT<50 | top-3 <70 | pockets/protein (median) |",
              "|---|---|---|---|---|---|"]
        for r in p["rows"]:
            L.append(f"| {r['method']} | {r['wall_s']} | {r['per_min']} | {r['top3_lt50']} | {r['top3_lt70']} | {r['median_pockets']} |")
    (OUT / "HUMAN.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    print("written", OUT / "HUMAN.md", file=sys.stderr)
