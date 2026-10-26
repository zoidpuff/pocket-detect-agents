#!/usr/bin/env python
"""Stage 2 of the dimer benchmark: seeded complex selection and file generation.

Selection (seed ``pocketdetect.seeds.SEEDS[0]``):
  * pass 1: LIGYSIS UniProts that have >= 1 qualifying entry with an interface site (homo or
    hetero, strict unique sites), in seeded random order; for each UniProt not yet covered pick
    one of its interface entries uniformly at random among those not containing an already
    covered LIGYSIS UniProt; all LIGYSIS UniProts of the chosen complex become covered
    (one entry per UniProt / per hetero complex);
  * pass 2: remaining UniProts (entries without interface sites), same procedure, until
    ``--n-total`` complexes are selected (fills with single-chain-only complexes).

Outputs (data/processed/dimer/):
  holo/<pdb>.pdb                assembly 1 with polymer + all non-water hetero groups (GT source)
  inputs/assembly/<pdb>.pdb     assembly 1, polymer only (method input, full assembly)
  inputs/chains/<pdb>_<C>.pdb   each protein chain alone, polymer only (same frame)
  gt/<pdb>.json                 ground truth incl. all ligand copies (atoms, centroid, residues),
                                classes and symmetric-copy grouping (read by scripts/eval_dimer.py)
  complexes.tsv, chain_map.tsv, site_copies.tsv, unique_sites.tsv, manifest_chains.tsv
  gt_manifest_assembly.csv      scripts/evaluate.py manifest (ligand-centric, no collapse)
  gt_manifest_primary_chain.csv scripts/evaluate.py manifest for the isolated primary chains
  inputs_assembly.csv, inputs_chains.csv  method-input lists
  selection.json                selection parameters and counts

Usage: python scripts/data/build_dimer_select.py [--n-total 400] [--jobs 32] [--stage manifest|all]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gemmi  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_dimer_common import json_dump, load_biolip, process_entry, unique_sites, write_pdb  # noqa: E402
from build_dimer_scan import BIOLIP, ligysis_accessions  # noqa: E402
from pocketdetect.seeds import SEEDS  # noqa: E402
from pocketdetect.structure import PROTEIN, load_structure  # noqa: E402

OUT = ROOT / "data/processed/dimer"
INTERFACE = ("homo_interface", "hetero_interface")


def chain_relevant(chain_counts: dict, chain: str) -> bool:
    """Copy counts for an isolated chain if it has >= min(3, max residues on any chain) residues on it."""
    if not chain_counts:
        return False
    return chain_counts.get(chain, 0) >= min(3, max(chain_counts.values()))


def load_scan() -> list[dict]:
    with gzip.open(OUT / "scan/entries.jsonl.gz", "rt") as f:
        return [json.loads(l) for l in f]


def select(entries: list[dict], acc: set, n_total: int, seed: int) -> tuple[list[str], dict]:
    ok = {e["pdb_id"]: e for e in entries if e["status"] == "ok"}
    e_acc, e_int = {}, {}
    by_acc: dict[str, list] = defaultdict(list)
    for pid in sorted(ok):
        e = ok[pid]
        accs = sorted({a for en in e["entities"].values() for a in en["uniprot"]} & acc)
        e_acc[pid] = accs
        e_int[pid] = sum(u["site_class"] in INTERFACE for u in unique_sites(e["copies"], "site_strict"))
        for a in accs:
            by_acc[a].append(pid)
    rng = np.random.default_rng(seed)
    covered: set = set()
    chosen: list[str] = []
    log = {"pass1": 0, "pass2": 0, "skipped_no_free_entry": 0}
    for pas in (1, 2):
        accs = sorted(a for a in by_acc if (pas == 1) == any(e_int[p] > 0 for p in by_acc[a]))
        for a in [accs[i] for i in rng.permutation(len(accs))]:
            if a in covered:
                continue
            if pas == 2 and len(chosen) >= n_total:
                break
            cand = [p for p in by_acc[a] if (e_int[p] > 0 or pas == 2) and not (set(e_acc[p]) & covered)]
            if not cand:
                log["skipped_no_free_entry"] += 1
                continue
            p = cand[int(rng.integers(len(cand)))]
            chosen.append(p)
            covered |= set(e_acc[p])
            log[f"pass{pas}"] += 1
    log["n_acc_with_entry"] = len(by_acc)
    log["n_acc_with_interface_entry"] = sum(any(e_int[p] > 0 for p in v) for v in by_acc.values())
    log["n_acc_covered"] = len(covered)
    return chosen, log


def manifest_chains(entries: dict, chosen: list[str]) -> pd.DataFrame:
    rows = []
    for pid in chosen:
        e = entries[pid]
        for c in e["chains"]:
            ent = e["entities"][c["entity_id"]]
            rows.append(
                {
                    "pdb_id": pid,
                    "chain": c["chain"],
                    "orig_auth_chain": c["orig_auth_chain"],
                    "operator": c["operator"],
                    "entity_id": c["entity_id"],
                    "uniprot": ";".join(ent["uniprot"]),
                    "sequence": ent["sequence"],
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- file generation
def _write_polymer_only(holo: Path, out: Path, chains: list[str] | None) -> int:
    """Write protein residues (structure.py PROTEIN class, heavy atoms) of ``holo``; returns #atoms."""
    s = load_structure(str(holo), protein_only=True)
    keep = {(str(c), int(n), str(i)) for c, n, i in zip(s.r_chain, s.r_num, s.r_icode)}
    st = gemmi.read_structure(str(holo))
    st.remove_hydrogens()
    for ch in st[0]:
        for i in range(len(ch) - 1, -1, -1):
            r = ch[i]
            ic = "" if r.seqid.icode == " " else r.seqid.icode
            if (ch.name, r.seqid.num, ic) not in keep or (chains is not None and ch.name not in chains):
                del ch[i]
    st.remove_empty_chains()
    write_pdb(st, out)
    s2 = load_structure(str(out))
    ref = s.select(np.isin(s.chain, chains)) if chains is not None else s
    if s2.n_atoms != ref.n_atoms or not np.array_equal(s2.coords, ref.coords) or not (s2.kind == PROTEIN).all():
        raise RuntimeError(f"polymer-only file {out} differs from holo protein atoms")
    return int(s2.n_atoms)


_G: dict = {}


def _init(biolip, acc):
    _G.update(biolip=biolip, acc=acc)


def _build(args) -> dict:
    pid, scan_rec = args
    holo = OUT / "holo" / f"{pid}.pdb"
    r = process_entry(pid, _G["biolip"].get(pid, set()), _G["acc"], holo)
    if r["status"] != "ok":
        raise RuntimeError(f"{pid}: rebuild status {r['status']}")
    # determinism check against the scan
    a = [(c["ligand_id"], c["site_class"], c["site_strict"], c["site_j50"]) for c in r["copies"]]
    b = [(c["ligand_id"], c["site_class"], c["site_strict"], c["site_j50"]) for c in scan_rec["copies"]]
    if a != b:
        raise RuntimeError(f"{pid}: rebuild differs from scan")
    chains = [c["chain"] for c in r["chains"]]
    r["n_atoms_assembly_input"] = _write_polymer_only(holo, OUT / "inputs/assembly" / f"{pid}.pdb", None)
    r["n_atoms_chain_input"] = {
        c: _write_polymer_only(holo, OUT / "inputs/chains" / f"{pid}_{c}.pdb", [c]) for c in chains
    }
    # spec token per ligand group, for per-copy specs
    tok = {}
    for t in r["ligand_specs"]:
        name, rest = t.split("[")
        ch, num = rest.rstrip("]").rsplit("_", 1)
        tok[f"{ch}:{num}"] = t
    for c in r["copies"]:
        groups = c["ligand_id"].split("|", 1)[1].split("&")
        c["spec"] = ",".join(tok[g] for g in groups)
        c["primary_chain_relevant"] = chain_relevant(c["chain_counts"], r["primary_chain"])
    r["unique_sites_strict"] = unique_sites(r["copies"], "site_strict")
    r["unique_sites_j50"] = unique_sites(r["copies"], "site_j50")
    r["holo_sha256"] = hashlib.sha256(holo.read_bytes()).hexdigest()
    json_dump(r, OUT / "gt" / f"{pid}.json")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-total", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--stage", choices=["manifest", "all"], default="all")
    a = ap.parse_args(argv)
    seed = SEEDS[0]
    acc = ligysis_accessions()
    entries = load_scan()
    chosen, log = select(entries, acc, a.n_total, seed)
    ok = {e["pdb_id"]: e for e in entries if e["status"] == "ok"}
    mc = manifest_chains(ok, chosen)
    mc.to_csv(OUT / "manifest_chains.tsv", sep="\t", index=False)
    sel = {"seed": seed, "n_total_target": a.n_total, "n_selected": len(chosen), "selected": chosen, **log}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps({k: v for k, v in sel.items() if k != "selected"}, indent=2), flush=True)
    if a.stage == "manifest":
        return
    biolip = load_biolip(BIOLIP, set(chosen))
    for d in ("holo", "inputs/assembly", "inputs/chains", "gt"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(biolip, acc)) as ex:
        recs = list(ex.map(_build, [(p, ok[p]) for p in chosen], chunksize=2))
    write_tables(recs)


def write_tables(recs: list[dict]) -> None:
    cx, cm, sc, us, gma, gmc, ina, inc = [], [], [], [], [], [], [], []
    for r in recs:
        pid = r["pdb_id"]
        ents = r["entities"]
        u = r["unique_sites_strict"]
        uj = r["unique_sites_j50"]
        cnt = lambda lst, k: sum(x["site_class"] == k for x in lst)  # noqa: E731
        cx.append(
            {
                "complex_id": pid,
                "pdb_id": pid,
                "method": r["method"],
                "resolution": r["resolution"],
                "assembly_details": r["assembly_details"],
                "oligomeric_details": r["oligomeric_details"],
                "n_chains": len(r["chains"]),
                "n_entities": len(ents),
                "complex_type": "homo" if len(ents) == 1 else "hetero",
                "uniprots": ";".join(sorted({x for e in ents.values() for x in e["uniprot"]})),
                "primary_chain": r["primary_chain"],
                "n_copies": len(r["copies"]),
                "n_unique_sites": len(u),
                "n_unique_sites_j50": len(uj),
                **{f"n_{k}": cnt(u, k) for k in ("single_chain", "homo_interface", "hetero_interface", "minor_interface")},
                **{f"n_{k}_j50": cnt(uj, k) for k in ("single_chain", "homo_interface", "hetero_interface", "minor_interface")},
                "n_ion_unique_sites": sum(x["is_ion"] for x in u),
                "n_pp_clash_pairs": r["n_pp_clash_pairs"],
                "n_protein_atoms": r["n_protein_atoms"],
                "n_biolip_ligands_entry": r["n_biolip_ligands_entry"],
                "n_biolip_ligands_in_assembly": r["n_biolip_ligands_in_assembly"],
                "n_copies_no_contact_dropped": r["n_copies_no_contact"],
            }
        )
        for c in r["chains"]:
            e = ents[c["entity_id"]]
            cm.append(
                {
                    "complex_id": pid,
                    "chain": c["chain"],
                    "orig_auth_chain": c["orig_auth_chain"],
                    "operator": c["operator"],
                    "label_asym_ids": " ".join(c["label_asym_ids"]),
                    "entity_id": c["entity_id"],
                    "uniprot": ";".join(e["uniprot"]),
                    "n_observed_residues": c["n_observed_residues"],
                    "site_contacts": r["chain_contacts"][c["chain"]],
                    "is_primary": c["chain"] == r["primary_chain"],
                }
            )
            inc.append(
                {
                    "structure_id": f"{pid}_{c['chain']}",
                    "complex_id": pid,
                    "chain": c["chain"],
                    "is_primary": c["chain"] == r["primary_chain"],
                    "input_path": f"inputs/chains/{pid}_{c['chain']}.pdb",
                    "n_atoms": r["n_atoms_chain_input"][c["chain"]],
                }
            )
        for c in r["copies"]:
            sc.append(
                {
                    "complex_id": pid,
                    "ligand_id": c["ligand_id"],
                    "ccd": c["ccd"],
                    "n_atoms": c["n_atoms"],
                    "is_ion": c["is_ion"],
                    "site_class": c["site_class"],
                    "site_strict": c["site_strict"],
                    "site_j50": c["site_j50"],
                    "center_x": c["center"][0],
                    "center_y": c["center"][1],
                    "center_z": c["center"][2],
                    "chain_counts": " ".join(f"{k}:{v}" for k, v in c["chain_counts"].items()),
                    "primary_chain_relevant": c["primary_chain_relevant"],
                    "min_dist_protein": c["min_dist_protein"],
                    "n_atoms_clash": c["n_atoms_clash"],
                    "binding_residues": " ".join(c["binding_residues"]),
                    "entity_residues": " ".join(c["entity_residues"]),
                    "spec": c["spec"],
                }
            )
        for grouping, lst in (("strict", u), ("j50", uj)):
            for x in lst:
                us.append({"complex_id": pid, "grouping": grouping, **x})
        gma.append({"structure_id": pid, "structure_path": f"holo/{pid}.pdb", "ligand_spec": ",".join(c["spec"] for c in r["copies"]), "chains": ""})
        pc = r["primary_chain"]
        spec_pc = ",".join(c["spec"] for c in r["copies"] if c["primary_chain_relevant"])
        gmc.append({"structure_id": f"{pid}_{pc}", "structure_path": f"holo/{pid}.pdb", "ligand_spec": spec_pc, "chains": ""})
        ina.append({"structure_id": pid, "complex_id": pid, "input_path": f"inputs/assembly/{pid}.pdb", "n_atoms": r["n_atoms_assembly_input"]})
    pd.DataFrame(cx).to_csv(OUT / "complexes.tsv", sep="\t", index=False)
    pd.DataFrame(cm).to_csv(OUT / "chain_map.tsv", sep="\t", index=False)
    pd.DataFrame(sc).to_csv(OUT / "site_copies.tsv", sep="\t", index=False)
    pd.DataFrame(us).to_csv(OUT / "unique_sites.tsv", sep="\t", index=False)
    pd.DataFrame(gma).to_csv(OUT / "gt_manifest_assembly.csv", index=False)
    pd.DataFrame(gmc).to_csv(OUT / "gt_manifest_primary_chain.csv", index=False)
    pd.DataFrame(ina).to_csv(OUT / "inputs_assembly.csv", index=False)
    pd.DataFrame(inc).to_csv(OUT / "inputs_chains.csv", index=False)
    print(f"[select] wrote {len(recs)} complexes to {OUT}")


if __name__ == "__main__":
    main()
