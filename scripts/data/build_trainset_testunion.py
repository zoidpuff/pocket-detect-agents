#!/usr/bin/env python
"""Step 1 of the training-set build: freeze the *test union* (every chain/sequence of every test set).

Outputs
  data/manifests/test_union.tsv           id, sequence, source_set (frozen; sha256 recorded)
  data/manifests/test_union.sha256        sha256 of test_union.tsv
  data/manifests/test_union_meta.tsv      id, source_set, pdb_id, chain, uniprot, seq_len, seq_source
  data/manifests/test_union_pdb_ids.txt   all PDB IDs touched by any test set (incl. holo partners)
  data/manifests/test_union_uniprots.txt  all UniProt accessions of any test entry
  data/processed/test_sequences.fasta     same sequences, FASTA, header ">{id} source_set=.. pdb=.. uniprot=.."
  data/manifests/test_union_counts.json   per-dataset counts

Test sets (DESIGN.md §2.5 / §9): LIGYSIS (2,775 evaluated representative chains = source 'ligysis';
the other 673 chains of the 3,448-chain LIGYSIS list = 'ligysis_extra', included conservatively),
COACH420 and HOLO4K (ALL protein entities of each PDB entry), CryptoBench (all splits; apo chain
and holo chain of every pair), PocketMiner (val + test apo chains; PDB IDs of all holo partner
files), PocketWatch test split (UniProt canonical sequences), AF set (canonical UniProt sequences
from Track C's data/processed/af/af_sequences.fasta), dimer set (Track D manifest, if it
exists — all chains). CHEN11 is NOT a test set.

Sequences: canonical deposited entity sequence (_entity_poly.pdbx_seq_one_letter_code_can) from the
local PDB mirror (covers unobserved residues => conservative); falls back to the observed sequence
of the dataset's own file when the entry is absent from the mirror (obsolete IDs).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from pocketdetect.datasets.pdbinfo import entry_info  # noqa: E402
from safe_pickle import safe_load  # noqa: E402

RAW = ROOT / "data" / "raw"
MAN = ROOT / "data" / "manifests"
PROC = ROOT / "data" / "processed"
SIFTS = PROC / "af" / "external" / "pdb_chain_uniprot.tsv.gz"
PROTEIN_TYPES = ("polypeptide(L)", "polypeptide(D)")


def _info(pid):
    try:
        return pid, entry_info(pid), None
    except Exception as e:  # noqa: BLE001
        return pid, None, f"{type(e).__name__}: {e}"


def _file_chain_seqs(path: str) -> dict:
    """Observed protein sequence per author chain of a PDB/mmCIF file (fallback only)."""
    st = gemmi.read_structure(path)
    st.setup_entities()
    out = {}
    for ch in st[0]:
        aa = [r.name for r in ch if (gemmi.find_tabulated_residue(r.name) or None) is not None
              and gemmi.find_tabulated_residue(r.name).is_amino_acid()]
        seq = gemmi.one_letter_code(aa).upper()
        if len(seq.replace("X", "")) >= 10:
            out[ch.name] = seq
    return out


def load_sifts() -> dict:
    m = defaultdict(set)
    if SIFTS.exists():
        df = pd.read_csv(SIFTS, sep="\t", comment="#", dtype=str, keep_default_na=False, usecols=["PDB", "CHAIN", "SP_PRIMARY"])
        for p, c, u in zip(df.PDB, df.CHAIN, df.SP_PRIMARY):
            m[(p.lower(), c)].add(u)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--dimer-manifest", default=None, help="Track D manifest (auto-detected if omitted)")
    args = ap.parse_args()

    # (source_set, pdb_id or None, chain or None ('*' = all protein entities), uniprot hint, seq or None, fallback file)
    reqs: list[tuple] = []
    extra_pdb: dict[str, set] = defaultdict(set)  # PDB IDs used by a test set without own chain entry

    # ---------------- LIGYSIS
    lr = RAW / "ligysis" / "lbs_comparison_results"
    chains = safe_load(lr / "LIGYSIS_3448_chains.pkl")
    labels = safe_load(lr / "LIGYSIS_ress_binary_labels.pkl")
    sites = safe_load(lr / "LIGYSIS_sites_DEF_TRANS.pkl")
    eval_chains = set(labels["rep_chain"].unique())
    acc_of = sites.groupby("rep_chain")["ACC"].first().to_dict()
    trans_dir = next((RAW / "ligysis" / "lbs_comparison_code").glob("LBS-comparison-*/*/DATA/trans_rep_chains"))
    for rc in chains:
        pid, ch = rc.split("_", 1)
        src = "ligysis" if rc in eval_chains else "ligysis_extra"
        reqs.append((src, pid.lower(), ch, acc_of.get(rc, ""), None, str(trans_dir / f"{rc}.trans.pdb")))
    assert sum(1 for r in reqs if r[0] == "ligysis") == 2775, "expected 2,775 LIGYSIS evaluation chains"

    # ---------------- COACH420 / HOLO4K: all protein entities of each entry
    p2 = RAW / "p2rank_datasets" / "current"
    for ds, name in (("coach420.ds", "coach420"), ("holo4k.ds", "holo4k")):
        for line in open(p2 / ds):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("HEADER"):
                continue
            f = line.split()[0]
            pid = Path(f).stem[:4].lower()
            reqs.append((name, pid, "*", "", None, str(p2 / f)))

    # ---------------- CryptoBench (all splits): apo chain + holo chain of every pair
    cb = RAW / "cryptobench" / "osf" / "cryptobench" / "cryptobench-dataset"
    cbd = json.load(open(cb / "dataset.json"))
    cif_dir = RAW / "cryptobench" / "cif-files" / "cif-files"
    seen = set()
    for apo, pairs in cbd.items():
        for h in pairs:
            pc = [(apo, c) for c in h["apo_chain"].split("-")] + [(h["holo_pdb_id"], c) for c in h["holo_chain"].split("-")]
            for pid, ch in pc:  # multi-chain selections are written "A-B"
                if (pid, ch) in seen:
                    continue
                seen.add((pid, ch))
                reqs.append(("cryptobench", pid.lower(), ch, h.get("uniprot_id", ""), None, str(cif_dir / f"{pid}.cif")))

    # ---------------- PocketMiner (val + test apo chains; holo partner PDB IDs)
    pm = RAW / "pocketminer" / "data" / "pm-dataset"
    for split in ("val", "test"):
        for x in np.load(pm / f"{split}_apo_ids_with_chainids.npy", allow_pickle=False):
            x = str(x)
            reqs.append(("pocketminer", x[:4].lower(), x[4:] or "*", "", None, str(pm / "apo-structures" / f"{x}.pdb")))
    for p in (pm / "all-structures").glob("*.pdb"):
        extra_pdb["pocketminer"].add(p.name[:4].lower())

    # ---------------- PocketWatch test split (canonical UniProt sequences)
    pw = RAW / "pocketwatch" / "current" / "data"
    seqs = pd.read_parquet(pw / "sequences.parquet")
    seqs = seqs[seqs.split == "test"]
    for u, s in zip(seqs.normalized_uniprot_id, seqs.sequence):
        reqs.append(("pocketwatch_test", None, None, u, s, None))
    pws = pd.read_parquet(pw / "pocket_label_sites.parquet", columns=["normalized_uniprot_id", "split", "pdb_ids"])
    for ids in pws.loc[pws.split == "test", "pdb_ids"]:
        extra_pdb["pocketwatch_test"].update(str(i).lower() for i in ids)

    # ---------------- AF set (Track C, final): data/processed/af/af_sequences.fasta (all records,
    # 'included' and 'excluded' alike = conservative); headers '>AF-<acc>-F1 acc=<acc> status=...'
    af_fa = PROC / "af" / "af_sequences.fasta"
    af_used = None
    if af_fa.exists():
        acc, n = None, 0
        buf: dict[str, list] = {}
        for line in open(af_fa):
            if line.startswith(">"):
                acc = next(t.split("=", 1)[1] for t in line.split() if t.startswith("acc="))
                buf[acc] = []
            elif acc:
                buf[acc].append(line.strip())
        for a, parts in buf.items():
            reqs.append(("af_set", None, None, a, "".join(parts), None))
        af_used = dict(fasta=str(af_fa.relative_to(ROOT)), n_seqs=len(buf), sha256=hashlib.sha256(af_fa.read_bytes()).hexdigest())

    # ---------------- Dimer set (Track D): manifest_chains.tsv lists EVERY chain (partners incl.)
    dimer_used = None
    dm = Path(args.dimer_manifest) if args.dimer_manifest else PROC / "dimer" / "manifest_chains.tsv"
    if dm.exists():
        ddf = pd.read_csv(dm, sep="\t", dtype=str, keep_default_na=False)
        ddf = ddf.drop_duplicates(["pdb_id", "orig_auth_chain", "sequence"])
        for r in ddf.itertuples():
            reqs.append(("dimer", None, None, r.uniprot, r.sequence, None, f"{r.pdb_id.lower()}_{r.orig_auth_chain}"))
        extra_pdb["dimer"].update(ddf.pdb_id.str.lower())
        dimer_used = dict(manifest=str(dm.relative_to(ROOT)), n_rows=int(len(ddf)), n_entries=int(ddf.pdb_id.nunique()),
                          sha256=hashlib.sha256(dm.read_bytes()).hexdigest())

    # ---------------- resolve sequences from mmCIF entity records
    pids = sorted({r[1] for r in reqs if r[1]} | set().union(*extra_pdb.values()))
    with ProcessPoolExecutor(args.procs) as ex:
        infos = {}
        errs = {}
        for pid, info, err in ex.map(_info, pids, chunksize=16):
            if info is None:
                errs[pid] = err
            else:
                infos[pid] = info
    print(f"entry info: {len(infos)} ok, {len(errs)} missing/failed", flush=True)
    sifts = load_sifts()

    rows, fallbacks, unresolved = [], [], []
    for req in reqs:
        src, pid, ch, uhint, seq, fpath = req[:6]
        if pid is None:
            if len(req) > 6:  # explicit sequence with a PDB chain id (dimer manifest)
                p_, c_ = req[6].rsplit("_", 1)
                rows.append(dict(id=f"{src}|{req[6]}", sequence=seq, source_set=src, pdb_id=p_, chain=c_,
                                 uniprot=",".join(sorted({uhint} | sifts.get((p_, c_), set()) - {""})), seq_source="dimer_manifest"))
            else:
                rows.append(dict(id=f"{src}|{uhint}", sequence=seq, source_set=src, pdb_id="", chain="",
                                 uniprot=uhint, seq_source="uniprot_canonical"))
            continue
        info = infos.get(pid)
        got = []
        if info is not None:
            for e in info["entities"]:
                if e["type"] not in PROTEIN_TYPES or not e["seq"]:
                    continue
                if ch == "*":
                    for c in e["chains"][:1]:  # one row per protein entity
                        got.append((c, e["seq"], e["uniprot"], "pdb_entity_can"))
                elif ch in e["chains"]:
                    got.append((ch, e["seq"], e["uniprot"], "pdb_entity_can"))

        if not got and info is not None and ch not in ("*",) and ch != ch.upper():  # PocketMiner '5x8ua'
            for e in info["entities"]:
                if e["type"] in PROTEIN_TYPES and e["seq"] and ch.upper() in e["chains"]:
                    got.append((ch.upper(), e["seq"], e["uniprot"], "pdb_entity_can"))
        if not got and fpath and Path(fpath).exists():
            try:
                fs = _file_chain_seqs(fpath)
                for c, s in fs.items():
                    if ch == "*" or c == ch:
                        got.append((c, s, [], "dataset_file_observed"))
                fallbacks.append((src, pid, ch))
            except Exception as e:  # noqa: BLE001
                unresolved.append((src, pid, ch, f"file parse: {e}"))
        if not got:
            unresolved.append((src, pid, ch, errs.get(pid, "chain not found")))
            continue
        for c, s, unps, ss in got:
            u = set(unps) | sifts.get((pid, c), set())
            if uhint:
                u.add(uhint)
            rows.append(dict(id=f"{src}|{pid}_{c}", sequence=s, source_set=src, pdb_id=pid, chain=c,
                             uniprot=",".join(sorted(u)), seq_source=ss))

    df = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)
    df["seq_len"] = df.sequence.str.len()
    MAN.mkdir(parents=True, exist_ok=True)
    tu = MAN / "test_union.tsv"
    df[["id", "sequence", "source_set"]].to_csv(tu, sep="\t", index=False)
    sha = hashlib.sha256(tu.read_bytes()).hexdigest()
    (MAN / "test_union.sha256").write_text(f"{sha}  test_union.tsv\n")
    df[["id", "source_set", "pdb_id", "chain", "uniprot", "seq_len", "seq_source"]].to_csv(
        MAN / "test_union_meta.tsv", sep="\t", index=False)
    with open(PROC / "test_sequences.fasta", "w") as fh:
        for r in df.itertuples():
            fh.write(f">{r.id} source_set={r.source_set} pdb={r.pdb_id}_{r.chain} uniprot={r.uniprot}\n{r.sequence}\n")
    test_pdb = set(df.pdb_id[df.pdb_id != ""]) | set().union(*extra_pdb.values())
    test_unp = {u for x in df.uniprot for u in x.split(",") if u}
    (MAN / "test_union_pdb_ids.txt").write_text("\n".join(sorted(test_pdb)) + "\n")
    (MAN / "test_union_uniprots.txt").write_text("\n".join(sorted(test_unp)) + "\n")

    counts = dict(
        test_union_sha256=sha,
        n_rows=len(df), n_unique_sequences=int(df.sequence.nunique()),
        per_source_rows=df.source_set.value_counts().to_dict(),
        per_source_unique_pdb=df[df.pdb_id != ""].groupby("source_set").pdb_id.nunique().to_dict(),
        per_source_seq_source=df.groupby(["source_set", "seq_source"]).size().rename("n").reset_index().to_dict("records"),
        extra_pdb_ids={k: len(v) for k, v in extra_pdb.items()},
        n_test_pdb_ids=len(test_pdb), n_test_uniprots=len(test_unp),
        n_requests=len(reqs), n_fallback_to_file=len(fallbacks), unresolved=unresolved,
        entries_missing_in_mirror=sorted(errs)[:200], n_entries_missing_in_mirror=len(errs),
        af_set=af_used, dimer_set=dimer_used,
        sifts=str(SIFTS.relative_to(ROOT)) if SIFTS.exists() else None,
    )
    json.dump(counts, open(MAN / "test_union_counts.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in counts.items() if k not in ("entries_missing_in_mirror", "per_source_seq_source")}, indent=1))


if __name__ == "__main__":
    main()
