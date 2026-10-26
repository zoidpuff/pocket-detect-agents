"""Apply the independent leakage audit (results/audit/) to the train/dev manifests.

PI decision D6 (conservative): drop every train/dev ENTRY with any input (representative chain or
assembly-1 partner, primary or broad tier) flagged in results/audit/violations_vs_test.tsv, any
train-vs-dev violation, and any entry sharing a UniProt with a test set
(results/audit/uniprot_overlap_primary.tsv). The pre-audit manifests are kept as manifest_preaudit.csv.
Idempotent: always filters from manifest_preaudit.csv.
"""
import csv, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUD = ROOT / "results/audit"

def read_tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))

excl = {}
for r in read_tsv(AUD / "violations_vs_test.tsv"):
    split, entry, kind, _ = r["input_id"].split("|", 3)
    excl.setdefault(entry, f"seq_vs_test:{r['test_sets']}:{kind}:id={r['fident']}")
for r in read_tsv(AUD / "violations_dev_vs_train.tsv"):
    entry = r.get("input_id", "").split("|")[1] if r.get("input_id") else None
    if entry:
        excl.setdefault(entry, "dev_vs_train")
for r in read_tsv(AUD / "uniprot_overlap_primary.tsv"):
    excl.setdefault(r["entry"], f"uniprot_overlap:{r['test_set']}")

out = ROOT / "data/manifests/train_dev_audit_exclusions.tsv"
with open(out, "w") as f:
    f.write("entry_id\treason\n")
    for e, why in sorted(excl.items()):
        f.write(f"{e}\t{why}\n")

for split in ("train", "dev"):
    man = ROOT / f"data/processed/{split}/manifest.csv"
    pre = man.with_name("manifest_preaudit.csv")
    if not pre.exists():
        shutil.copy(man, pre)
    with open(pre) as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["entry_id"] not in excl]
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(kept)
    print(f"{split}: {len(rows)} -> {len(kept)} (dropped {len(rows) - len(kept)})")
print(f"exclusion list: {out} ({len(excl)} entries)")
