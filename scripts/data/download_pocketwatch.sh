#!/usr/bin/env bash
# PocketWatch (zPocket paper, arXiv 2610.17696): temporally held-out binding-site benchmark on
# ESMFold2 folds. HF dataset genesisml/PocketWatch (MIT), pinned revision.
# Fetched: labels/splits/strata/projection + ESMFold2 folds (val+test dir, train zip).
# Skipped: msas_test.tar (2.7 GB pre-merged MSAs; not needed for structure-based methods).
source "$(dirname "$0")/_common.sh"
REV="${POCKETWATCH_REV:-db7bd0611a14cb98e3b1d24b343565418b7193ef}"   # main @ 2026-10-14
OUT="$RAW/pocketwatch/$REV"
mkdir -p "$OUT"
uv run --no-project --with "huggingface_hub==0.26.2" python -I - "$REV" "$OUT" <<'PY'
import sys
from huggingface_hub import snapshot_download
rev, out = sys.argv[1], sys.argv[2]
snapshot_download("genesisml/PocketWatch", repo_type="dataset", revision=rev, local_dir=out,
                  ignore_patterns=["msas_test.tar"], max_workers=8)
PY
ln -sfn "$REV" "$RAW/pocketwatch/current"
cd "$OUT"
[[ -d structures_train ]] || unzip -q structures_train.zip
# upstream md5 list (skip entries for the MSA tarball contents we did not fetch)
grep -v -E '  (\./)?msas_test/' CHECKSUMS.txt | md5sum --quiet -c - && echo "upstream CHECKSUMS.txt: OK (excluding msas_test/)"
# our sha256 manifest of everything except per-fold cifs (covered by CHECKSUMS.txt md5s)
find . -type f ! -path './.cache/*' ! -path './structures/*.cif.gz' ! -path './structures_train/*' ! -name MANIFEST.tsv | sort | \
  while read -r f; do printf '%s\t%s\t%s\t%s\thf://datasets/genesisml/PocketWatch@%s\n' "${f#./}" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REV"; done > MANIFEST.tsv
