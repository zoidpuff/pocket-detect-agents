# Shared helpers for dataset download scripts. Source, don't execute.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW="$ROOT/data/raw"
# fetch URL OUT : download (resumable) unless present, then record sha256 + url + date in OUTDIR/MANIFEST.tsv
fetch() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ ! -s "$out" ]]; then
    curl -fL --retry 5 --retry-delay 5 -C - -o "$out.part" "$url"
    mv "$out.part" "$out"
  fi
  local sha; sha=$(sha256sum "$out" | cut -d' ' -f1)
  local man; man="$(dirname "$out")/MANIFEST.tsv"
  grep -q -F "$(basename "$out")	" "$man" 2>/dev/null || \
    printf '%s\t%s\t%s\t%s\t%s\n' "$(basename "$out")" "$sha" "$(stat -c %s "$out")" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$url" >> "$man"
  echo "$sha  $out"
}
