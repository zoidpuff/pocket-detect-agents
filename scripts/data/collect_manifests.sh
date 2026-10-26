#!/usr/bin/env bash
# Copy every data/raw/**/MANIFEST.tsv (file, sha256, bytes, retrieved_utc, url) into the tracked
# directory data/manifests/ so checksums are versioned in git while the data itself is not.
source "$(dirname "$0")/_common.sh"
DEST="$ROOT/data/manifests"; mkdir -p "$DEST"
find "$RAW" -name MANIFEST.tsv | sort | while read -r m; do
  rel="${m#$RAW/}"; name="$(echo "${rel%/MANIFEST.tsv}" | tr '/' '__')"
  { printf '# source manifest: data/raw/%s\n# file\tsha256\tbytes\tretrieved_utc\turl\n' "$rel"; cat "$m"; } > "$DEST/$name.tsv"
  echo "$DEST/$name.tsv ($(wc -l < "$m") files)"
done
