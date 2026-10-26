#!/usr/bin/env bash
# Static Linux AVX2 binaries of MMseqs2 and Foldseek (pinned GitHub releases) -> tools/bin/
# Used for sequence/structure leakage filtering between training labels and test sets.
source "$(dirname "$0")/_common.sh"
MMSEQS_TAG="${MMSEQS_TAG:-18-8cc5c}"
MMSEQS_SHA="bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce"   # GitHub release asset digest
FOLDSEEK_TAG="${FOLDSEEK_TAG:-10-941cd33}"
DL="$ROOT/tools/downloads"; BIN="$ROOT/tools/bin"; mkdir -p "$DL" "$BIN"
fetch "https://github.com/soedinglab/MMseqs2/releases/download/$MMSEQS_TAG/mmseqs-linux-avx2.tar.gz" "$DL/mmseqs-$MMSEQS_TAG-linux-avx2.tar.gz"
fetch "https://github.com/steineggerlab/foldseek/releases/download/$FOLDSEEK_TAG/foldseek-linux-avx2.tar.gz" "$DL/foldseek-$FOLDSEEK_TAG-linux-avx2.tar.gz" "$([[ "$FOLDSEEK_TAG" == 10-941cd33 ]] && echo e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7)"
[[ "$MMSEQS_TAG" != 18-8cc5c ]] || echo "$MMSEQS_SHA  $DL/mmseqs-$MMSEQS_TAG-linux-avx2.tar.gz" | sha256sum -c -
tmp=$(mktemp -d)
tar -xzf "$DL/mmseqs-$MMSEQS_TAG-linux-avx2.tar.gz" -C "$tmp"
tar -xzf "$DL/foldseek-$FOLDSEEK_TAG-linux-avx2.tar.gz" -C "$tmp"
install -m 755 "$tmp/mmseqs/bin/mmseqs" "$BIN/mmseqs"
install -m 755 "$tmp/foldseek/bin/foldseek" "$BIN/foldseek"
rm -rf "$tmp"
"$BIN/mmseqs" version
"$BIN/foldseek" version
