#!/usr/bin/env python
"""Input files for baselines on AF-dev models (identical inputs for every method).

  results/dev_af/inputs/full/<structure_id>.cif.gz     symlink to the AF model
  results/dev_af/inputs/pruned/<structure_id>.cif      AF model after BURROW spaghetti pruning
                                                       (pocketdetect.burrow.af.prune_mask), B-factors kept
plus full.list / pruned.list.  (DESIGN §2.6: baselines also run on our pruned input.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketdetect.burrow.af_variants import CONFIGS, keep_mask  # noqa: E402
from pocketdetect.sites import read_sites  # noqa: E402
from pocketdetect.structure import load_structure  # noqa: E402

OUT = ROOT / "results/dev_af/inputs"


def main(sites=ROOT / "results/dev_af/af_dev_sites_all.tsv"):
    st = read_sites(sites).drop_duplicates("structure_id")
    for d in ("full", "pruned"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    (OUT / ".gitignore").write_text("*\n")
    full, pruned = [], []
    for sid, path in zip(st.structure_id, st.structure_path):
        f = OUT / "full" / f"{sid}.cif.gz"
        if not f.exists():
            f.symlink_to(path)
        full.append(str(f))
        s = load_structure(path, structure_id=sid, protein_only=True)
        s = s.select(s.protein_heavy_mask())
        keep = keep_mask(s, CONFIGS["prune"])
        drop = {(c, int(n)) for c, n in zip(s.r_chain[~keep], s.r_num[~keep])}
        g = gemmi.read_structure(path)
        g.remove_hydrogens()
        for ch in g[0]:
            for i in reversed(range(len(ch))):
                if (ch.name, ch[i].seqid.num) in drop:
                    del ch[i]
        p = OUT / "pruned" / f"{sid}.cif"
        g.make_mmcif_document().write_file(str(p))
        pruned.append(str(p))
        n_left = sum(len(ch) for ch in g[0])
        assert n_left == int(keep.sum()), (sid, n_left, int(keep.sum()))
    (OUT / "full.list").write_text("\n".join(full) + "\n")
    (OUT / "pruned.list").write_text("\n".join(pruned) + "\n")
    print(f"{len(full)} inputs; pruned files written")


if __name__ == "__main__":
    main()
