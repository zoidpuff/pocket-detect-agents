"""Reference-drift guard.

The fast stages replicate specific reference functions.  Their source hashes at validation time are
stored in ``validated.json``; if the reference code changes (the method is still evolving), the
affected fast stage is disabled and the pipeline falls back to the reference implementation with a
warning, so results stay correct.  Re-validate (tests/test_burrow_fast_*.py) and run
``python -m pocketdetect.burrow.fast.guard --update`` to re-enable (several validated reference
versions can be listed per stage, e.g. the frozen v1 code and a later working tree).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

STAGES = {
    "features": ["burrow.features:point_features", "burrow.features:metal_candidates",
                 "burrow.features:BurrowConfig", "burrow.prep:protein_from_structure",
                 "burrow.raycast:march", "burrow.raycast:dual_grid", "geometry.neighbors:neighbor_features",
                 "geometry.neighbors:_nf_kernel"],
    "formation": ["burrow.model:candidates", "burrow.model:geo_value", "burrow.model:geo_score",
                  "burrow.model:longest_bur",
                  "burrow.pockets:components", "burrow.pockets:generate", "burrow.pockets:make_pocket",
                  "burrow.pockets:_kmeans_split", "burrow.pockets:_extent", "burrow.pockets:PointGraph",
                  "burrow.pockets:assign_residues", "burrow.pockets:nms", "burrow.model:FormationConfig"],
    "pocket_features": ["burrow.model:pocket_features", "burrow.model:_seg_sorted", "burrow.model:_seg_quantile",
                        "burrow.model:_seg_topk_mean"],
    "scoring": ["burrow.model:score_points", "burrow.pipeline:predict_protein", "burrow.pipeline:finalize",
                "burrow.pipeline:output_residues", "burrow.model:rank_predict", "burrow.model:BurrowModel"],
}
_FILE = Path(__file__).with_name("validated.json")


def _source(ref: str) -> str:
    import importlib
    mod, name = ref.split(":")
    m = importlib.import_module(f"pocketdetect.{mod}")
    obj = getattr(m, name, None)
    if obj is None:
        return "<missing>"
    obj = getattr(obj, "py_func", obj)
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return repr(obj)


def current_hashes() -> dict:
    return {st: hashlib.sha256("\n".join(_source(r) for r in refs).encode()).hexdigest()[:16]
            for st, refs in STAGES.items()}


@lru_cache(maxsize=None)
def _status() -> dict:
    try:
        val = json.loads(_FILE.read_text())
    except FileNotFoundError:
        val = {}
    cur = current_hashes()
    acc = lambda v: v if isinstance(v, list) else [v]       # several validated reference versions allowed
    ok = {st: h in acc(val.get(st)) for st, h in cur.items()}
    bad = [st for st, v in ok.items() if not v]
    if bad:
        warnings.warn(f"burrow.fast: reference code changed since validation for stages {bad}; "
                      "falling back to the reference implementation for them", RuntimeWarning, stacklevel=3)
    return ok


def stage_ok(stage: str) -> bool:
    return _status().get(stage, False)


def update(keep_old: bool = True):
    """Record the current reference hashes as validated (keeping previously validated ones)."""
    try:
        val = json.loads(_FILE.read_text())
    except FileNotFoundError:
        val = {}
    out = {}
    for st, h in current_hashes().items():
        old = val.get(st, [])
        old = (old if isinstance(old, list) else [old]) if keep_old else []
        out[st] = old + [h] if h not in old else old
    _FILE.write_text(json.dumps(out, indent=1) + "\n")
    _status.cache_clear()


if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
    print(json.dumps(current_hashes(), indent=1))
