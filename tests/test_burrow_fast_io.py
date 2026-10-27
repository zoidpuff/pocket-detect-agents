"""Fast loader (burrow.fast.io) == reference protein_from_structure(load_structure(path))."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/dev/biolip/inputs_audited.csv"
TRAIN = ROOT / "data/processed/train/manifest.csv"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="dev data not available")

FIELDS = ("id", "coords", "radii", "element", "atom_name", "res_name", "res_index", "residue_ids", "chain", "plddt",
          "props", "prop_names", "hit_class", "bb_mask", "coord_mask", "coord_sc_mask")


def _same(a, b):
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        return list(a) == list(b) if a.dtype == object else a.tobytes() == b.tobytes()
    return a == b


def _paths():
    from pocketdetect.burrow.devdata import read_manifest
    df = read_manifest(MANIFEST).sample(8, random_state=7)
    out = [str(ROOT / p) for p in df.structure_path]
    if TRAIN.exists():   # whole PDB entries (mmCIF, ligands, waters, alternate locations)
        import pandas as pd
        t = pd.read_csv(TRAIN)
        t = t[(t.status == "ok") & t.structure_path.map(lambda p: Path(p).exists())]
        out += t.sample(min(6, len(t)), random_state=7).structure_path.tolist()
    return out


@pytest.mark.parametrize("path", _paths() if MANIFEST.exists() else [])
def test_fast_loader_identical(path):
    from pocketdetect.burrow.fast.io import load_protein
    from pocketdetect.burrow.prep import protein_from_structure
    from pocketdetect.structure import load_structure
    ref = protein_from_structure(load_structure(path, structure_id="x"))
    fast = load_protein(path, "x")
    bad = [f for f in FIELDS if not _same(getattr(ref, f), getattr(fast, f))]
    assert not bad, bad


def test_fast_loader_fallback_identical():
    """The per-atom fallback (used when FlatStructure guards fail) gives the same Protein."""
    import gemmi
    from pocketdetect.burrow.fast import io as FIO
    for path in _paths()[:4]:
        st = gemmi.read_structure(path, merge_chain_parts=True)
        st.setup_entities()
        a, b = FIO._from_flat(st, "x"), FIO._from_model(st, "x")
        assert a is not None
        assert all(_same(getattr(a, f), getattr(b, f)) for f in FIELDS)
