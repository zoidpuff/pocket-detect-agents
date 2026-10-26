"""AF-mode input variants, AF ranker features and the no-op guarantee (DESIGN §2.6, §9)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pocketdetect.burrow.af import prune_mask
from pocketdetect.burrow.af_rank import AF_FEATURES, af_pocket_features, domain_split
from pocketdetect.burrow.af_variants import CONFIGS, AFConfig, af_input, keep_mask
from pocketdetect.burrow.pockets import Pocket
from pocketdetect.structure import PROTEIN, Structure

ROOT = Path(__file__).resolve().parents[1]
BB = ("N", "CA", "C", "O", "CB", "CG")


def make_structure(plddt_res, coords_res=None, sid="toy"):
    """One chain, residues 1..R of 'LEU'-like atoms N CA C O CB CG on a 3.8 Å-step helix-ish path."""
    R = len(plddt_res)
    if coords_res is None:
        t = np.arange(R)
        coords_res = np.stack([2.3 * np.cos(t * 1.75), 2.3 * np.sin(t * 1.75), 1.5 * t], 1)
    off = np.array([[-1.2, 0.3, 0], [0, 0, 0], [1.2, 0.4, 0], [1.9, 1.3, 0.2], [0, -1.5, 0.3], [0.3, -2.9, 0.5]])
    coords = (coords_res[:, None, :] + off[None]).reshape(-1, 3).astype(np.float32)
    n = len(coords)
    ri = np.repeat(np.arange(R), len(BB)).astype(np.int32)
    names = np.array(list(BB) * R)
    el = np.array([a[0] for a in names])
    b = np.repeat(np.asarray(plddt_res, np.float32), len(BB))
    ch = np.array(["A"] * n)
    rn = np.array(["LEU"] * n)
    num = (ri + 1).astype(np.int32)
    return Structure(id=sid, coords=coords, element=el, atom_name=names, altloc=np.array([""] * n),
                     occupancy=np.ones(n, np.float32), b_factor=b, serial=np.arange(1, n + 1, dtype=np.int32),
                     res_index=ri, chain=ch, res_name=rn, res_num=num, ins_code=np.array([""] * n),
                     is_hetero=np.zeros(n, bool), r_chain=np.array(["A"] * R), r_name=np.array(["LEU"] * R),
                     r_num=np.arange(1, R + 1, dtype=np.int32), r_icode=np.array([""] * R),
                     r_kind=np.full(R, PROTEIN, np.int8), r_hetero=np.zeros(R, bool), r_entity=np.array(["Polymer"] * R))


def toy_plddt():
    # terminal low run (3), confident, internal low run of 7 (pruned), confident, internal low run of 4 (kept), confident
    return [30] * 3 + [95] * 20 + [40] * 7 + [92] * 15 + [45] * 4 + [88] * 10


def test_prune_rule():
    s = make_structure(toy_plddt())
    keep = prune_mask(s)
    assert not keep[:3].any()
    assert keep[3:23].all()
    assert not keep[23:30].any()
    assert keep[30:].all()  # the 4-residue internal run is kept
    assert (~keep).sum() == 10


def test_random_controls_matched():
    s = make_structure(toy_plddt(), sid="toyA")
    af = keep_mask(s, CONFIGS["prune"])
    seg = keep_mask(s, CONFIGS["rand_seg"])
    assert (~seg).sum() == (~af).sum()
    assert np.array_equal(seg, keep_mask(s, CONFIGS["rand_seg"]))  # deterministic per id
    rr = keep_mask(s, CONFIGS["rand_res"])
    cnt = np.bincount(s.res_index)
    assert cnt[~rr].sum() >= cnt[~af].sum()
    assert cnt[~rr].sum() - cnt[~af].sum() < cnt.max()
    crop = keep_mask(s, CONFIGS["crop70"])
    assert (~crop).sum() == 3 + 7 + 4


@pytest.mark.parametrize("name", ["prune", "soft", "prune_soft", "prune_soft_lam0.5low", "prune_soft_lam0.25low", "prune_soft30_70", "prune_soft50_90", "crop70"])
def test_noop_high_plddt(name):
    s = make_structure([95.0] * 60)
    P, w, st = af_input(s, CONFIGS[name])
    P0, w0, _ = af_input(s, CONFIGS["naive"])
    assert w is None and w0 is None
    assert st["n_res_removed"] == 0
    assert np.array_equal(P.coords, P0.coords)


def test_side_chain_lambda():
    s = make_structure([95.0] * 10 + [60.0] * 10)
    P, w, _ = af_input(s, AFConfig(soft=True, lam=0.5, lam_scope="low"))
    sc = ~P.bb_mask
    hi = P.plddt >= 90
    assert np.allclose(w[hi], 1.0)
    assert np.allclose(w[~hi & sc], 0.5 * 0.5)  # (60-40)/40 * λ
    assert np.allclose(w[~hi & ~sc], 0.5)
    P, w, _ = af_input(s, AFConfig(lam=0.25))
    assert np.allclose(w[~P.bb_mask], 0.25) and np.allclose(w[P.bb_mask], 1.0)


def test_af_pocket_features_confident():
    s = make_structure([100.0] * 40)
    P, _, _ = af_input(s, CONFIGS["naive"])
    pts = P.coords[:20] + np.float32(1.5)
    pk = [Pocket(idx=np.arange(10), strength=1.0, center=pts[:10].mean(0), residues=frozenset({"A:1", "A:2"}))]
    F = af_pocket_features(pk, pts, P)
    f = dict(zip(AF_FEATURES, F[0]))
    assert f["af_pl_mean"] == 100 and f["af_pl_min"] == 100
    assert f["af_res_lt70"] == 0 and f["af_pts_lt70"] == 0 and f["af_pts_lt50"] == 0


def test_domain_split_two_blobs():
    # two 4x4x4 lattices (4 Å) whose facing layers are 7 Å apart: 16 inter-domain CA contacts
    g = np.stack(np.meshgrid(*[np.arange(4) * 4.0] * 3, indexing="ij"), -1).reshape(-1, 3)
    s = make_structure([95.0] * 128, coords_res=np.concatenate([g, g + np.array([19.0, 0, 0])]))
    P, _, _ = af_input(s, CONFIGS["naive"])
    part, pack = domain_split(P)
    assert part is not None
    assert len(set(part[:64])) == 1 and len(set(part[64:])) == 1 and part[0] != part[64]
    assert pack == pytest.approx(16 / 64)


def test_noop_real_pdb_dev_input():
    """Pruning + soft occlusion (+ λ restricted to pLDDT < 90) on a PDB-dev chain with uniform pLDDT 95
    gives bit-identical predictions to standard mode."""
    f = ROOT / "results/dev/biolip/inputs/A0A009IHW8__7uxu_B.cif"
    if not f.exists():
        pytest.skip("dev inputs not present")
    from dataclasses import replace

    from pocketdetect.burrow.pipeline import predict_protein
    from pocketdetect.burrow.prep import protein_from_structure
    from pocketdetect.structure import load_structure
    s = load_structure(f, protein_only=True)
    ref = predict_protein(protein_from_structure(s), None)
    for name in ("prune_soft", "prune_soft_lam0.25low"):
        P, w, _ = af_input(s, replace(CONFIGS[name], const_plddt=95.0))
        out = predict_protein(P, None, weights=w)
        assert np.array_equal(out.centers, ref.centers) and np.array_equal(out.scores, ref.scores)
