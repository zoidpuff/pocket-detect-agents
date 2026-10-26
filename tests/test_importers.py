"""Tests for the CryptoBench / PocketWatch ground-truth importers.

Unit tests run everywhere; data tests are skipped when the raw downloads or the built site tables
(scripts/build_gt_*.py) are absent. Independent recomputations use gemmi / scipy directly rather
than the importer code paths."""

from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from pocketdetect.importers.cryptobench import input_chains, merge_pockets, parse_selection_residue
from pocketdetect.importers.pocketwatch import kabsch, project_centroid
from pocketdetect.sites import read_sites, sites_by_structure

ROOT = Path(__file__).resolve().parents[1]
CB_RAW = ROOT / "data/raw/cryptobench"
CB_DS = CB_RAW / "osf/cryptobench/cryptobench-dataset"
CB_CIF = CB_RAW / "cif-files/cif-files"
CB_TSV = ROOT / "data/processed/gt/cryptobench_test/sites.tsv"
PW_RAW = ROOT / "data/raw/pocketwatch/current"
PW_TSV = ROOT / "data/processed/gt/pocketwatch_test/sites.tsv"

needs_cb = pytest.mark.skipif(not (CB_TSV.exists() and CB_DS.exists() and CB_CIF.exists()), reason="CryptoBench data / table absent")
needs_pw = pytest.mark.skipif(not (PW_TSV.exists() and PW_RAW.exists()), reason="PocketWatch data / table absent")


# ----------------------------------------------------------------------------- unit tests
def test_merge_pockets_overlap_rule():
    a = [f"A_{i}" for i in range(1, 11)]  # 10 residues
    b = [f"A_{i}" for i in range(3, 11)] + ["A_50"]  # 8/9 of b in a -> merge
    c = [f"A_{i}" for i in range(100, 110)]  # disjoint
    d = [f"A_{i}" for i in range(1, 4)] + ["A_200", "A_201"]  # 3/5 = 0.6 -> no merge
    out = merge_pockets([a, b, c, d])
    assert len(out) == 3
    assert set(out[0]) == set(a) | set(b)
    assert set(out[1]) == set(c) and set(out[2]) == set(d)
    # exactly 75 % overlap does not merge (strict ">")
    e = ["A_1", "A_2", "A_3", "A_4"]
    f = ["A_1", "A_2", "A_3", "B_9", "B_10"]
    assert len(merge_pockets([e, f])) == 2
    assert len(merge_pockets([e, ["A_1", "A_2", "A_3", "A_4", "B_1"]])) == 1


def test_merge_pockets_transitive():
    """b overlaps c by exactly 75 % (no merge) but merges with the grown union a|c in pass 2."""
    a = ["A_1", "A_2", "A_3", "A_4", "A_5"]
    b = ["A_1", "A_6", "A_7", "A_8"]
    c = ["A_2", "A_3", "A_4", "A_5", "A_6", "A_7", "A_8"]  # 4/5 of a -> merges in pass 1
    out = merge_pockets([a, b, c])
    assert len(out) == 1 and set(out[0]) == set(a) | set(b) | set(c)


def test_parse_selection_residue_and_chains():
    assert parse_selection_residue("B_12") == "B:12"
    assert parse_selection_residue("AAA_7") == "AAA:7"
    assert parse_selection_residue("d_-3") == "d:-3"
    holos = [{"apo_chain": "C"}, {"apo_chain": "C-O"}, {"apo_chain": "O"}]
    assert input_chains(holos) == ["C", "O"]
    with pytest.raises(ValueError):
        input_chains([{"apo_chain": "A"}, {"apo_chain": "B"}])  # union not covered by one entry


def test_kabsch_projection_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3)) * 5
    rot = Rotation.from_euler("zyx", [30, -50, 110], degrees=True).as_matrix()
    t = np.array([3.0, -7.0, 11.0])
    tgt = src @ rot.T + t
    r, tr = kabsch(src, tgt)
    assert np.allclose(r, rot, atol=1e-8) and np.allclose(tr, t, atol=1e-8)
    lig = np.array([1.0, 2.0, 3.0])
    xyz, rmsd = project_centroid(src, tgt, lig)
    assert np.allclose(xyz, rot @ lig + t) and rmsd < 1e-8
    assert project_centroid(src[:2], tgt[:2], lig) is None


# ----------------------------------------------------------------------------- helpers
def _gemmi_residues(path, chains=None):
    """{residue_id: {'ca': xyz or None, 'heavy': (k,3)}} for polymer residues, first model,
    first altloc, independent of pocketdetect.structure."""
    st = gemmi.read_structure(str(path), merge_chain_parts=True)
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    out = {}
    for ch in st[0]:
        if chains is not None and ch.name not in chains:
            continue
        for r in ch:
            icode = r.seqid.icode.strip()
            rid = f"{ch.name}:{r.seqid.num}{icode}"
            ca = r.find_atom("CA", "*")
            out[rid] = {
                "name": r.name,
                "ca": None if ca is None else np.array(ca.pos.tolist()),
                "heavy": np.array([a.pos.tolist() for a in r]),
            }
    return out


# ----------------------------------------------------------------------------- CryptoBench
@pytest.fixture(scope="module")
def cb():
    return read_sites(CB_TSV)


@needs_cb
def test_cryptobench_counts(cb):
    with open(CB_DS / "folds.json") as fh:
        test_ids = set(json.load(fh)["test"])
    assert cb["structure_id"].nunique() == 222
    assert set(cb["apo_pdb_id"]) == test_ids
    assert len(cb) == 270
    assert cb["scored"].all()
    per = cb.groupby("structure_id").agg(n=("site_id", "size"), n_sites=("n_sites", "first"), u=("n_sites", "nunique"))
    assert (per["u"] == 1).all() and (per["n"] == per["n_sites"]).all()
    assert not cb[["center_x", "center_y", "center_z", "center_heavy_x"]].isna().any().any()
    assert (cb["n_residues_unmapped"] == 0).all()
    assert all(Path(p).is_absolute() and Path(p).exists() for p in cb["structure_path"].unique())
    sites = sites_by_structure(cb)
    assert sum(len(v) for v in sites.values()) == 270


@needs_cb
def test_cryptobench_known_site_1arl(cb):
    """1arl (carboxypeptidase A, chain A): its 4 holo entries merge into one site; recompute
    residues and both centres from the raw JSON + gemmi."""
    with open(CB_DS / "folds/test.json") as fh:
        holos = json.load(fh)["1arl"]
    union = sorted({t for h in holos for t in h["apo_pocket_selection"]}, key=lambda t: int(t.split("_")[1]))
    expected = [t.replace("_", ":") for t in union]
    row = cb[cb["structure_id"] == "1arl_A"]
    assert len(row) == 1 and int(row["n_sites"].iloc[0]) == 1
    row = row.iloc[0]
    assert row["residues"].split() == expected
    res = _gemmi_residues(CB_CIF / "1arl.cif", chains={"A"})
    ca = np.mean([res[r]["ca"] for r in expected], axis=0)
    heavy = np.concatenate([res[r]["heavy"] for r in expected]).mean(axis=0)
    assert np.allclose([row.center_x, row.center_y, row.center_z], ca, atol=2e-3)
    assert np.allclose([row.center_heavy_x, row.center_heavy_y, row.center_heavy_z], heavy, atol=2e-3)
    assert row["n_residues_structure"] == sum(1 for r in res.values() if r["ca"] is not None and gemmi.find_tabulated_residue(r["name"]) is not None and gemmi.find_tabulated_residue(r["name"]).is_amino_acid())


@needs_cb
def test_cryptobench_residues_exist_and_centres(cb):
    """Every site residue exists (author numbering) in the input chains of the apo file, and every
    CA centre matches an independent gemmi recomputation."""
    for sid, g in cb.groupby("structure_id"):
        chains = set(g["input_chains"].iloc[0].split("-"))
        res = _gemmi_residues(g["structure_path"].iloc[0], chains=chains)
        for row in g.itertuples():
            rids = row.residues.split()
            assert all(r in res for r in rids), (sid, [r for r in rids if r not in res])
            assert {r.rpartition(":")[0] for r in rids} <= chains
            ca = np.mean([res[r]["ca"] for r in rids], axis=0)
            assert np.allclose([row.center_x, row.center_y, row.center_z], ca, atol=2e-3), sid


@needs_cb
def test_cryptobench_multichain_input():
    df = read_sites(CB_TSV)
    r = df[df["structure_id"] == "1fd4_G-H"]
    assert len(r) == 1 and r["input_chains"].iloc[0] == "G-H"
    assert set(x.split(":")[0] for x in r["residues"].iloc[0].split()) <= {"G", "H"}


# ----------------------------------------------------------------------------- PocketWatch
@pytest.fixture(scope="module")
def pw():
    return read_sites(PW_TSV)


@needs_pw
def test_pocketwatch_counts(pw):
    assert pw["structure_id"].nunique() == 1566
    assert len(pw) == 2301
    assert int(pw["scored"].sum()) == 1573
    assert pw[pw["scored"]]["structure_id"].nunique() == 1279
    per = pw.groupby("structure_id").agg(n_sites=("n_sites", "first"), u=("n_sites", "nunique"), k=("scored", "sum"))
    assert (per["u"] == 1).all() and (per["n_sites"] == per["k"]).all()
    assert not pw.loc[pw["scored"], ["center_x", "center_y", "center_z"]].isna().any().any()
    assert not pw[["center_x", "center_y", "center_z"]].isna().any().any()
    assert set(pw.loc[~pw["scored"], "mask_reason"]) <= {"interface_only", "pre_cutoff", "subsumed_by_seen"}
    assert (pw.loc[pw["scored"], "mask_reason"] == "").all()
    pol = pd.read_parquet(PW_RAW / "data/site_policy.parquet")
    pol = pol[pol["split"] == "test"].set_index("pocket_site_id")
    assert (pol.loc[pw["site_id"], "decision"].eq("scored").to_numpy() == pw["scored"].to_numpy()).all()


@needs_pw
def test_pocketwatch_centres_match_shipped(pw):
    ship = pd.read_parquet(PW_RAW / "data/site_centroids.parquet").set_index("pocket_site_id")
    ref = ship.loc[pw["site_id"], ["ligand_centroid_x", "ligand_centroid_y", "ligand_centroid_z"]].to_numpy()
    got = pw[["center_x", "center_y", "center_z"]].to_numpy()
    assert np.abs(ref - got).max() < 1e-3  # table written with 3 decimals
    assert (ship.loc[pw["site_id"], "centroid_member_count"].to_numpy() == pw["centroid_member_count"].to_numpy()).all()


@needs_pw
def test_pocketwatch_independent_projection(pw):
    """Recompute a deterministic sample of site centroids from projection/ + the folds with gemmi
    and scipy (Rotation.align_vectors), independently of the importer."""
    lig = pd.read_parquet(PW_RAW / "projection/ligand_centroids.parquet")
    cr = pd.read_parquet(PW_RAW / "projection/contact_residues.parquet")
    sample = pw.sort_values("site_id").iloc[::97]
    assert len(sample) >= 20
    for row in sample.itertuples():
        res = _gemmi_residues(row.structure_path)
        L = row.n_residues_structure
        assert list(res) == [f"A:{i}" for i in range(1, L + 1)]
        for r in row.residues.split():
            assert r in res
        pts = []
        for m in lig[lig["pocket_site_id"] == row.site_id].itertuples():
            c = cr[cr["pocket_observation_id"] == m.pocket_observation_id]
            pos = c["canonical_position"].to_numpy()
            src = c[["ca_x", "ca_y", "ca_z"]].to_numpy(dtype=float)
            keep = (pos >= 1) & (pos <= L)
            pos, src = pos[keep], src[keep]
            tgt = np.array([res[f"A:{p}"]["ca"] if res[f"A:{p}"]["ca"] is not None else [np.nan] * 3 for p in pos], dtype=np.float32).astype(float)
            ok = ~np.isnan(tgt).any(axis=1)
            src, tgt = src[ok], tgt[ok]
            if len(src) < 3:
                continue
            rot, _ = Rotation.align_vectors(tgt - tgt.mean(0), src - src.mean(0))
            lc = np.array([m.ligand_centroid_x, m.ligand_centroid_y, m.ligand_centroid_z])
            pts.append(rot.apply(lc - src.mean(0)) + tgt.mean(0))
        assert pts, row.site_id
        assert np.allclose(np.mean(pts, axis=0), [row.center_x, row.center_y, row.center_z], atol=2e-3), row.site_id
        ca = np.mean([res[r]["ca"] for r in row.residues.split()], axis=0)
        assert np.allclose(ca, [row.center_ca_x, row.center_ca_y, row.center_ca_z], atol=2e-3)


@needs_pw
def test_pocketwatch_label_residues_match_labels(pw):
    """Scored sites' label_residues are positives in pocket_labels.parquet (that file's labels also
    include interface-only positions, so only inclusion is checked), and every residue of every
    site lies within 1..L of its fold."""
    lab = pd.read_parquet(PW_RAW / "data/pocket_labels.parquet").set_index("normalized_uniprot_id")
    for uid, g in pw.groupby("structure_id"):
        y = np.asarray(lab.loc[uid, "label"])
        pos = set(np.flatnonzero(y == 1) + 1)
        ours = {int(r.split(":")[1]) for s in g.loc[g["scored"], "label_residues"] for r in s.split()}
        assert ours <= pos, uid
        allpos = [int(r.split(":")[1]) for s in g["residues"] for r in s.split()]
        assert all(r.startswith("A:") for s in g["residues"] for r in s.split())
        assert min(allpos) >= 1 and max(allpos) <= len(y) == int(g["n_residues_structure"].iloc[0]), uid
