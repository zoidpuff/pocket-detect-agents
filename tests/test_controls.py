"""Positive / negative controls on synthetic globular proteins with one carved pocket + ligand,
and on a real PDB entry. Positive control: ligand centres as predictions -> 100 % success.
Negative control: random points on the protein surface -> low success."""

import numpy as np
import pytest
from scipy.spatial import cKDTree
from conftest import A, R, build, ligand_res, write

from pocketdetect.evaluate import evaluate, write_predictions
from pocketdetect.ligands import build_ground_truth
from pocketdetect.seeds import SEEDS
from pocketdetect.structure import load_structure, pdb_entry_path

RADIUS = 16.0
N_PROT = 30


def ball_protein(rng):
    """Lattice 'protein' (3 A spacing) in a sphere, pocket carved at depth 3 A, PLUS ligand inside."""
    g = np.arange(-RADIUS, RADIUS + 0.1, 3.0)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts[np.linalg.norm(pts, axis=1) <= RADIUS]
    lig_shape = 1.2 * np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])
    while True:  # resample until the ligand is unambiguously "relevant" under P2Rank rules
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        p0 = (RADIUS - 3.0) * u
        kept = pts[np.linalg.norm(pts - p0, axis=1) > 4.5]
        lig = p0 + lig_shape
        dmin = np.linalg.norm(lig[:, None] - kept[None], axis=-1).min()
        dcen = np.linalg.norm(kept - lig.mean(0), axis=1).min()
        if dmin < 3.9 and dcen < 5.3:
            break
    pts = kept
    rng.shuffle(pts)
    pts = pts[: len(pts) // 5 * 5]
    names = ["N", "CA", "C", "O", "CB"]
    res = [
        R("A", "ALA", i + 1, [A(names[k], names[k][0], tuple(pts[5 * i + k])) for k in range(5)])
        for i in range(len(pts) // 5)
    ]
    res.append(ligand_res("A", "LIG", 900, lig))
    return res, p0, pts


def surface_points(protein_xyz, rng, k, shell=(2.5, 3.5)):
    """Uniform random points in the solvent shell around the protein (nearest protein atom
    2.5-3.5 A away), i.e. on the surface *including* pocket mouths, so the expected hit rate is
    small but not trivially zero."""
    tree = cKDTree(protein_xyz)
    lo, hi = protein_xyz.min(0) - shell[1], protein_xyz.max(0) + shell[1]
    out = np.zeros((0, 3))
    while len(out) < k:
        cand = rng.uniform(lo, hi, size=(4000, 3))
        d, _ = tree.query(cand)
        out = np.vstack([out, cand[(d >= shell[0]) & (d <= shell[1])]])
    return out[:k]


@pytest.fixture(scope="module")
def ball_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("balls")
    rng = np.random.default_rng(SEEDS[0])
    gts, prots = [], {}
    for i in range(N_PROT):
        res, p0, pts = ball_protein(rng)
        path = write(build(res), d / f"ball{i:02d}.cif")
        gt = build_ground_truth(path, structure_id=f"ball{i:02d}")
        gts.append(gt)
        prots[gt.structure_id] = pts
    return d, gts, prots


def test_ball_ground_truth(ball_dataset):
    _, gts, _ = ball_dataset
    assert all(g.n_ligands == 1 for g in gts), [ (g.structure_id, [(l.category, l.contact_dist, l.center_dist) for l in g.other_ligands]) for g in gts if g.n_ligands != 1]


def test_positive_control(ball_dataset, tmp_path):
    _, gts, _ = ball_dataset
    for g in gts:
        write_predictions(tmp_path / f"{g.structure_id}.csv", [l.center for l in g.ligands])
    res = evaluate(gts, tmp_path, criteria=["DCA4", "DCC4", "LDCC4"], n_boot=100)
    s = res.summary.set_index(["metric", "average"])["value"]
    for crit in ("DCA4", "DCC4", "LDCC4"):
        for k in ("top1", "topn", "topn2", "top10"):
            assert s[(f"success_{crit}_{k}", "ligand")] == 1.0
            assert s[(f"success_{crit}_{k}", "protein")] == 1.0
    assert s[("ap", "protein")] == 1.0 and s[("ap_pooled", "pooled")] == 1.0
    assert s[("pocket_precision_top1", "protein")] == 1.0


def test_negative_control(ball_dataset, tmp_path):
    _, gts, prots = ball_dataset
    rates = []
    for seed in SEEDS[:3]:
        rng = np.random.default_rng(seed)
        out = tmp_path / f"neg{seed}"
        out.mkdir()
        for g in gts:
            write_predictions(out / f"{g.structure_id}.csv", surface_points(prots[g.structure_id], rng, 10))
        res = evaluate(gts, out, n_boot=0)
        s = res.summary.set_index(["metric", "average"])["value"]
        rates.append((s[("success_DCA4_top1", "ligand")], s[("success_DCA4_topn2", "ligand")], s[("success_DCA4_top10", "ligand")]))
    r = np.mean(rates, axis=0)
    # measured (seeds 20261026,1,2): top1 ~0.06, top-(n+2) ~0.11, top10 ~0.26
    assert r[0] <= 0.15 and r[1] <= 0.3, r
    assert 0.0 < r[2] < 0.6, r  # chance hits occur (control is not trivially 0) but stay rare


def test_controls_on_real_entry(mirror_available, tmp_path):
    """1m17 (EGFR kinase + erlotinib/AQ4): ligand centroid -> success; random surface points -> rarely."""
    p = pdb_entry_path("1m17")
    gt = build_ground_truth(p, structure_id="1m17")
    assert [l.name for l in gt.ligands] == ["AQ4"]
    pos = tmp_path / "pos"
    pos.mkdir()
    write_predictions(pos / "1m17.csv", [gt.ligands[0].center])
    s = evaluate([gt], pos, n_boot=0).summary.set_index(["metric", "average"])["value"]
    assert s[("success_DCA4_top1", "ligand")] == 1.0 and s[("success_DCC4_top1", "ligand")] == 1.0
    prot = load_structure(p, protein_only=True).coords.astype(float)
    hits = []
    for seed in SEEDS:
        neg = tmp_path / f"neg{seed}"
        neg.mkdir()
        write_predictions(neg / "1m17.csv", surface_points(prot, np.random.default_rng(seed), 1))
        s = evaluate([gt], neg, n_boot=0).summary.set_index(["metric", "average"])["value"]
        hits.append(s[("success_DCA4_top1", "ligand")])
    assert np.mean(hits) <= 0.2  # measured chance rate over 200 seeds: 0.02
