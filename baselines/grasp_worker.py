#!/usr/bin/env python
"""GrASP worker -- runs INSIDE the py3.7 GrASP env (baselines/tools/grasp/env), not the project venv.
Called by baselines/run_grasp.py. Python-3.7 compatible code only.

  grasp_worker.py --list files.txt --work DIR --out result.json [--procs N] [--device cuda:0]

files.txt: one cleaned PDB path per line (stem = file stem).  Pipeline (mirrors GrASP 'production' mode):
  1. parse_files.convert_to_mol2 (MDAnalysis protein/20-aa/CNOS filter + OpenBabel 2.4.1 CorrectForPH+AddHydrogens)
  2. featurize_protein.process_system (rdkit features, mdtraj SASA, graph)
  3. GASPData (k_hops=1 surface subgraph, sasa>1e-4) -> GATv2 model (trained_models/train_full/.../epoch_49)
  4. site clustering with the GrASP paper defaults (site_metrics.py CLI defaults): average-linkage agglomerative
     clustering (distance threshold 15 A) of surface atoms with p>0.3, site score = sum(p^2), site centre = convex-hull
     centre of mass (centroid if <4 atoms).
Output json: {stem: {status, note, pockets:[{rank,score,center,residues:[chain:resnum..],points:[[x,y,z],..]}],
                     timing:{parse_s,infer_s,cluster_s}}}
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tools", "grasp", "src")
sys.path.insert(0, SRC)
os.chdir(SRC)  # GrASP code uses cwd-relative paths in places

import numpy as np  # noqa: E402

MODEL = "train_full/trained_model_s_train_full_ag_multi_1680643832.8660116/cv_0/epoch_49"


def parse_one(args):
    path, work = args
    stem = os.path.splitext(os.path.basename(path))[0]
    t = time.time()
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from parse_files import convert_to_mol2
        from featurize_protein import process_system
        mol2_dir = os.path.join(work, "ready_to_parse_mol2") + "/"
        convert_to_mol2(path, stem, mol2_dir, addH=True, in_format="pdb", out_name="protein", parse_prot=True)
        r = process_system(mol2_dir + stem, save_directory=work, parse_ligands=False)
        if r is not None or not os.path.exists(os.path.join(work, "raw", stem + ".npz")):
            return stem, "featurization failed (code %s)" % (r,), time.time() - t
        return stem, None, time.time() - t
    except Exception as e:  # noqa
        return stem, "%s: %s" % (type(e).__name__, str(e)[:200]), time.time() - t


def read_pdb_atoms(path):
    """heavy atoms of cleaned PDB -> coords, (chain,resnum,icode) labels"""
    xyz, lab = [], []
    for l in open(path):
        if l.startswith(("ATOM", "HETATM")):
            xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
            lab.append((l[21], int(l[22:26]), l[26].strip()))
    return np.array(xyz), lab


def hull_center(points):
    from scipy.spatial import ConvexHull, Delaunay
    hull = ConvexHull(points)
    com = np.zeros(3)
    tet = Delaunay(points[hull.vertices])
    for s in tet.simplices:
        a, b, c, d = tet.points[s]
        vol = abs(np.linalg.det([a - d, b - d, c - d])) / 6
        com += np.mean([a, b, c, d], axis=0) * vol
    return com / hull.volume


def cluster_sites(coords, probs, threshold=0.3, eps=15.0):
    """GrASP site_metrics defaults: average linkage, distance_threshold=eps, score = sum(p^2). returns sites sorted best-first"""
    from sklearn.cluster import AgglomerativeClustering
    sel = probs > threshold
    if sel.sum() == 0:
        return []
    bc = coords[sel]
    bp = probs[sel]
    if len(bc) == 1:
        ids = np.zeros(1, dtype=int)
    else:
        ids = AgglomerativeClustering(linkage="average", n_clusters=None, distance_threshold=eps).fit(bc).labels_
    sites = []
    for c in np.unique(ids):
        m = ids == c
        pts = bc[m]
        score = float(np.sum(bp[m] ** 2))
        center = hull_center(pts) if len(pts) >= 4 else pts.mean(0)
        sites.append(dict(score=score, center=center, idx=np.where(sel)[0][m]))
    sites.sort(key=lambda s: -s["score"])
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=15.0)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    files = [l.strip() for l in open(a.list) if l.strip()]
    work = os.path.abspath(a.work)
    for d in ("raw", "ready_to_parse_mol2", "mol2"):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    res = {}

    # ---- 1/2 parse + featurize (CPU, parallel over structures)
    t0 = time.time()
    jobs = [(f, work) for f in files]
    if a.procs > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            out = p.map(parse_one, jobs, chunksize=1)
    else:
        out = [parse_one(j) for j in jobs]
    parse_s = {}
    ok = []
    for (stem, err, dt), f in zip(out, files):
        parse_s[stem] = dt
        if err:
            res[stem] = dict(status="error", note=err, pockets=[], timing=dict(parse_s=dt))
        else:
            ok.append((stem, f))
    t_parse = time.time() - t0
    if not ok:
        json.dump(res, open(a.out, "w"))
        return

    # ---- 3 inference
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from GASP_dataset import GASPData
    from utils import initialize_model

    class A:  # initialize_model args
        model = "gatv2"
        weight_groups = 1
        group_layers = 12
        aggregator = "multi"

    dev = torch.device(a.device if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    model = initialize_model(A)
    model.load_state_dict(torch.load(os.path.join(SRC, "trained_models", a.model), map_location=dev))
    model.to(dev).eval()
    t1 = time.time()
    ds = GASPData(work, max(1, a.procs), cutoff=5, surface_subgraph_hops=1, sasa_threshold=1e-4)
    ds.process()
    t_graph = time.time() - t1
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    names = ds.raw_file_names
    labels = dict(ok)
    with torch.no_grad():
        for (batch, name) in loader:
            stem = name[0][:-4]
            ti = time.time()
            try:
                out_, _ = model(batch.to(dev))
                sm = batch.surf_mask
                probs = F.softmax(out_[sm], dim=-1)[:, 1].cpu().numpy()
                node_idx = batch.atom_index[sm].cpu().numpy()
                infer_s = time.time() - ti
                # coordinates of graph nodes (heavy atoms, same order as raw coords)
                coords = batch.coords[sm].cpu().numpy()
                tc = time.time()
                sites = cluster_sites(coords, probs, a.threshold, a.eps)
                # map graph atoms back to the cleaned PDB atoms by coordinates
                pxyz, plab = read_pdb_atoms(labels[stem])
                from scipy.spatial import cKDTree
                dist, nn = cKDTree(pxyz).query(coords)
                if dist.max() > 0.05:
                    raise RuntimeError("atom mapping failed (max dist %.3f)" % dist.max())
                pockets = []
                for r, s in enumerate(sites, 1):
                    idx = s["idx"]
                    resid = sorted(set(plab[nn[i]] for i in idx))
                    pockets.append(dict(rank=r, score=s["score"], center=[float(v) for v in s["center"]],
                                        residues=["%s:%d%s" % (c, n, ic) for c, n, ic in resid],
                                        points=[[float(v) for v in coords[i]] for i in idx]))
                res[stem] = dict(status="ok", note="%d surface atoms, %d with p>%.2f" % (len(probs), int((probs > a.threshold).sum()), a.threshold),
                                 pockets=pockets,
                                 timing=dict(parse_s=parse_s[stem], infer_s=infer_s, cluster_s=time.time() - tc,
                                             n_atoms_graph=int(len(batch.x))))
            except Exception as e:  # noqa
                res[stem] = dict(status="error", note="%s: %s" % (type(e).__name__, str(e)[:200]), pockets=[],
                                 timing=dict(parse_s=parse_s.get(stem, 0)), tb=traceback.format_exc())
    for k in res:
        if "timing" in res[k]:
            res[k]["timing"].update(batch_parse_wall_s=t_parse, batch_graph_s=t_graph, batch_n=len(ok))
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
