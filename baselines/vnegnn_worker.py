#!/usr/bin/env python
"""VN-EGNN worker -- runs in baselines/tools/vnegnn/venv (torch 2.4, PyG 2.4, lightning 2.2, fair-esm).

Uses the ORIGINAL release (ml-jku/vnegnn commit 1407eec, April 2024) + HF checkpoint fses91/VNEGNN-MODEL/best_model.ckpt
(the 2025 'restructure' commit of the repo ships no weights and a different architecture -> not used).

  vnegnn_worker.py --list files.txt --out res.json [--device cuda:0] [--seed 0] [--res-radius 6.0]

Pipeline (= predict.py of the release, but batched over structures with models loaded once):
  1. nodes = residues with a CA atom (standard + a few modified residues; unknown -> 'X'); coords = CA
  2. ESM2-650M (layer 33) embedding of the sequence (all chains concatenated like the original; if the total length >
     --max-esm-len the embedding is computed per chain instead)
  3. HeteroGraph (kNN<=10 within 10 A, 8 virtual nodes on a randomly rotated sphere around the centroid), 1 forward pass
  4. MeanShift on concat[pos, confidence] of the 8 virtual nodes -> pockets, ranked by (mean) confidence
  5. residues of a pocket := residues with any heavy atom within --res-radius of the pocket centre (derived; VN-EGNN
     does not predict residues)
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tools", "vnegnn", "src_v1")
CKPT = os.path.join(HERE, "tools", "vnegnn", "best_model.ckpt")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common modified residues -> parent
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", "HYP": "P", "MLY": "K", "KCX": "K", "CME": "C",
    "SEC": "C", "PCA": "E", "HIP": "H", "HID": "H", "HIE": "H",
}


class Info:
    """duck-typed ProteinInfoInference"""
    pocket_center = None
    ligand_coords = None


def read_pdb(path):
    """returns residue list [(chain,resnum,icode,resname)], CA coords (n,3), all heavy atom coords + residue index"""
    res, ca, allxyz, allres = [], [], [], []
    cur = None
    for l in open(path):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        key = (l[21], int(l[22:26]), l[26].strip(), l[17:20].strip())
        if key != cur:
            res.append(key)
            ca.append(None)
            cur = key
        xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        allxyz.append(xyz)
        allres.append(len(res) - 1)
        if l[12:16].strip() == "CA":
            ca[-1] = xyz
    return res, ca, np.array(allxyz), np.array(allres)


def build_info(path, name, esm_model, converter, dev, max_len, chain=None, parsed=None):
    res, ca, allxyz, allres = parsed if parsed is not None else read_pdb(path)
    keep = [i for i, c in enumerate(ca) if c is not None and (chain is None or res[i][0] == chain)]
    if not keep:
        raise ValueError("no CA atoms")
    info = Info()
    info.name = name
    info.kept_res = [res[i] for i in keep]
    info.coordiantes = np.array([ca[i] for i in keep])
    info.res_names = [res[i][3] for i in keep]
    info.chains = [res[i][0] for i in keep]
    info.res_numbers = [res[i][1] for i in keep]
    info.atom_names = ["CA"] * len(keep)
    seq = "".join(THREE2ONE.get(n, "X") for n in info.res_names)
    info.seq = seq
    feats = []
    if len(seq) <= max_len:
        segs = [(0, len(seq))]
    else:  # per chain
        segs, s = [], 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or info.chains[i] != info.chains[s]:
                segs.append((s, i))
                s = i
    for a, b in segs:
        _, _, toks = converter([("p", seq[a:b])])
        with torch.no_grad():
            out = esm_model(toks.to(dev), repr_layers=[33])
        feats.append(out["representations"][33][0][1:-1].float().cpu().numpy())
    info.res_features = np.concatenate(feats, 0)
    assert len(info.res_features) == len(seq)
    info.all_xyz, info.all_res = allxyz, allres
    info.res_all = res
    info.keep_idx = keep
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res-radius", type=float, default=6.0)
    ap.add_argument("--max-esm-len", type=int, default=3000)
    ap.add_argument("--split-chains", action="store_true", help="predict each chain separately and pool the virtual nodes (authors' HOLO4K protocol; MeanShift is run per chain, sites pooled and ranked by confidence)")
    a = ap.parse_args()
    files = [l.strip() for l in open(a.list) if l.strip()]

    import esm
    from sklearn.cluster import MeanShift
    from scipy.spatial import cKDTree
    from src.models.egnn import EGNNClassifierGlobalNodeHetero
    from src.utils.graph import HeteroGraphBuilder

    dev = torch.device(a.device if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    t0 = time.time()
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.to(dev).eval()
    conv = alphabet.get_batch_converter()
    model = EGNNClassifierGlobalNodeHetero.load_from_checkpoint(CKPT, map_location=dev, strict=False)
    model.to(dev).eval()
    load_s = time.time() - t0

    res = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            t1 = time.time()
            parsed = read_pdb(f)
            chains = sorted(set(r[0] for r, c in zip(parsed[0], parsed[1]) if c is not None))
            units = [None] if (not a.split_chains or len(chains) == 1) else chains
            pos_l, conf_l, t_esm, t_model = [], [], 0.0, 0.0
            for ch in units:
                t1 = time.time()
                info = build_info(f, stem, esm_model, conv, dev, a.max_esm_len, chain=ch, parsed=parsed)
                if len(info.res_names) < 10 and ch is not None:
                    continue
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_esm += time.time() - t1
                t2 = time.time()
                g = HeteroGraphBuilder(protein_info=info, neigh_dist_cutoff=10, max_neighbours=10,
                                       number_of_global_nodes=8, esm_features=True).create_graph()
                if isinstance(g, str):
                    raise RuntimeError("graph construction failed")
                g = g.to(dev)
                with torch.no_grad():
                    _, pos, _, conf = model(g)
                pos_l.append(pos.cpu().numpy())
                conf_l.append(conf.cpu().numpy().reshape(-1, 1))
                t_model += time.time() - t2
            if not pos_l:
                raise RuntimeError("no chain with >=10 residues")
            n_res_total = len(parsed[1])
            # all-atom coordinates / residue table of the WHOLE structure for residue assignment
            info.all_xyz, info.all_res, info.res_all = parsed[2], parsed[3], parsed[0]
            t3 = time.time()
            # original clustering (src.utils.common.get_clusterd_predictions)
            # MeanShift per prediction unit (whole structure, or per chain with --split-chains)
            sites = []
            tree = cKDTree(info.all_xyz)
            for pu, cu in zip(pos_l, conf_l):
                labels = MeanShift().fit_predict(np.concatenate([pu, cu], axis=1))
                for lab in np.unique(labels):
                    m = labels == lab
                    c = pu[m].mean(0)
                    sc = float(cu[m].mean())
                    idx = tree.query_ball_point(c, a.res_radius)
                    rr = sorted(set(int(info.all_res[i]) for i in idx))
                    residues = ["%s:%d%s" % (info.res_all[r][0], info.res_all[r][1], info.res_all[r][2]) for r in rr]
                    sites.append(dict(score=sc, center=[float(v) for v in c], residues=residues,
                                      points=[[float(v) for v in p] for p in pu[m]], n_vn=int(m.sum())))
            sites.sort(key=lambda s: -s["score"])
            for i, s in enumerate(sites, 1):
                s["rank"] = i
            res[stem] = dict(status="ok", note="%d residues, %d clusters" % (n_res_total, len(sites)),
                             pockets=sites, timing=dict(esm_s=t_esm, model_s=t_model, cluster_s=time.time() - t3,
                                                        load_s=load_s, n_res=n_res_total))
        except Exception as e:  # noqa
            res[stem] = dict(status="error", note="%s: %s" % (type(e).__name__, str(e)[:200]), pockets=[],
                             tb=traceback.format_exc(), timing={})
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
