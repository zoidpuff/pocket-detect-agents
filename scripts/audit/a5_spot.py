"""Spot-check 20 random train entries: npz rep_seq vs structure-file sequence, ligand proximity to chain, ligand identity."""
import numpy as np, pandas as pd, gemmi, random, json, sys
sys.path.insert(0, '/cluster/project/beltrao/gmagnus/pocket_detect/scripts/audit')
from pathlib import Path
R = Path('/cluster/project/beltrao/gmagnus/pocket_detect')
m = pd.read_csv(R/'data/processed/train/manifest.csv', dtype=str, keep_default_na=False)
random.seed(20261026); idx = random.sample(range(len(m)), 20)
water_like = {'HOH','DOD'}; lines = []
for i in idx:
    r = m.iloc[i]
    z = np.load(R/r.npz, allow_pickle=False)
    st = gemmi.read_structure(r.structure_path); st.setup_entities()
    ch = st[0][r.chain]; poly = ch.get_polymer()
    ca = np.array([a.pos.tolist() for res in poly for a in res if a.name == 'CA'])
    seq = gemmi.one_letter_code([x.name for x in poly])
    rep = str(z['rep_seq']) if z['rep_seq'].ndim == 0 else ''.join(z['rep_seq'])
    ptr = z['lig_ptr']; xyz = z['lig_coords']; ccd = [str(c) for c in z['lig_ccd']]; ion = z['lig_is_ion']
    d = []
    for k in range(len(ccd)):
        c = xyz[ptr[k]:ptr[k+1]]
        d.append(float(np.min(np.linalg.norm(c[:, None] - ca[None], axis=2))) if len(c) else -1)

    lines.append(f"{r.entry_id}\tpdb={r.pdb}{r.chain}\tres={r.resolution}\t{r.method}\tlen_struct={len(seq)}\tn_res={r.n_res}\trep_seq==struct:{rep==seq}\tn_lig={len(ccd)} ({sum(map(bool,ion))} ion)\tccd={','.join(sorted(set(ccd))[:8])}\tmax_min_CA_dist={max(d) if d else None:.1f}\tn_sites={r.n_sites}\ttitle={st.info['_struct.title'][:60] if '_struct.title' in st.info else ''}\tseq={seq[:30]}")
open(R/'results/audit/spotcheck20.tsv', 'w').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
