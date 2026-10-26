"""Independent audit step 1: extract sequences of train/dev inputs and of all test sets from structure files."""
import sys, glob, os, json, pickle, gzip
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, gemmi
gemmi.set_leak_warning(False) if hasattr(gemmi,'set_leak_warning') else None
ROOT = Path('/cluster/project/beltrao/gmagnus/pocket_detect'); OUT = ROOT/'results/audit'
MIR = Path('/cluster/work/beltrao/shared/rsyncPDB/data/structures/divided/mmCIF')

def mirror(pdb): return MIR/pdb[1:3]/f'{pdb}.cif.gz'

def chains_obs(path, only=None):
    """{chain_name: (observed protein seq)} model 0; polymer peptide chains only"""
    st = gemmi.read_structure(str(path)); out = {}
    for ch in st[0]:
        p = ch.get_polymer()
        if len(p) == 0: continue
        pt = p.check_polymer_type()
        if pt not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD, gemmi.PolymerType.Unknown): continue
        if only is not None and ch.name not in only: continue
        seq = gemmi.one_letter_code([r.name for r in p]) if False else ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper() if gemmi.find_tabulated_residue(r.name) and gemmi.find_tabulated_residue(r.name).one_letter_code.strip() else 'X' for r in p)
        if pt == gemmi.PolymerType.Unknown:
            # require mostly standard amino acids
            if sum(c != 'X' for c in seq) < 0.5*len(seq): continue
        out.setdefault(ch.name, seq)
    return out, st

def entity_seqs(st, chain):
    """SEQRES (full) entity sequence of an author chain"""
    res = []
    for ent in st.entities:
        if ent.polymer_type in (gemmi.PolymerType.PeptideL,) and chain in [s.split('/')[0] for s in ent.subchains] + [c.name for c in st[0] if any(sc == ent.name or sc in ent.subchains for sc in [c.get_polymer().subchain_id() if len(c.get_polymer()) else ''])]:
            res.append(''.join(gemmi.find_tabulated_residue(n).one_letter_code.upper() if gemmi.find_tabulated_residue(n) and gemmi.find_tabulated_residue(n).one_letter_code.strip() else 'X' for n in ent.full_sequence))
    return res

# ---------- train/dev
def train_job(args):
    split, eid, pdb, chain, spath, asm = args
    rows = []
    try:
        c, _ = chains_obs(spath, only={chain})
        if chain not in c: rows.append((f'{split}|{eid}|rep|{chain}|MISSING', '')); 
        for k, s in c.items(): rows.append((f'{split}|{eid}|rep|{k}', s))
    except Exception as e:
        rows.append((f'{split}|{eid}|rep|ERR', '')); print('ERR', eid, e, file=sys.stderr)
    if asm:
        try:
            c, _ = chains_obs(asm)
            for k, s in c.items(): rows.append((f'{split}|{eid}|asm|{k}', s))
        except Exception as e: print('ERRasm', eid, e, file=sys.stderr)
    return rows

def do_train():
    jobs = []
    for split in ('train', 'dev'):
        m = pd.read_csv(ROOT/f'data/processed/{split}/manifest.csv', dtype=str, keep_default_na=False)
        for r in m.itertuples():
            asm = ''
            if r.assembly_status == 'stored':
                with np.load(ROOT/r.npz, allow_pickle=False) as z: asm = str(z['asm_path'])
            jobs.append((split, r.entry_id, r.pdb, r.chain, r.structure_path, asm))
    with Pool(14) as p: res = p.map(train_job, jobs, chunksize=20)
    with open(OUT/'train_dev_inputs.tsv', 'w') as f:
        for rows in res:
            for i, s in rows: f.write(f'{i}\t{s}\n')

# ---------- test sets
def test_ligysis(_=None):
    ids = pickle.load(open(ROOT/'data/raw/ligysis/lbs_comparison_results/LIGYSIS_3448_chains.pkl', 'rb'))
    rows = []
    for x in ids:
        pdb, ch = x.split('_')
        p = mirror(pdb)
        if not p.exists(): rows.append((f'ligysis|{pdb}|{ch}|NOFILE', '')); continue
        c, st = chains_obs(p, only={ch})
        for k, s in c.items(): rows.append((f'ligysis|{pdb}|{ch}|obs', s))
        for ent in st.entities:
            if ent.polymer_type == gemmi.PolymerType.PeptideL and any(sc in ent.subchains for sc in [r.subchain for r in st[0][ch].get_polymer()[:1]] if ch in [c_.name for c_ in st[0]]):
                rows.append((f'ligysis|{pdb}|{ch}|seqres', ''.join((gemmi.find_tabulated_residue(n).one_letter_code.upper() if gemmi.find_tabulated_residue(n) and gemmi.find_tabulated_residue(n).one_letter_code.strip() else 'X') for n in ent.full_sequence)))
    return rows

def pdbfile_job(a):
    tag, p = a
    try:
        c, _ = chains_obs(p)
        return [(f'{tag}|{Path(p).stem}|{k}', s) for k, s in c.items()]
    except Exception as e:
        print('ERR', p, e, file=sys.stderr); return [(f'{tag}|{Path(p).stem}|ERR', '')]

def ds_files(prefix):
    D = ROOT/'data/raw/p2rank_datasets/current'; fs = set()
    for ds in D.glob(f'{prefix}*.ds'):
        for l in ds.read_text().splitlines():
            l = l.strip()
            if l and not l.startswith(('HEADER', 'PARAM')) and l.endswith('.pdb'): fs.add(l.split()[0])
    return sorted(str(D/f) for f in fs if (D/f).exists()), sorted(f for f in fs if not (D/f).exists())

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1]
    if which == 'train': do_train(); return
    rows = []
    with Pool(14) as pool:
        if which == 'pocketwatch': pass
        elif which == 'ligysis':
            ids = pickle.load(open(ROOT/'data/raw/ligysis/lbs_comparison_results/LIGYSIS_3448_chains.pkl', 'rb'))
            for r in pool.map(lig_job, ids, chunksize=20): rows += r
        else:
            if which == 'coach420': fs, miss = ds_files('coach420'); print('missing', len(miss), miss[:3])
            elif which == 'holo4k': fs, miss = ds_files('holo4k'); print('missing', len(miss), miss[:3])
            elif which == 'cryptobench': fs = sorted(glob.glob(str(ROOT/'data/raw/cryptobench/cif-files/cif-files/*.cif')))
            elif which == 'pocketminer': fs = sorted(glob.glob(str(ROOT/'data/raw/pocketminer/data/pm-dataset/*-structures/*.pdb')))
            elif which == 'dimer': fs = sorted(glob.glob(str(ROOT/'data/processed/dimer/inputs/*/*.pdb')))
            elif which == 'af':
                m = pd.read_csv(ROOT/'data/processed/af/af_manifest.csv', dtype=str, keep_default_na=False)
                fs = [str(ROOT/'data/processed/af'/p) for p in m.model_path if p]
            for r in pool.map(pdbfile_job, [(which, f) for f in fs], chunksize=10): rows += r
    if which == 'pocketwatch':
        s = pd.read_parquet(ROOT/'data/raw/pocketwatch/current/data/sequences.parquet'); s = s[s.split == 'test']
        rows = [(f'pocketwatch|{u}|U', q) for u, q in zip(s.normalized_uniprot_id, s.sequence)]
    with open(OUT/f'test_{which}.tsv', 'w') as f:
        for i, s in rows: f.write(f'{i}\t{s}\n')
    print(which, len(rows))

def lig_job(x):
    pdb, ch = x.split('_'); p = mirror(pdb); rows = []
    if not p.exists(): return [(f'ligysis|{pdb}|{ch}|NOFILE', '')]
    c, st = chains_obs(p, only={ch})
    for k, s in c.items(): rows.append((f'ligysis|{pdb}|{ch}|obs', s))
    for ent in st.entities:
        if ent.polymer_type != gemmi.PolymerType.PeptideL: continue
        chs = set()
        for c_ in st[0]:
            pl = c_.get_polymer()
            if len(pl) and pl.subchain_id() in ent.subchains: chs.add(c_.name)
        if ch in chs:
            rows.append((f'ligysis|{pdb}|{ch}|seqres', ''.join((gemmi.find_tabulated_residue(n).one_letter_code.upper() if gemmi.find_tabulated_residue(n) and gemmi.find_tabulated_residue(n).one_letter_code.strip() else 'X') for n in ent.full_sequence)))
    return rows

if __name__ == '__main__': main()
