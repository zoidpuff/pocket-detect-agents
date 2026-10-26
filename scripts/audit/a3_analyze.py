"""Audit step 3: apply leakage rule, PDB/UniProt overlap, train-vs-dev."""
import json, gzip, glob, re
from pathlib import Path
from collections import defaultdict
import pandas as pd
ROOT = Path('/cluster/project/beltrao/gmagnus/pocket_detect'); A = ROOT/'results/audit'
cols = 'query,target,fident,alnlen,qcov,tcov,evalue,qlen,tlen,bits'.split(',')
idm = json.load(open(A/'idmaps.json'))
def rule(df):
    return df[(df.evalue < 1e-3) & (df.fident >= 0.30) & ((df.qcov >= 0.5) | (df.tcov >= 0.5) | (df.alnlen >= 80))]
sets = ['ligysis','coach420','holo4k','cryptobench','pocketminer','pocketwatch','af','dimer']
out = {}
# ---- (2) train/dev vs test
h = pd.read_csv(A/'trdev_vs_test.m8', sep='\t', names=cols)
v = rule(h).copy()
print('hits', len(h), 'rule hits', len(v))
import numpy as np
pm = set()
for n in ('test','val'):
    for x in np.load(ROOT/f'data/raw/pocketminer/data/pm-dataset/{n}_apo_ids_with_chainids.npy', allow_pickle=True): pm.add(str(x)[:4].lower()+'|'+str(x)[4:])
cb = set(); cbtest = set()
_d = json.load(open(ROOT/'data/raw/cryptobench/osf/cryptobench/cryptobench-dataset/dataset.json'))
_f = json.load(open(ROOT/'data/raw/cryptobench/osf/cryptobench/cryptobench-dataset/folds.json'))
for k, lst in _d.items():
    for e in lst:
        for a, b in ((k, e['apo_chain']), (e['holo_pdb_id'], e['holo_chain'])):
            for c in b.split('-') if False else [b]: cb.add(a.lower()+'|'+c)
def primary(i):
    t = i.split('|')
    if t[0] == 'pocketminer':
        m = re.match(r'^(\w{4})(\w)_', t[1]); return bool(m) and (m.group(1).lower()+'|'+m.group(2)) in pm
    if t[0] == 'cryptobench': return (t[1].lower()+'|'+t[2]) in cb
    return True
rows = []
for r in v.itertuples():
    prim = any(primary(i) for i in idm['test'][r.target])
    tset = sorted({i.split('|')[0] for i in idm['test'][r.target]})
    for qi in idm['trdev'][r.query]:
        rows.append(dict(primary=prim, input_id=qi, query_seq=r.query, target=r.target, test_sets=','.join(tset), test_ids=';'.join(idm['test'][r.target][:3]),
                         fident=r.fident, alnlen=r.alnlen, qcov=r.qcov, tcov=r.tcov, evalue=r.evalue))
V = pd.DataFrame(rows)
V.to_csv(A/'violations_vs_test.tsv', sep='\t', index=False)
out['n_rule_violating_inputs'] = int(V.input_id.nunique()) if len(V) else 0
# near-misses: loosen identity to >=0.25 or E<1e-1 -- from the raw -e 1e-3 hits only report fident 0.25-0.3 etc
nm = h[(h.evalue < 1e-3) & ~h.index.isin(v.index) & (h.fident >= 0.25)]
out['near_miss_hits_e1e-3_ident25-30_or_cov_fail'] = int(len(nm))
# per-set counts
per = {}; perp = {}
for s in sets:
    per[s] = int(V[V.test_sets.str.contains(s)].input_id.nunique()) if len(V) else 0
    perp[s] = int(V[V.primary & V.test_sets.str.contains(s)].input_id.nunique()) if len(V) else 0
out['violating_inputs_per_test_set_broad'] = per; out['violating_inputs_per_test_set_primary'] = perp
out['n_violating_inputs_primary'] = int(V[V.primary].input_id.nunique())
out['violating_by_kind_primary'] = V[V.primary].assign(k=lambda d: d.input_id.str.split('|').str[0]+'|'+d.input_id.str.split('|').str[2]).groupby('k').input_id.nunique().to_dict()
out['n_primary_test_ids'] = dict(pocketminer=len(pm), cryptobench=len(cb))
# strongest hits
if len(V): print(V.sort_values('evalue').head(15).to_string())
# ---- (4) train vs dev
h2 = pd.read_csv(A/'dev_vs_train.m8', sep='\t', names=cols); v2 = rule(h2)
rows = []
for r in v2.itertuples():
    for qi in idm['dev'][r.query]:
        rows.append(dict(dev_input=qi, train_target=';'.join(idm['train'][r.target][:3]), fident=r.fident, alnlen=r.alnlen, qcov=r.qcov, tcov=r.tcov, evalue=r.evalue))
V2 = pd.DataFrame(rows); V2.to_csv(A/'violations_dev_vs_train.tsv', sep='\t', index=False)
out['dev_vs_train_rule_hits'] = len(h2) and int(len(v2)); out['dev_inputs_violating_vs_train'] = int(V2.dev_input.nunique()) if len(V2) else 0
out['dev_vs_train_raw_hits'] = int(len(h2))
if len(V2): print(V2.sort_values('evalue').head(10).to_string())
# ---- (3) PDB / UniProt overlap
sifts = pd.read_csv(ROOT/'data/processed/af/external/pdb_chain_uniprot.tsv.gz', sep='\t', comment='#', names=['pdb','chain','up','a','b','c','d','e','f'], skiprows=2, dtype=str)
by_pdb = sifts.groupby('pdb').up.apply(set).to_dict()
by_pc = sifts.groupby(['pdb','chain']).up.apply(set).to_dict()
tp = defaultdict(set); tu = defaultdict(set)
def add(s, pdb, chains=None):
    pdb = pdb.lower(); tp[s].add(pdb)
    tu[s] |= by_pdb.get(pdb, set())      # entry-wide (conservative)
for l in open(A/'test_ligysis.tsv'):
    i = l.split('\t')[0].split('|'); add('ligysis', i[1])
for s in ['coach420','holo4k','pocketminer','dimer','cryptobench']:
    for l in open(A/f'test_{s}.tsv'):
        i = l.split('\t')[0].split('|'); stem = i[1]
        if s == 'pocketminer' or s == 'coach420' or s == 'holo4k' or s == 'cryptobench' or s == 'dimer':
            if re.match(r'^[0-9][A-Za-z0-9]{3}', stem): add(s, stem[:4])
for f in glob.glob(str(ROOT/'data/raw/cryptobench/osf/cryptobench/cryptobench-dataset/dataset.json')):
    d = json.load(open(f))
    for k, lst in d.items():
        add('cryptobench', k)
        for e in lst: add('cryptobench', e['holo_pdb_id']); tu['cryptobench'].add(e['uniprot_id'])
for l in open(A/'test_af.tsv'): pass
af = pd.read_csv(ROOT/'data/processed/af/af_manifest.csv', dtype=str, keep_default_na=False)
tu['af'] |= set(af.acc)
pw = pd.read_parquet(ROOT/'data/raw/pocketwatch/current/data/sequences.parquet'); pw = pw[pw.split == 'test']
tu['pocketwatch'] |= set(pw.normalized_uniprot_id)
_ps = pd.read_parquet(ROOT/'data/raw/pocketwatch/current/data/pocket_label_sites.parquet'); _ps = _ps[_ps.split == 'test']
for ids, acc in zip(_ps.pdb_ids, _ps.primary_accessions):
    for x in ids: tp['pocketwatch'].add(x.lower())
    tu['pocketwatch'] |= set(acc)
dm = pd.read_csv(ROOT/'data/processed/dimer/manifest_chains.tsv', sep='\t', dtype=str, keep_default_na=False)
for u in dm.uniprot:
    tu['dimer'] |= {x for x in re.split('[,;]', u) if x}
# ligysis uniprots also from LIGYSIS data? (SIFTS only; plus AF set acc which is subset of LIGYSIS UniProts)
tu['ligysis'] |= set(af.acc)
# train/dev side
def ptr(split):
    m = pd.read_csv(ROOT/f'data/processed/{split}/manifest.csv', dtype=str, keep_default_na=False)
    strict = {}; wide = {}
    for r in m.itertuples():
        s = {x for x in r.uniprot.split(',') if x} | by_pc.get((r.pdb, r.chain), set())
        strict[r.entry_id] = (r.pdb.lower(), s)
        wide[r.entry_id] = by_pdb.get(r.pdb.lower(), set()) | s
    return m, strict, wide
res = {}
for split in ('train','dev'):
    m, strict, wide = ptr(split)
    for s in sets:
        pdb_hit = sorted(e for e, (p, u) in strict.items() if p in tp[s])
        up_hit = sorted(e for e, (p, u) in strict.items() if u & tu[s])
        up_wide = sorted(e for e, u in wide.items() if u & tu[s])
        res[f'{split}|{s}'] = dict(n_test_pdb=len(tp[s]), n_test_uniprot=len(tu[s]), pdb_overlap=len(pdb_hit), uniprot_overlap_strict=len(up_hit), uniprot_overlap_entrywide=len(up_wide), examples=(pdb_hit + up_hit + up_wide)[:5])
out['overlap'] = res
# also train vs dev PDB / UniProt
mt, st, wt = ptr('train'); md, sd, wd = ptr('dev')
tpdb = {p for p, u in st.values()}; tup = set().union(*[u for p, u in st.values()])
out['train_dev_pdb_overlap'] = sorted(e for e, (p, u) in sd.items() if p in tpdb)[:20]
out['train_dev_uniprot_overlap'] = sorted(e for e, (p, u) in sd.items() if u & tup)[:20]
out['train_dev_pdb_overlap_n'] = sum(p in tpdb for p, u in sd.values()); out['train_dev_uniprot_overlap_n'] = sum(bool(u & tup) for p, u in sd.values())
# also: each train/dev assembly chain's PDB id (assembly partners are same pdb) - already covered.
json.dump(out, open(A/'analysis.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))

# ---- detail of strict UniProt overlaps
det = []
for split in ('train','dev'):
    m, strict, wide = ptr(split)
    mu = dict(zip(m.entry_id, m.uniprot))
    for e, (p, u) in strict.items():
        for s in sets:
            x = u & tu[s]
            if x: det.append(dict(split=split, entry=e, test_set=s, shared_uniprot=','.join(sorted(x)), manifest_uniprot=mu[e], sifts_rep_chain=','.join(sorted(by_pc.get((p, e.split('_')[-1]), set())))))
pd.DataFrame(det).to_csv(A/'uniprot_overlap_strict.tsv', sep='\t', index=False)

# ---- primary-tier UniProt overlap (chain-specific for LIGYSIS / CryptoBench pairs / PocketMiner test+val)
tuP = defaultdict(set)
for l in open(A/'test_ligysis.tsv'):
    i = l.split('\t')[0].split('|'); tuP['ligysis'] |= by_pc.get((i[1], i[2]), set())
tuP['ligysis'] |= set(af.acc)
for k in cb:
    p, c = k.split('|'); tuP['cryptobench'] |= by_pc.get((p, c), set())
for k, lst in _d.items():
    for e in lst: tuP['cryptobench'].add(e['uniprot_id'])
for k in pm:
    p, c = k.split('|'); tuP['pocketminer'] |= by_pc.get((p, c), set())
for s in ('coach420','holo4k','dimer','af','pocketwatch'): tuP[s] = tu[s]
detP = []
for split in ('train','dev'):
    m, strict, wide = ptr(split)
    for e, (p, u) in strict.items():
        for s in sets:
            x = u & tuP[s]
            if x: detP.append(dict(split=split, entry=e, test_set=s, shared_uniprot=','.join(sorted(x))))
DP = pd.DataFrame(detP); DP.to_csv(A/'uniprot_overlap_primary.tsv', sep='\t', index=False)
print('PRIMARY uniprot overlap', DP.groupby(['split','test_set']).entry.nunique().to_dict() if len(DP) else 0)
out['uniprot_primary_overlap'] = {f'{a}|{b}': int(n) for (a, b), n in DP.groupby(['split','test_set']).entry.nunique().items()} if len(DP) else {}
json.dump(out, open(A/'analysis.json', 'w'), indent=1, default=str)
