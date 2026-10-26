"""Audit step 2: build FASTAs (unique sequences) and run MMseqs2 -s 7.5 train/dev vs test union, train vs dev."""
import subprocess, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path('/cluster/project/beltrao/gmagnus/pocket_detect'); A = ROOT/'results/audit'; MM = ROOT/'tools/bin/mmseqs'
sets = ['ligysis','coach420','holo4k','cryptobench','pocketminer','pocketwatch','af','dimer']
def rd(p):
    for l in open(p):
        i, s = l.rstrip('\n').split('\t'); yield i, s
def uniq_fasta(items, out, prefix):
    """items: (id, seq); writes unique seqs as >prefix{n}; returns {n: [ids]}"""
    d = defaultdict(list)
    for i, s in items:
        s = s.replace('*', 'X')
        if len(s) >= 10: d[s].append(i)
    m = {}
    with open(out, 'w') as f:
        for n, (s, ids) in enumerate(d.items()):
            f.write(f'>{prefix}{n}\n{s}\n'); m[f'{prefix}{n}'] = ids
    return m
import json
test_items = []
for s in sets:
    test_items += [(i, q) for i, q in rd(A/f'test_{s}.tsv') if q]
tm = uniq_fasta(test_items, A/'test_union.fasta', 'T')
tr = list(rd(A/'train_dev_inputs.tsv'))
qm_all = uniq_fasta([x for x in tr if x[0].startswith(('train|','dev|'))], A/'trdev_all.fasta', 'Q')
qm_train = uniq_fasta([x for x in tr if x[0].startswith('train|')], A/'train.fasta', 'R')
qm_dev = uniq_fasta([x for x in tr if x[0].startswith('dev|')], A/'dev.fasta', 'D')
json.dump(dict(test=tm, trdev=qm_all, train=qm_train, dev=qm_dev), open(A/'idmaps.json', 'w'))
print(len(tm), len(qm_all), len(qm_train), len(qm_dev))
fmt = 'query,target,fident,alnlen,qcov,tcov,evalue,qlen,tlen,bits'
def run(q, t, o, tmp):
    subprocess.run([str(MM), 'easy-search', str(q), str(t), str(o), str(tmp), '-s', '7.5', '--max-seqs', '5000', '-e', '1e-3', '--threads', '16',
                    '--format-output', fmt], check=True)
run(A/'trdev_all.fasta', A/'test_union.fasta', A/'trdev_vs_test.m8', A/'tmp1')
run(A/'dev.fasta', A/'train.fasta', A/'dev_vs_train.m8', A/'tmp2')
print('done')
