"""Diagnostic: search primary-violating queries vs builder's canonical test_union (data only), to separate real misses from representation differences."""
import pandas as pd, json, subprocess
from pathlib import Path
R=Path('/cluster/project/beltrao/gmagnus/pocket_detect'); A=R/'results/audit'
P=pd.read_csv(A/'violations_primary.tsv',sep='\t')
q={}
for l in open(A/'trdev_all.fasta'):
    if l[0]=='>': k=l[1:].strip()
    else: q[k]=l.strip()
qs=sorted(set(P.query_seq))
with open(A/'diag_q.fasta','w') as f:
    for k in qs: f.write(f'>{k}\n{q[k]}\n')
U=pd.read_csv(R/'data/manifests/test_union.tsv',sep='\t',dtype=str)
with open(A/'diag_t.fasta','w') as f:
    for s in sorted(set(U.sequence)): f.write(f'>{abs(hash(s))}\n{s}\n')
subprocess.run([str(R/'tools/bin/mmseqs'),'easy-search',str(A/'diag_q.fasta'),str(A/'diag_t.fasta'),str(A/'diag.m8'),str(A/'tmp3'),'-s','7.5','--max-seqs','5000','-e','1e-3','--threads','16','--format-output','query,target,fident,alnlen,qcov,tcov,evalue'],check=True)
h=pd.read_csv(A/'diag.m8',sep='\t',names='query,target,fident,alnlen,qcov,tcov,evalue'.split(','))
v=h[(h.evalue<1e-3)&(h.fident>=.3)&((h.qcov>=.5)|(h.tcov>=.5)|(h.alnlen>=80))]
hit=set(v['query']); P['still_hits_builder_union']=P.query_seq.isin(hit)
g=P.groupby('input_id').still_hits_builder_union.max()
print('primary violators',len(g),'still violating vs builder union:',int(g.sum()))
print(sorted(g[g].index))
P.to_csv(A/'violations_primary.tsv',sep='\t',index=False)
