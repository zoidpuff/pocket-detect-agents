"""Verifier: paired cluster bootstrap (30% clusters, 1000 resamples) of C1 / C1-strict deltas from the
verifier's own per-site tables. No pocketdetect imports."""
import numpy as np, pandas as pd
R = "results/verify_test/ligysis"
cl = pd.read_csv("data/processed/clusters/test_clusters.tsv", sep="\t")
cl = cl[cl.source_set.isin(["ligysis", "ligysis_extra"])]
cl["sid"] = cl.id.str.split("|").str[1]
unit = dict(zip(cl.sid, cl.cluster30))
pub = pd.read_csv(f"{R}/dca_published_sites.csv"); rer = pd.read_csv(f"{R}/dca_rerun_sites.csv")
out = []
for v in ["v1", "v2"]:
    for s in [0, 1, 2]:
        b = pd.read_csv(f"{R}/dca_burrow_{v}_s{s}_sites.csv")
        for cname, c in [("published", pub), ("rerun", rer)]:
            for col, sub in [("hit_ldcc12_topn2", None), ("hit_ldca4_topn2", "has_lig")]:
                x = b.merge(c, on=["structure_id", "site_id"], suffixes=("_b", "_c"))
                if sub: x = x[x[sub + "_b"] == 1]
                x["u"] = x.structure_id.map(unit)
                assert x.u.notna().all()
                g = x.groupby("u").agg(hb=(col + "_b", "sum"), hc=(col + "_c", "sum"), n=(col + "_b", "size"))
                rng = np.random.default_rng(20261026)
                k = len(g); hb, hc, n = g.hb.values, g.hc.values, g.n.values
                d = []
                for _ in range(1000):
                    w = np.bincount(rng.integers(0, k, k), minlength=k)
                    d.append(100 * ((w * hb).sum() - (w * hc).sum()) / (w * n).sum())
                delta = 100 * (hb.sum() - hc.sum()) / n.sum()
                out.append(dict(version=v, seed=s, comparator=cname, endpoint="C1" if sub is None else "C1-strict",
                                n_sites=int(n.sum()), n_units=k, burrow=round(100 * hb.sum() / n.sum(), 2),
                                comp=round(100 * hc.sum() / n.sum(), 2), delta=round(delta, 2),
                                ci_lo=round(np.percentile(d, 2.5), 2), ci_hi=round(np.percentile(d, 97.5), 2)))
df = pd.DataFrame(out); df.to_csv(f"{R}/paired_boot.csv", index=False); print(df.to_string())
