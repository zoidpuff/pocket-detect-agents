# DATA — datasets, sources, versions, checksums, licenses

All raw data lives in `data/raw/<dataset>/` (git-ignored). Each entry: source URL, version/date retrieved, checksum (sha256 of downloaded archive or manifest), license, and notes on restrictions. Never commit restricted data.

| Dataset | Source | Version / retrieved | Checksum | License | Local path | Notes |
|---|---|---|---|---|---|---|
| PDB mirror | /cluster/work/beltrao/shared/rsyncPDB (wwPDB rsync) | snapshot as mounted | n/a (read-only mirror) | CC0 (wwPDB) | read-only | mmCIF divided, assemblies |
| AFDB v6 | /cluster/work/beltrao/shared/alphafold/v6 | v6 proteome + SwissProt tars | n/a (read-only mirror) | CC-BY 4.0 | read-only | 157 GB tars |
