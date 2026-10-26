# FREEZE v2 — BURROW = FREEZE v1 models + output score floor (H5, decision D12)

Written 2026-10-26 23:15. Each model = the v1 model of the same name (results/dev/freeze/models/, unchanged point models and rankers) with `formation.min_score = 0.03`, `formation.min_keep = 3` (pockets with ranker score < 0.03 are dropped once 3 pockets are kept). tau selected on TRAIN cluster-CV only (results/dev/biolip/tau_cv_p3_{small,all}_default_s0.csv; joint criterion over both rankers), one dev confirmation (results/dev/biolip/V2_CYCLE.md). Same tau for all/small and default/accurate. git HEAD at writing: 90386b5f7dfee7dd323ca63c2f6e59ee2091cfaa (working tree with uncommitted v2-cycle files).

| model | sha256 | v1 source sha256 |
|---|---|---|
| results/dev/freeze_v2/models/burrow_all_default_s0.json | 0fa4da97c64cc40fb739b87c38372813aa721d52f37a538d3c57dbc8bfc30f56 | e7d1699e192ede6d7f28ebd5fa055cde2bf4ee95011dff8256b4f68f09e4373e |
| results/dev/freeze_v2/models/burrow_all_default_s1.json | 317570d46e157c1147634ab35b40ce33f5640a44f1433d5309516e103e9894a9 | 713d38b55a13113d05599db402239eec0b4f7c9f8cc855480037f762bee168ed |
| results/dev/freeze_v2/models/burrow_all_default_s2.json | 3e7f1ac500f7bec9322f61510adf7465bc8a79558da3012785cc2057f48acb1e | 7cc48d9c2b384b26840b037117280804ce99035824a85da9aaf558cce7094d6b |
| results/dev/freeze_v2/models/burrow_small_default_s0.json | c0b88419a328332ed8c5c0a2dab3ee2da96aea882001d1ff6aaf20dab8ab34c4 | e3a8dcbda725c0b2d3eebb113a953839267ce4e2f1c8054fd9fe8034da16ff3d |
| results/dev/freeze_v2/models/burrow_small_default_s1.json | da798f489b9d843286b08375e50a4a25760b7b477a6e96d8370883554327fa53 | bb14970360a68aef81957a43c35aff0214cbe716c382910ed31d1b6d1b0c8d60 |
| results/dev/freeze_v2/models/burrow_small_default_s2.json | b770e7227e8f7ea5d12de7eb42dd5cc401c9e0b4bf05de44832a7a68f9e3fd4a | 260df7959d77bb35a0e80afeeb40f7c185a404c6d70c7856c101388c80af246f |
| results/dev/freeze_v2/models/burrow_all_accurate_s0.json | 257df1d8b3595f79d502102ce43755831ee5015fc3d4fe6b49b299aecd44012a | d7f7bc97d4daf393c20896e02cdb2eb9df7e4f34b35dae44717e5aa41fcbbf50 |
| results/dev/freeze_v2/models/burrow_all_accurate_s1.json | 145e7ad7d451a2c0ac1541418f1e03351905159821aa309ff0a65c4313861ad5 | 9471cbca78f31236f98f78c636a4c2413f2c8cfe25fce131254ab1825f5ad0c3 |
| results/dev/freeze_v2/models/burrow_all_accurate_s2.json | 96424bdbe55aaae5749c90cc91692471c6760d1f235842efd4546d5a659a2a59 | 3d868c253f675b79fb372ec3ba61a8c62210464eb89b0d417c1c644df7d78687 |
| results/dev/freeze_v2/models/burrow_small_accurate_s0.json | 0533eb6178728258dc78c28f923800f272c3665668bea4b745203d03e54f0d98 | e44025ea1ad684fccdab41683c14d1c30a3354e503afd0078fe14c963643dd57 |
| results/dev/freeze_v2/models/burrow_small_accurate_s1.json | 9052eb6c0627dd122c0ed217144903a19b8f9a183a7a37565a2085d045fc3414 | b67469c31b9a895e72f6524fe2926ed58beb29b72316386bd09685e844c169bb |
| results/dev/freeze_v2/models/burrow_small_accurate_s2.json | 9e2e51489ea2468e8109bc1504c595505deb53989023ba62847d84ed18fcec74 | bd7e044d4e3635d9636c8b1aadf0f2d1bbbd7f795bc279d607901e52977e3dc3 |
