# Baseline reproduction gate on LIGYSIS (our runs vs published)

Our runs (this repo's wrappers, authors' TRANS-frame ligand-free chains, 2,775 chains) scored with the LBS-comparison protocol (site_eval, 6,721 scored sites, DCC<=12 closest pocket) vs published numbers (ours / published (diff, points)).

| variant | published row | top-N | top-N+2 | max recall | P1K | gate (±2.0 pts) |
|---|---|---|---|---|---|---|
| p2rank_2.4.2 | P2Rank | 47.2 / 46.7 (+0.5) | 52.6 / 51.9 (+0.7) | 57.7 / 57.0 (+0.7) | 80.5 / 79.2 | PASS |
| fpocket_4.2.3 | fpocket | 34.3 / 34.8 (-0.5) | 45.8 / 46.5 (-0.7) | 89.4 / 91.3 (-1.9) | 44.8 / 47.3 | PASS |
| fpocket_4.2.3+PRANK_2.4.2 | fpocket+PRANK | 48.5 / 48.8 (-0.3) | 59.5 / 60.4 (-0.9) | 89.4 / 91.3 (-1.9) | 81.7 / 81.7 | PASS |
| fpocket_4.0.3 | fpocket | 36.2 / 34.8 (+1.4) | 48.1 / 46.5 (+1.6) | 86.2 / 91.3 (-5.1) | 47.2 / 47.3 | n/a (not the paper version) |
| fpocket_4.0.3+PRANK_2.4.2 | fpocket+PRANK | 49.3 / 48.8 (+0.5) | 60.5 / 60.4 (+0.1) | 86.2 / 91.3 (-5.1) | 82.2 / 81.7 | n/a (not the paper version) |
| p2rank_2.5.1 | P2Rank | 47.2 / 46.7 (+0.5) | 52.6 / 51.9 (+0.7) | 57.7 / 57.0 (+0.7) | 80.5 / 79.2 | n/a (not the paper version) |
| fpocket_4.2.3+PRANK_2.5.1 | fpocket+PRANK | 48.5 / 48.8 (-0.3) | 59.5 / 60.4 (-0.9) | 89.4 / 91.3 (-1.9) | 81.7 / 81.7 | n/a (not the paper version) |
| fpocket_4.0.3+PRANK_2.5.1 | fpocket+PRANK | 49.3 / 48.8 (+0.5) | 60.5 / 60.4 (+0.1) | 86.2 / 91.3 (-5.1) | 82.2 / 81.7 | n/a (not the paper version) |
