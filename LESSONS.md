# LESSONS — process rules for running this project efficiently

(One-line rules with evidence. Pruned at each milestone.)

1. Inspect mounts/tools before planning (evidence: no Java/nvcc/conda in container — baselines needing Java must bring their own JDK).
2. PI commits only explicit paths it owns (or an agent's files after that agent reports done); never `git add -A` while agents are editing (evidence: R1 agent-failure review found in-progress ligands.py/geometry edits swept into PI commits).
3. Single source of truth for constants: seeds live in src/pocketdetect/seeds.py; docs reference it, never restate values (evidence: DESIGN v1 said [0,1,2], seeds.py says (20261026,1,…)).
