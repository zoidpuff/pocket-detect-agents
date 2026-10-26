/goal You are the PI of a computational structural biology project and work autonomously. You lead a team of agents (Opus 5.5 or Sonnet 5.5, as you see fit). You delegate, review, integrate and decide; you keep the research record. The goal is the best possible outcome, not the fastest completion. Make decisions yourself, document why, and keep going.


The goal of this project is to build a novel, ultra-fast, state-of-the-art protein binding pocket detection and ranking method. It must clearly beat other existing baseline methods like P2Rank and others. We also want this method to be applicable for homo- and heterodimer structures. Explore which benchmarks or stratifications are suitable for this sub-goal.

Explore various approaches for identifying, scoring and ranking pockets using explicit 3D geometry, physics, and analytical biophysics or other approaches. Experiment with representation and feature creation. 

True Generalization: Your method should generalize across diverse protein families, unseen folds, apo-to-holo transitions, and cryptic or allosteric sites—not just memorize common co-factor shapes. Large deep learning based methods are particularly prone to overfitting. To this end you will need to develop a robust benchmarking approach. Please try to use established benchmarks for pocket detection and ranking. Make sure to perform fair comparisons with existing methods. Some example datasets to look use are LIGYSIS, COACH420 and HOLO4K, scPDB, CryptoBench, HOTPocket, AF2BIND. This is an important part of the project so try to develop a meaningful comparison.

Dedicated AlphaFold Mode (`--af-mode`): A major application of this tool is proteome-wide pocket discovery on AlphaFold2/AlphaFold3 structures. Standard pocket finders can fail on AF structures due to:
   - Low-confidence disordered regions (IDRs / "spaghetti loops" with low pLDDT) packing against the core and creating hallucinated pockets or occluding real ones.
   - Apo-like side-chain conformations partially blocking binding clefts.
   - Uncertain inter-domain orientations (high PAE) generating fake inter-domain cavities.
   You must design and validate a dedicated mode that leverages B-factor column pLDDT values (and optionally PAE matrices or structural packing heuristics) to handle confidence weighting, loop pruning/softening, and flexible side-chain tolerance.

Dual CPU & GPU Optimization: The method must include a clean, fast CPU implementation (e.g., vectorized NumPy/SciPy, Numba, or C++/Cython/Rust extensions) AND a GPU-accelerated backend (e.g., PyTorch tensor ops, CuPy, or Triton/CUDA kernels without neural networks) capable of high-throughput screening across thousands of structures.

Check the local directory first to inspect available datasets and hardware resources (GPU/CPU specs).


We will come back and revisit progress in 24 hours so please aim to work within that time budget.

Briefing: you are running inside a Singularity container on the ETH Euler cluster
The container is the sandbox. Permission prompts are switched off. The mounted directories are the real boundary: nothing else exists inside, and read-only mounts reject writes at the kernel level. If a path you expect is missing, it was not mounted — say so.
No root. apt-get cannot install anything. Use uv, as a fallback use conda or pixi
Git: commit, do not push. There is no push credential. Commit in the read-write code mount with the preset identity (a Co-Authored-By trailer is added automatically), leave the work on a branch, and tell the user what to push.
Where to write. Large or generated output into the workspace, which is on project storage. Scratch into /tmp, which vanishes with the job

We have 4 rtx 4090s, 64 CPUs, and 128GB of RAM available for use. We also prefetched the AFDB and the PDB. We don’t have unlimited storage, so try to not exceed 500GB in retrieved/generated data.

/cluster/work/beltrao/shared on euler has:
alphafold/  - copy of the alphafold database
rsyncPDB/  - copy of the PDB

Literature: Design through repeated and independent critique: Search the current literature and available datasets, and write LITERATURE.md. Then draft DESIGN.md. The project goal and its parts are fixed, the suggestive ideas are a human priors, not a plan. Have the design reviewed and revised in at most three rounds by an external agentic PI, and agentic scientists assessing prior work, statistics, data, algorithms and compute, AI-agent failure modes, and other expertise the design needs. Each reviewer proposes concrete changes that stay within the project's scope, or states that none are needed.

Reproducibility: Never change metrics, controls, labels or tests to make results look better. Keep everything end-to-end reproducible: commit results to git, fix the random seeds list, and keep DATA.md listing the source, version, checksum and license of each dataset. Never commit restricted data, never hardcode data, prefer scripts over one-offs, make sources explicit, no silent assumptions. Don't trust a result until it has been checked. Write verification scripts using toy data with known answers, positive and negative controls, and sanity checks. An agent that didn't produce a result reviews it before you rely on it.

Research tree: Save this prompt verbatim to `PROMPT.md`. Keep track of the research in a research tree (RESEARCH_TREE.md), using a status legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding. At each milestone, add a short decision note and a two-paragraph status summary at the top. The tree is your memory: keep your own context lean by reading agents' summaries by default rather than raw logs and if your context is reset, resume from the tree. 

AI lessons: Keep LESSONS.md, a list of general process rules for running this project efficiently. Whenever something wastes or saves significant effort, add a one-line rule with its evidence. At each milestone, prune it to the most useful rules and ask a fresh agent "what should we stop doing?" and “what worked well?”. 

Auto-research: Your job is to build the best possible method. Report runtime alongside accuracy. Keep going until you genuinely run out of promising ideas or close to the time limit. Then write REPORT.md with a breakdown of the approach, the best methods, test-set results, follow-ups, limitations, and remaining ideas.

Good luck!
