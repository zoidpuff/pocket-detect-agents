"""Project-wide fixed random seed list. Every stochastic step (bootstrap, controls, splits) takes
its seed(s) from here so that results are reproducible. Do not change existing entries."""

SEEDS: tuple[int, ...] = (20261026, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BOOTSTRAP_SEED: int = SEEDS[0]
