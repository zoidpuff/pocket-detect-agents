"""Design-independent geometric primitives for pocket detection.

Every primitive has a NumPy/Numba CPU implementation and a batched PyTorch
implementation (suffix ``_torch``) with matching outputs (see
``tests/test_geometry_*.py``).  Inputs are ``coords (N,3) float32`` (Å),
``radii (N,)`` vdW radii, optional ``weights (N,)`` opacity in [0,1] and
``props (N,F)`` per-atom properties.  GPU functions additionally take a
``batch`` vector (structure id per atom / point) so that many structures are
processed in a single call; they return dicts of tensors.

Primitives
----------
sas_points / grid_candidates          query points (Shrake–Rupley SAS, solvent-shell grid)
psp_counts / cascade_mask             LIGSITE-style prefilter for grid candidates
occupancy_grid / ray_buriedness       multi-scale ray-cast buriedness, soft occlusion
neighbor_features / protrusion        radius & Gaussian sums of per-atom properties
smooth_point_values                   neighbour-mean denoising of point features
single_linkage / mean_shift / quick_shift   clustering (canonical labels)
canonical_frame / brick_boxes         rotation invariance (PCA frame), memory bricking
compute_point_features(_batch_torch)  full pipeline (CPU / batched GPU)
"""
from .atomtypes import ATOM_CLASSES, atom_class_matrix, vdw_radii
from .buriedness import (Buriedness, OccupancyGrid, TorchOccupancyGrid, absorption_from_weights, occupancy_grid,
                         ray_buriedness, ray_buriedness_torch)
from .candidates import (PSP_DIRECTIONS, GridPoints, cascade_mask, cascade_mask_torch, grid_candidates,
                         grid_candidates_torch, psp_counts, psp_counts_torch, sublattice_mask)
from .cluster import (canonical_labels, mean_shift, mean_shift_torch, quick_shift, quick_shift_torch,
                      single_linkage, single_linkage_torch)
from .features import (FeatureConfig, PointFeatures, compute_point_features,
                       compute_point_features_batch_torch, concat_structures)
from .frames import brick_boxes, canonical_frame, from_frame, in_box, to_frame
from .neighbors import (NeighborFeatures, neighbor_features, neighbor_features_torch, protrusion, protrusion_torch,
                        smooth_point_values, smooth_point_values_torch)
from .sas import SASPoints, sas_points, sas_points_torch
from .sphere import fibonacci_sphere

__all__ = [
    "ATOM_CLASSES", "atom_class_matrix", "vdw_radii", "fibonacci_sphere",
    "SASPoints", "sas_points", "sas_points_torch",
    "GridPoints", "grid_candidates", "grid_candidates_torch", "PSP_DIRECTIONS", "psp_counts", "psp_counts_torch",
    "sublattice_mask", "cascade_mask", "cascade_mask_torch",
    "OccupancyGrid", "TorchOccupancyGrid", "Buriedness", "absorption_from_weights", "occupancy_grid",
    "ray_buriedness", "ray_buriedness_torch",
    "NeighborFeatures", "neighbor_features", "neighbor_features_torch", "protrusion", "protrusion_torch",
    "smooth_point_values", "smooth_point_values_torch",
    "canonical_labels", "single_linkage", "single_linkage_torch", "mean_shift", "mean_shift_torch",
    "quick_shift", "quick_shift_torch",
    "canonical_frame", "to_frame", "from_frame", "brick_boxes", "in_box",
    "FeatureConfig", "PointFeatures", "compute_point_features", "compute_point_features_batch_torch",
    "concat_structures",
]
