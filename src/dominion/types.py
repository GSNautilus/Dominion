"""Shared dataclasses for the DOMINION pipeline.

The pipeline is signal-agnostic: ``signal`` is the immunolabel-of-interest
channel (intermediate-filament markers, membrane markers, anything that
delineates the objects you're partitioning into domains); ``nuclei`` is
an optional nuclear-stain channel used by the nuclei-guided mode for
candidate object centers. ``extra_channels`` carries any further
channels that are not driving the segmentation but get measured per
domain (Feature 1).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ImageData:
    signal: np.ndarray            # 2D, immunolabel-of-interest channel
    nuclei: Optional[np.ndarray]  # 2D, optional nuclei-stain channel
    tissue_mask: np.ndarray       # 2D bool
    pixel_size_um: float
    source_path: Path
    # Channels beyond signal + nuclei. Keyed by display name (e.g.
    # "channel_2"); each value is a 2D array of the same (H, W) shape as
    # ``signal``. Empty dict for 1- and 2-channel TIFFs.
    extra_channels: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class NucleiResult:
    label_mask: np.ndarray        # 2D int32, 0=bg
    centroids: np.ndarray         # (N, 2) float, (row, col) in pixel coords
    params: dict


@dataclass
class SeedsResult:
    kept_indices: np.ndarray      # indices into NucleiResult.centroids
    scores: np.ndarray            # (N,) score for ALL candidates (for the histogram)
    params: dict


@dataclass
class TessellationResult:
    domain_labels: np.ndarray     # 2D int32; label k corresponds to kept_indices[k-1]
    params: dict


@dataclass
class SkeletonsResult:
    """Per-domain skeleton extraction.

    Each entry in ``per_domain`` describes one cell's skeleton:

    * ``branch_paths``: list of ``(M, 2)`` int arrays of (row, col) pixel
      coords, one per branch in the skeleton graph.
    * ``branch_lengths_um``: array of branch lengths in microns,
      same order as ``branch_paths``.
    * ``branch_types``: array of skan branch-type codes
      (0=endpoint→endpoint, 1=endpoint→branch, 2=branch→branch, 3=loop).
    * ``root_rc``: (row, col) of the skeleton pixel chosen as the tree
      root (nearest skeleton pixel to the seed, in pixel coords).
    * ``total_length_um``, ``n_branches``, ``n_endpoints``,
      ``n_branchpoints``: scalar summaries.

    ``skeleton_label_image`` is a 2D int32 image where each skeleton
    pixel carries the domain ID it belongs to (0 = not on any skeleton).
    Convenient for a napari Labels overlay.
    """

    per_domain: dict[int, dict]              # {domain_id: {...}}
    skeleton_label_image: np.ndarray          # 2D int32
    params: dict


@dataclass
class SholResult:
    """Per-cell Sholl analysis derived from a SkeletonsResult.

    For each cell present in ``per_domain``:

    * ``radii_um`` — (R,) float, ring center radii in microns
      (``ring_spacing, 2*ring_spacing, ...``) up to ``max_radius_um``.
    * ``intersections`` — (R,) int32, number of skeleton-branch crossings
      at each radius (connected components of the skeleton within the
      annular ring, NOT raw pixel counts).
    * Scalar derived metrics: ``peak_intersections``, ``peak_radius_um``,
      ``max_radius_um``, ``critical_radius_um``, ``auc``,
      ``ramification_index``.
    """

    per_domain: dict[int, dict]   # {domain_id: {radii_um, intersections, peak_*, ...}}
    params: dict                  # {"ring_spacing_um": float}


@dataclass
class MeasurementsResult:
    """Per-domain measurements across one or more channels.

    ``per_channel`` is keyed by channel name (e.g. ``"signal"``,
    ``"nuclei"``, ``"channel_2"``) and each value is a dict of named
    statistics, each statistic being an (N,) array where N is the number
    of domains. Domain index ``i`` in these arrays corresponds to the
    i-th non-background label in ``TessellationResult.domain_labels`` —
    the canonical ordering is the sorted unique labels, also stored in
    ``domain_ids``.
    """

    domain_ids: np.ndarray                              # (N,) int32, sorted unique labels
    per_channel: dict[str, dict[str, np.ndarray]]       # {channel: {stat: (N,) array}}
    morphology: dict[str, np.ndarray]                   # {stat: (N,) array} (area, centroid_y/x, etc.)
    params: dict
