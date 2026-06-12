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
