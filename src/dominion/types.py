"""Shared dataclasses for the Dominion pipeline.

The pipeline is signal-agnostic: ``signal`` is the immunolabel-of-interest
channel (intermediate-filament markers, membrane markers, anything that
delineates the objects you're partitioning into domains); ``nuclei`` is
an optional nuclear-stain channel used by the nuclei-guided mode for
candidate object centers.
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
