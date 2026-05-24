"""Shared dataclasses for the Dominion pipeline.

Downstream agents code against these exact signatures — do not add or
rename fields without a coordinated change across all submenus.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ImageData:
    gfap: np.ndarray              # 2D uint16
    dapi: np.ndarray              # 2D uint16
    tissue_mask: np.ndarray       # 2D bool
    pixel_size_um: float
    source_path: Path


@dataclass
class NucleiResult:
    label_mask: np.ndarray        # 2D int32, 0=bg
    centroids: np.ndarray         # (N, 2) float, (row, col) in pixel coords
    params: dict


@dataclass
class AstrocyteSeedsResult:
    kept_indices: np.ndarray      # indices into NucleiResult.centroids
    scores: np.ndarray            # (N,) score for ALL nuclei (for the histogram)
    params: dict


@dataclass
class TessellationResult:
    territory_labels: np.ndarray  # 2D int32; label k corresponds to kept_indices[k-1]
    params: dict
