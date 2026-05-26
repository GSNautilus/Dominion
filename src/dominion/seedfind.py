"""Signal-based seed-finding algorithms.

Two algorithms for identifying object centers from a single signal
channel, used by the ``--mode signal`` pipeline:

* :func:`find_seeds_local_max` — peaks of the (smoothed) signal intensity.
  Each peak is the brightest point of a candidate object. Fast and
  intuitive; the natural choice when objects show clear intensity peaks.
* :func:`find_seeds_dist_transform_peaks` — threshold the signal, compute
  the distance transform of the resulting mask, find peaks. Each peak is
  the geometric "deepest interior" of a signal-bright region. More
  robust to intensity heterogeneity within an object, but requires a
  threshold step.

Both return ``(peaks, peak_values)`` where ``peaks`` is an (N, 2) float
array of ``(row, col)`` pixel coords and ``peak_values`` is an (N,)
array of the per-peak score: smoothed signal intensity for local-max,
distance-from-background for dist-transform.

Pure functions — no UI, no AppState. Imported by both the signal-seeds
submenu and any future tests.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian


def _smooth(signal: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth the signal, returning float32. No-op for sigma <= 0."""
    s = signal.astype(np.float32, copy=False)
    if sigma_px <= 0:
        return s
    return gaussian(s, sigma=float(sigma_px), preserve_range=True).astype(
        np.float32, copy=False
    )


def find_seeds_local_max(
    signal: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    sigma_px: float,
    threshold_abs: float,
    min_distance_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the signal, then find local maxima above ``threshold_abs`` with
    minimum spacing ``min_distance_px``, restricted to the tissue mask.

    Returns ``(peaks, peak_values)`` where ``peak_values`` is the smoothed
    signal intensity at each peak.
    """
    smoothed = _smooth(signal, sigma_px)
    # Zero out pixels outside tissue so they can't pass threshold_abs.
    image = np.where(tissue_mask, smoothed, np.float32(0.0))
    coords = peak_local_max(
        image,
        min_distance=max(1, int(min_distance_px)),
        threshold_abs=float(threshold_abs),
    )
    if coords.size == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=np.float32)
    values = image[coords[:, 0], coords[:, 1]]
    return coords.astype(float, copy=False), values.astype(np.float32, copy=False)


def find_seeds_dist_transform_peaks(
    signal: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    sigma_px: float,
    threshold_abs: float,
    min_distance_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold the (smoothed) signal at ``threshold_abs``, distance-transform
    the resulting mask, then find peaks of the distance map.

    Peaks are the most-interior points of each connected signal-bright
    region. ``peak_values`` is the distance (in pixels) from the peak to
    the nearest background pixel — a rough "radius" of the region around
    each object.
    """
    smoothed = _smooth(signal, sigma_px)
    fg = (smoothed >= float(threshold_abs)) & tissue_mask
    if not fg.any():
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=np.float32)
    dist = ndi.distance_transform_edt(fg).astype(np.float32, copy=False)
    coords = peak_local_max(
        dist,
        min_distance=max(1, int(min_distance_px)),
    )
    if coords.size == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=np.float32)
    values = dist[coords[:, 0], coords[:, 1]]
    return coords.astype(float, copy=False), values.astype(np.float32, copy=False)
