"""Sholl analysis on per-cell skeletons (Feature 3).

For each cell with a skeleton + root, count how many separate skeleton
branches cross a concentric ring around the root, for rings spaced
``ring_spacing_um`` µm apart. This is the classical Sholl ring-intersection
count (connected components within each annulus), NOT a raw pixel count
— a long radial branch still counts as one intersection even though it
occupies many pixels at that radius.

Derived per-cell metrics:

* ``peak_intersections`` — maximum intersection count across all rings
* ``peak_radius_um`` — radius at which the maximum occurs
* ``max_radius_um`` — distance to the farthest skeleton pixel
* ``critical_radius_um`` — first radius beyond the peak where
  intersections drop to ≤ peak/2 (Sholl's classical "critical value")
* ``auc`` — sum of intersections (proxy for total branching length)
* ``ramification_index`` — peak_intersections / max(1, intersections at
  the smallest ring) — how much branching expands away from the soma

Pure functions — no UI, no AppState.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _cell_sholl(
    cell_skeleton_mask: np.ndarray,
    root_rc: tuple[int, int],
    pixel_size_um: float,
    ring_spacing_um: float,
) -> dict:
    """Compute Sholl for a single cell's skeleton mask + root pixel."""
    ys, xs = np.where(cell_skeleton_mask)
    if ys.size == 0:
        return {
            "radii_um": np.zeros(0, dtype=np.float64),
            "intersections": np.zeros(0, dtype=np.int32),
            "peak_intersections": 0,
            "peak_radius_um": 0.0,
            "max_radius_um": 0.0,
            "critical_radius_um": 0.0,
            "auc": 0,
            "ramification_index": 0.0,
        }

    ry, rx = int(root_rc[0]), int(root_rc[1])
    # Euclidean distance from root to each skeleton pixel, in microns.
    dy = ys - ry
    dx = xs - rx
    dist_px = np.sqrt(dy * dy + dx * dx)
    dist_um = dist_px * float(pixel_size_um)

    max_radius_um = float(dist_um.max())
    if max_radius_um <= 0 or ring_spacing_um <= 0:
        return {
            "radii_um": np.zeros(0, dtype=np.float64),
            "intersections": np.zeros(0, dtype=np.int32),
            "peak_intersections": 0,
            "peak_radius_um": 0.0,
            "max_radius_um": max_radius_um,
            "critical_radius_um": 0.0,
            "auc": 0,
            "ramification_index": 0.0,
        }

    n_rings = int(np.ceil(max_radius_um / float(ring_spacing_um)))
    radii_um = (np.arange(1, n_rings + 1) * float(ring_spacing_um)).astype(np.float64)

    # Build full-array distance map for component counting via ndi.label.
    h, w = cell_skeleton_mask.shape
    iy = np.arange(h)[:, None] - ry
    ix = np.arange(w)[None, :] - rx
    dist_map_um = np.sqrt(iy * iy + ix * ix) * float(pixel_size_um)

    structure = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    intersections = np.zeros(n_rings, dtype=np.int32)
    for k in range(n_rings):
        r = radii_um[k]
        lo = r - 0.5 * float(ring_spacing_um)
        hi = r + 0.5 * float(ring_spacing_um)
        annulus = (dist_map_um > lo) & (dist_map_um <= hi)
        in_ring = cell_skeleton_mask & annulus
        if in_ring.any():
            _, n = ndi.label(in_ring, structure=structure)
            intersections[k] = int(n)

    peak_intersections = int(intersections.max())
    peak_idx = int(intersections.argmax())
    peak_radius_um = float(radii_um[peak_idx])
    auc = int(intersections.sum())

    # Critical radius: first ring after the peak where intersections drop
    # to half-peak or less.
    half = peak_intersections / 2.0
    critical_radius_um = max_radius_um
    for k in range(peak_idx + 1, n_rings):
        if intersections[k] <= half:
            critical_radius_um = float(radii_um[k])
            break

    # Ramification index: peak / proximal-ring intersections (a measure of
    # how much the arbor expands away from the soma). Falls back to 1.0
    # when the innermost ring is empty.
    inner = int(intersections[0]) if n_rings > 0 else 0
    ramification_index = (
        float(peak_intersections) / float(max(inner, 1))
        if peak_intersections > 0
        else 0.0
    )

    return {
        "radii_um": radii_um,
        "intersections": intersections,
        "peak_intersections": peak_intersections,
        "peak_radius_um": peak_radius_um,
        "max_radius_um": max_radius_um,
        "critical_radius_um": critical_radius_um,
        "auc": auc,
        "ramification_index": ramification_index,
    }


def sholl_for_skeletons(
    skeleton_label_image: np.ndarray,
    per_domain: dict[int, dict],
    pixel_size_um: float,
    ring_spacing_um: float,
) -> dict[int, dict]:
    """Compute Sholl analysis for every cell in ``per_domain``.

    Returns a ``{domain_id: cell_sholl_dict}`` mapping. Cells whose
    skeleton consists of a single pixel (already filtered upstream) or
    whose root coincides with the only skeleton pixel return empty
    profiles with zero metrics.
    """
    out: dict[int, dict] = {}
    for k, info in per_domain.items():
        cell_mask_full = skeleton_label_image == k
        if not cell_mask_full.any():
            continue
        ys, xs = np.where(cell_mask_full)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cell_crop = cell_mask_full[y0:y1, x0:x1]
        root_in_crop = (int(info["root_rc"][0]) - y0, int(info["root_rc"][1]) - x0)
        out[int(k)] = _cell_sholl(
            cell_crop,
            root_in_crop,
            pixel_size_um,
            ring_spacing_um,
        )
    return out
