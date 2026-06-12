"""Per-domain skeleton extraction.

Given a tessellation's ``domain_labels`` and the seed centroids that
produced them, extract a one-pixel-wide skeleton inside each domain
mask, root it at the seed (or nearest skeleton pixel), and return both
per-domain branch info and a combined skeleton label image.

The min-signal carving in submenu 3 makes this much cleaner than naive
skeletonization on the whole signal channel — each domain mask is a
connected, cell-shaped region with the seed inside it. We only need to
skeletonize-then-pick-largest-component to handle the rare case of a
process tip pinching off from the soma.

Pure functions — no UI, no AppState. Imported by submenu_skeletons and
any future tests.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize as _skeletonize


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest 8-connected component of ``mask``.

    Used to drop disconnected fragments before skeletonization so each
    cell's skeleton is a single graph.
    """
    if not mask.any():
        return mask
    structure = ndi.generate_binary_structure(2, 2)  # 8-connectivity
    labeled, n = ndi.label(mask, structure=structure)
    if n <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    keep = int(counts.argmax())
    return labeled == keep


def _crop_bbox(mask: np.ndarray) -> tuple[slice, slice]:
    """Tight bounding box around a binary mask, as ``(slice_y, slice_x)``."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return slice(0, 0), slice(0, 0)
    return (
        slice(int(ys.min()), int(ys.max()) + 1),
        slice(int(xs.min()), int(xs.max()) + 1),
    )


def _nearest_skeleton_pixel(
    skeleton: np.ndarray, seed_rc: tuple[float, float]
) -> tuple[int, int]:
    """Return the (row, col) of the skeleton pixel closest to ``seed_rc``.

    Falls back to (0, 0) if ``skeleton`` is empty — callers should check
    for an empty skeleton before calling.
    """
    ys, xs = np.where(skeleton)
    if ys.size == 0:
        return 0, 0
    sy, sx = float(seed_rc[0]), float(seed_rc[1])
    d2 = (ys - sy) ** 2 + (xs - sx) ** 2
    i = int(np.argmin(d2))
    return int(ys[i]), int(xs[i])


def _skeletonize_one_domain(
    domain_mask_crop: np.ndarray,
    seed_rc_in_crop: tuple[float, float],
    pixel_size_um: float,
) -> dict | None:
    """Skeletonize one domain (already cropped to its bbox); return per-cell
    info dict or None if the domain has no skeleton.

    Coordinates in the returned dict are in the CROP frame — the caller
    is responsible for shifting them back to the full image by adding
    the crop's (y0, x0) offset.
    """
    # Drop disconnected fragments (a rare side effect of min-signal carving)
    # so the skeleton is a single graph.
    mask = _largest_component(domain_mask_crop.astype(bool, copy=False))
    if not mask.any():
        return None

    skel = _skeletonize(mask).astype(bool, copy=False)
    if skel.sum() < 2:
        # Single-pixel or empty skeleton — skan can't build a graph from this.
        # Treat as a degenerate "soma only" cell with zero branch length.
        return None

    # Build the skan Skeleton; ``spacing`` makes path lengths come out in
    # microns. We import lazily so building the module doesn't pull skan
    # into memory if nobody runs skeletonization.
    from skan import Skeleton, summarize

    try:
        skel_obj = Skeleton(skel.astype(np.uint8), spacing=float(pixel_size_um))
    except ValueError:
        # Other degenerate skeleton shapes that skan rejects.
        return None
    summary = summarize(skel_obj, separator="_")

    n_paths = int(skel_obj.n_paths)
    branch_paths: list[np.ndarray] = []
    branch_lengths: list[float] = []
    for i in range(n_paths):
        coords = np.asarray(skel_obj.path_coordinates(i)).astype(np.int32, copy=False)
        branch_paths.append(coords)
        branch_lengths.append(float(skel_obj.path_lengths()[i]))

    # Branch type codes (skan convention).
    branch_types = np.asarray(
        summary["branch_type"].to_numpy() if "branch_type" in summary else [],
        dtype=np.int32,
    )

    # Endpoint / branchpoint counts via degree on the skeleton pixels.
    # 4-connected neighbor count is enough to tell endpoints (1 neighbor)
    # from branchpoints (3+ neighbors) in a thin skeleton.
    nbr = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                       mode="constant", cval=0) - skel.astype(np.uint8)
    deg = nbr * skel.astype(np.uint8)
    n_endpoints = int((deg == 1).sum())
    n_branchpoints = int((deg >= 3).sum())

    root_rc = _nearest_skeleton_pixel(skel, seed_rc_in_crop)

    return {
        "branch_paths": branch_paths,
        "branch_lengths_um": np.asarray(branch_lengths, dtype=np.float64),
        "branch_types": branch_types,
        "root_rc": root_rc,
        "total_length_um": float(np.sum(branch_lengths)),
        "n_branches": int(n_paths),
        "n_endpoints": n_endpoints,
        "n_branchpoints": n_branchpoints,
        "_skel_mask": skel,  # for skeleton_label_image assembly; not part of public API
    }


def skeletonize_domains(
    domain_labels: np.ndarray,
    seed_positions: dict[int, tuple[float, float]],
    pixel_size_um: float,
) -> tuple[dict[int, dict], np.ndarray]:
    """Skeletonize every domain in ``domain_labels`` and return per-domain
    info dicts plus a combined skeleton label image.

    Parameters
    ----------
    domain_labels
        2D int32 array; label ``k`` is the k-th domain.
    seed_positions
        Mapping from domain ID to ``(row, col)`` pixel coordinate of the
        seed for that domain.
    pixel_size_um
        Pixel size in microns; passed to skan so branch lengths come out
        in real units.

    Returns
    -------
    (per_domain, skeleton_label_image)
        ``per_domain[k]`` is the dict described in
        :func:`_skeletonize_one_domain`, with coordinates shifted back to
        full-image frame and the ``_skel_mask`` key stripped.
        ``skeleton_label_image`` is a 2D int32 array where each skeleton
        pixel carries the domain ID it belongs to (0 = not skeleton).
    """
    per_domain: dict[int, dict] = {}
    skeleton_label_image = np.zeros(domain_labels.shape, dtype=np.int32)

    unique = np.unique(domain_labels)
    unique = unique[unique > 0]  # skip background

    for k in unique:
        k_int = int(k)
        if k_int not in seed_positions:
            continue
        mask_full = domain_labels == k
        sl_y, sl_x = _crop_bbox(mask_full)
        if sl_y.start == sl_y.stop or sl_x.start == sl_x.stop:
            continue

        seed_y, seed_x = seed_positions[k_int]
        seed_in_crop = (seed_y - sl_y.start, seed_x - sl_x.start)

        result = _skeletonize_one_domain(
            mask_full[sl_y, sl_x], seed_in_crop, pixel_size_um
        )
        if result is None:
            continue

        # Stamp the skeleton into the full-image label image.
        skeleton_label_image[sl_y, sl_x][result["_skel_mask"]] = k_int

        # Shift per-branch coords back to full-image frame.
        y0, x0 = sl_y.start, sl_x.start
        shifted_paths = []
        for path in result["branch_paths"]:
            p = path.copy()
            p[:, 0] += y0
            p[:, 1] += x0
            shifted_paths.append(p)
        result["branch_paths"] = shifted_paths
        result["root_rc"] = (
            int(result["root_rc"][0]) + y0,
            int(result["root_rc"][1]) + x0,
        )

        # Public dict — drop the private mask key.
        result.pop("_skel_mask", None)
        per_domain[k_int] = result

    return per_domain, skeleton_label_image
