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


_PRUNE_MAX_ITER = 8  # safety bound on the twig-pruning loop
_CYCLE_MAX_ITER = 32  # safety bound on the cycle-breaking loop


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


def _skeleton_degree(skel: np.ndarray) -> np.ndarray:
    """Per-pixel 8-connected neighbor count restricted to skeleton pixels."""
    k = np.ones((3, 3), dtype=np.uint8)
    nbr = ndi.convolve(skel.astype(np.uint8), k, mode="constant", cval=0) - skel.astype(np.uint8)
    return nbr * skel.astype(np.uint8)


def _break_cycles(
    skel: np.ndarray,
    pixel_size_um: float,
    signal_crop: np.ndarray | None = None,
) -> np.ndarray:
    """Break every topological cycle in the skeleton until the path graph
    is a tree.

    Works on the **branch graph** (nodes = skan's junctions/endpoints,
    edges = paths), not just on skan's ``branch_type == 3`` classification
    (that flag covers only isolated closed loops, missing the much more
    common case of two paths between the same pair of junctions — a side
    branch that rejoins the trunk). Algorithm:

    1. Build a list of paths with ``(src_node, dst_node, min_signal)``.
    2. Sort by signal descending (brightest first).
    3. Union-find: walk paths in order; if a path's endpoints are already
       connected via earlier (brighter) paths, that path is redundant
       (it creates a cycle). Mark it for removal.
    4. For each redundant path, delete **all interior pixels** (everything
       except the two branchpoint nodes, which are shared with kept
       paths).
    5. Re-decompose with skan and repeat in case removing interior
       pixels exposes new structure. Converges in 1–2 passes for typical
       inputs.

    A self-loop where ``src == dst`` (skan's type-3) is also caught and
    treated as redundant.
    """
    from skan import Skeleton, summarize

    pruned = skel.copy()
    for _ in range(_CYCLE_MAX_ITER):
        if int(pruned.sum()) < 2:
            break
        try:
            obj = Skeleton(pruned.astype(np.uint8), spacing=float(pixel_size_um))
            summary = summarize(obj, separator="_")
        except ValueError:
            break
        if (
            "node_id_src" not in summary.columns
            or "node_id_dst" not in summary.columns
        ):
            break

        n_paths = int(obj.n_paths)
        if n_paths == 0:
            break

        paths_info: list[tuple[int, int, int, float]] = []
        for i in range(n_paths):
            src = int(summary["node_id_src"].iloc[i])
            dst = int(summary["node_id_dst"].iloc[i])
            path = np.asarray(obj.path_coordinates(i)).astype(np.int64)
            if path.shape[0] == 0:
                continue
            if signal_crop is not None:
                # Use min along the path: a path is only as bright as its
                # weakest link.
                brightness = float(signal_crop[path[:, 0], path[:, 1]].min())
            else:
                brightness = float(path.shape[0])  # prefer longer paths
            paths_info.append((i, src, dst, brightness))

        # Sort brightest first so they win the union-find race.
        paths_info.sort(key=lambda x: -x[3])

        parent: dict[int, int] = {}

        def find(x: int) -> int:
            root = x
            while parent.get(root, root) != root:
                root = parent[root]
            # Path compression for amortized near-O(1).
            while parent.get(x, x) != root:
                nxt = parent[x]
                parent[x] = root
                x = nxt
            return root

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        redundant_paths: list[int] = []
        for i, src, dst, _b in paths_info:
            if src == dst:
                redundant_paths.append(i)
                continue
            if not union(src, dst):
                redundant_paths.append(i)

        if not redundant_paths:
            break

        any_modified = False
        for path_idx in redundant_paths:
            path = np.asarray(obj.path_coordinates(path_idx)).astype(np.int64)
            n_px = path.shape[0]
            if n_px == 0:
                continue
            if n_px >= 3:
                # Remove all interior pixels — keeps the two branchpoints
                # (shared with other paths).
                for r, c in path[1:-1]:
                    if pruned[int(r), int(c)]:
                        pruned[int(r), int(c)] = False
                        any_modified = True
            else:
                # 1- or 2-pixel redundant path: both pixels coincide with
                # branchpoint nodes shared by other paths, so removing
                # either may temporarily disconnect a small piece. We
                # still need to break the cycle — drop the dimmer pixel.
                # The _largest_component pass at the end re-knits things.
                if n_px == 1:
                    candidates = [0]
                else:
                    candidates = [0, 1]
                if signal_crop is not None:
                    vals = [
                        float(signal_crop[int(path[k, 0]), int(path[k, 1])])
                        for k in candidates
                    ]
                    k_drop = candidates[int(np.argmin(vals))]
                else:
                    k_drop = candidates[0]
                py, px = int(path[k_drop, 0]), int(path[k_drop, 1])
                if pruned[py, px]:
                    pruned[py, px] = False
                    any_modified = True

        if not any_modified:
            break

    return pruned


def _prune_twigs(
    skel: np.ndarray,
    signal_crop: np.ndarray | None,
    pixel_size_um: float,
    min_branch_length_um: float,
    min_branch_signal: float,
) -> np.ndarray:
    """Iteratively prune endpoint-terminated branches that fail BOTH the
    length test AND the signal test.

    A twig (type-1 branch: endpoint → branchpoint) is pruned only when
    every "armed" threshold fails:

      armed_length:  min_branch_length_um > 0
                     fails when length < min_branch_length_um
      armed_signal:  min_branch_signal > 0 AND signal_crop available
                     fails when mean(signal along branch) < min_branch_signal

    A short branch is kept if it's bright; a dim branch is kept if it's
    long. Setting either threshold to 0 disables that criterion entirely.
    Both at 0 → no-op.
    """
    if min_branch_length_um <= 0 and min_branch_signal <= 0:
        return skel
    from skan import Skeleton, summarize

    use_signal = signal_crop is not None and min_branch_signal > 0

    pruned = skel.copy()
    for _ in range(_PRUNE_MAX_ITER):
        if int(pruned.sum()) < 2:
            break
        try:
            obj = Skeleton(pruned.astype(np.uint8), spacing=float(pixel_size_um))
            summary = summarize(obj, separator="_")
        except ValueError:
            break
        if "branch_type" not in summary.columns:
            break

        deg = _skeleton_degree(pruned)
        n_paths = int(obj.n_paths)
        lengths = obj.path_lengths()
        any_pruned = False
        for i in range(n_paths):
            btype = int(summary["branch_type"].iloc[i])
            if btype != 1:
                continue  # only prune endpoint → branchpoint twigs
            length = float(lengths[i])
            path = np.asarray(obj.path_coordinates(i)).astype(np.int64)

            armed_failures: list[bool] = []
            if min_branch_length_um > 0:
                armed_failures.append(length < float(min_branch_length_um))
            if use_signal:
                mean_sig = float(signal_crop[path[:, 0], path[:, 1]].mean())
                armed_failures.append(mean_sig < float(min_branch_signal))
            if not armed_failures or not all(armed_failures):
                continue

            sy, sx = int(path[0, 0]), int(path[0, 1])
            ey, ex = int(path[-1, 0]), int(path[-1, 1])
            sd = int(deg[sy, sx])
            ed = int(deg[ey, ex])
            if sd == 1:
                for r, c in path[:-1]:
                    pruned[int(r), int(c)] = False
                any_pruned = True
            elif ed == 1:
                for r, c in path[1:]:
                    pruned[int(r), int(c)] = False
                any_pruned = True
        if not any_pruned:
            break
    return pruned


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
    signal_crop: np.ndarray | None = None,
    min_branch_length_um: float = 0.0,
    min_branch_signal: float = 0.0,
    force_tree: bool = True,
) -> dict | None:
    """Skeletonize one domain (already cropped to its bbox); return per-cell
    info dict or None if the domain has no skeleton.

    Coordinates in the returned dict are in the CROP frame — the caller
    is responsible for shifting them back to the full image by adding
    the crop's (y0, x0) offset.

    The output skeleton is **guaranteed connected** (single 8-connected
    component) — a ``_largest_component`` pass runs after all pruning
    steps to drop any fragments that loop-breaking or twig-pruning may
    have orphaned.
    """
    # Drop disconnected fragments in the input domain mask first.
    mask = _largest_component(domain_mask_crop.astype(bool, copy=False))
    if not mask.any():
        return None

    skel = _skeletonize(mask).astype(bool, copy=False)
    if skel.sum() < 2:
        # Single-pixel or empty skeleton — skan can't build a graph from this.
        return None

    if force_tree:
        skel = _break_cycles(
            skel, pixel_size_um=pixel_size_um, signal_crop=signal_crop
        )
        if skel.sum() < 2:
            return None

    if min_branch_length_um > 0 or min_branch_signal > 0:
        skel = _prune_twigs(
            skel,
            signal_crop=signal_crop,
            pixel_size_um=pixel_size_um,
            min_branch_length_um=min_branch_length_um,
            min_branch_signal=min_branch_signal,
        )
        if skel.sum() < 2:
            return None

    # Final connectivity guarantee. If any of the pruning / cycle-breaking
    # steps disconnected the skeleton, keep the largest piece only.
    skel = _largest_component(skel)
    if skel.sum() < 2:
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
    signal: np.ndarray | None = None,
    min_branch_length_um: float = 0.0,
    min_branch_signal: float = 0.0,
    force_tree: bool = True,
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

        # Always crop the signal if we have it — force_tree and
        # min_branch_signal both consume it.
        signal_crop = signal[sl_y, sl_x] if signal is not None else None
        result = _skeletonize_one_domain(
            mask_full[sl_y, sl_x],
            seed_in_crop,
            pixel_size_um,
            signal_crop=signal_crop,
            min_branch_length_um=min_branch_length_um,
            min_branch_signal=min_branch_signal,
            force_tree=force_tree,
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
