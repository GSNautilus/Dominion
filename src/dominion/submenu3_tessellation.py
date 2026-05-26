"""Submenu 3 — signal-guided watershed tessellation of object domains.

This widget watches the ``seeds`` slot. Whenever it changes, the widget
runs a seeded watershed on the signal channel masked to the tissue,
using the centroids of the kept seeds as marker positions. The
elevation combines a smoothed signal image (low-elevation = bright =
"inside" an object) and a distance transform from the seed markers,
blended by the ``Signal influence`` slider. The result lands in
``state.tessellation``.

Three constraints are folded into the watershed itself (not applied
as post-filters):

* **Min signal** — pixels whose smoothed signal falls below this
  threshold are dropped from the effective watershed mask. Useful for
  excluding low-signal background regions, tissue gaps, or any pixel
  the user doesn't consider valid tessellation space. Applied to the
  same smoothed signal used for elevation.
* **Max domain area (µm²)** — each domain is trimmed to at most this
  many pixels (converted from µm²), keeping the pixels closest to its
  seed. Pixels beyond the per-domain cap become background. This is a
  true area cap, not a radius approximation.
* **Min domain area (µm²)** — after a watershed pass, any domain
  smaller than the threshold has its seed removed and the watershed is
  re-run on the reduced seed set. Tiny domains thus get absorbed into
  their neighbors rather than left as orphan slivers. The max-area
  trim is re-applied after each re-watershed since neighbors grow when
  a seed is dropped. Loop is bounded to avoid pathological oscillation.

All three default to 0 (off).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.segmentation import watershed

from .state import AppState
from .types import TessellationResult
from .widgets.common import CollapsibleSection, HistogramSlider, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


_LAYER_NAME = "Domains"
_MIN_AREA_ITER_CAP = 5  # safety bound on the min-area re-watershed loop


def _trim_to_max_area(
    labels: np.ndarray, dist: np.ndarray, max_area_px: float
) -> np.ndarray:
    """For each domain over ``max_area_px`` pixels, drop the pixels furthest
    from the seed until the domain fits the cap. Returns ``labels`` modified
    in place.
    """
    if max_area_px <= 0:
        return labels
    cap = int(np.floor(max_area_px))
    if cap <= 0:
        # Pathologically small cap — wipe every labeled pixel.
        labels[labels > 0] = 0
        return labels
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    for k, c in zip(unique_labels, counts):
        if c <= cap:
            continue
        domain_mask = labels == k
        d = dist[domain_mask]
        # k-th smallest distance (0-indexed cap-1) is the inclusive cutoff.
        threshold = float(np.partition(d, cap - 1)[cap - 1])
        labels[domain_mask & (dist > threshold)] = 0
    return labels


def _compute_tessellation(
    signal: np.ndarray,
    tissue: np.ndarray,
    seeds_rc: np.ndarray,
    pixel_size_um: float,
    signal_influence: float,
    smoothing_sigma_um: float,
    min_signal: float = 0.0,
    min_area_um2: float = 0.0,
    max_area_um2: float = 0.0,
) -> np.ndarray:
    """Run the signal-guided seeded watershed with size + signal constraints.

    Returns int32 domain labels. Label ``k`` corresponds to ``seeds_rc[k-1]``
    if that seed survived the min-area pruning AND landed inside the
    effective mask, else label ``k`` is absent.
    """
    shape = signal.shape
    if seeds_rc.shape[0] == 0:
        return np.zeros(shape, dtype=np.int32)

    ppx2 = max(float(pixel_size_um) ** 2, 1e-12)
    min_area_px = float(min_area_um2) / ppx2 if min_area_um2 > 0 else 0.0
    max_area_px = float(max_area_um2) / ppx2 if max_area_um2 > 0 else 0.0

    signal_f = signal.astype(np.float32, copy=False)
    sigma_px = max(float(smoothing_sigma_um) / max(float(pixel_size_um), 1e-9), 0.0)
    if sigma_px > 0:
        signal_smooth = gaussian(signal_f, sigma=sigma_px, preserve_range=True).astype(
            np.float32, copy=False
        )
    else:
        signal_smooth = signal_f
    signal_norm = signal_smooth / max(float(signal_smooth.max()), 1.0)

    # Effective tessellation space: tissue minus any pixel whose smoothed
    # signal is below the min_signal floor. Pixels outside this mask never
    # get a domain label.
    if min_signal > 0:
        effective_mask = tissue & (signal_smooth >= float(min_signal))
    else:
        effective_mask = tissue

    markers = np.zeros(shape, dtype=np.int32)
    rows = np.clip(np.round(seeds_rc[:, 0]).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.round(seeds_rc[:, 1]).astype(int), 0, shape[1] - 1)
    markers[rows, cols] = np.arange(1, seeds_rc.shape[0] + 1, dtype=np.int32)

    si = float(signal_influence)

    def _watershed_round(current_markers: np.ndarray):
        """Run one watershed pass for the given marker set; apply max-area
        trim afterward. Returns ``(labels, dist_to_nearest_seed)`` so callers
        can re-use the distance map for downstream area trimming.
        """
        seed_indicator = current_markers > 0
        if not seed_indicator.any():
            return (
                np.zeros(shape, dtype=np.int32),
                np.zeros(shape, dtype=np.float32),
            )
        dist_local = ndi.distance_transform_edt(~seed_indicator).astype(
            np.float32, copy=False
        )
        dist_norm = dist_local / max(float(dist_local.max()), 1.0)
        elevation = si * (1.0 - signal_norm) + (1.0 - si) * dist_norm
        labels_local = watershed(
            elevation, markers=current_markers, mask=effective_mask
        ).astype(np.int32)
        if max_area_px > 0:
            labels_local = _trim_to_max_area(labels_local, dist_local, max_area_px)
        return labels_local, dist_local

    labels, _dist = _watershed_round(markers)

    if min_area_px > 0:
        for _ in range(_MIN_AREA_ITER_CAP):
            unique_labels, counts = np.unique(
                labels[labels > 0], return_counts=True
            )
            if unique_labels.size == 0:
                break
            tiny = unique_labels[counts < min_area_px]
            if tiny.size == 0:
                break
            tiny_positions = np.isin(markers, tiny)
            if not tiny_positions.any():
                break
            markers = markers.copy()
            markers[tiny_positions] = 0
            labels, _dist = _watershed_round(markers)

    return labels


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Build the tessellation submenu widget."""
    section = CollapsibleSection("Tessellation")

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(4)

    signal_slider = NumericSlider("Signal influence", 0.0, 1.0, step=0.01, value=0.5)
    sigma_slider = NumericSlider(
        "Smoothing σ (µm)", 0.0, 10.0, step=0.1, value=2.0, decimals=1
    )
    min_signal_slider = HistogramSlider(
        "Min signal", 0.0, 1.0, step=1.0, value=0.0, decimals=1
    )
    min_area_slider = NumericSlider(
        "Min domain area (µm²)", 0.0, 5000.0, step=25.0, value=0.0, decimals=0
    )
    max_area_slider = NumericSlider(
        "Max domain area (µm²)", 0.0, 100000.0, step=500.0, value=0.0, decimals=0
    )
    run_button = QPushButton("Run tessellation")
    count_label = QLabel("0 domains generated")
    count_label.setAlignment(Qt.AlignLeft)

    content_layout.addWidget(signal_slider)
    content_layout.addWidget(sigma_slider)
    content_layout.addWidget(min_signal_slider)
    content_layout.addWidget(min_area_slider)
    content_layout.addWidget(max_area_slider)
    content_layout.addWidget(run_button)
    content_layout.addWidget(count_label)
    section.set_content(content)

    # Disabled until seeds become available.
    content.setEnabled(False)

    suppress: dict = {"on": False}

    def _update_layer(domain_labels: np.ndarray) -> None:
        """Push labels to the napari Labels layer, creating or replacing in place."""
        if viewer is None:
            return
        pixel_size_um = (
            state.image.pixel_size_um if state.image is not None else 1.0
        )
        scale = (pixel_size_um, pixel_size_um)
        existing = None
        for layer in viewer.layers:
            if layer.name == _LAYER_NAME:
                existing = layer
                break
        if existing is None:
            viewer.add_labels(
                domain_labels,
                name=_LAYER_NAME,
                scale=scale,
                opacity=0.5,
            )
        else:
            existing.data = domain_labels
            try:
                existing.scale = scale
            except Exception:
                pass
            existing.opacity = 0.5

    def _recompute() -> None:
        if state.image is None or state.nuclei is None or state.seeds is None:
            return

        kept = np.asarray(state.seeds.kept_indices)
        centroids = state.nuclei.centroids
        signal_influence = signal_slider.value()
        smoothing_sigma_um = sigma_slider.value()
        min_signal = min_signal_slider.value()
        min_area_um2 = min_area_slider.value()
        max_area_um2 = max_area_slider.value()

        if kept.size == 0:
            domain_labels = np.zeros(state.image.signal.shape, dtype=np.int32)
            n_unique = 0
        else:
            seeds_rc = centroids[kept]
            domain_labels = _compute_tessellation(
                signal=state.image.signal,
                tissue=state.image.tissue_mask,
                seeds_rc=seeds_rc,
                pixel_size_um=state.image.pixel_size_um,
                signal_influence=signal_influence,
                smoothing_sigma_um=smoothing_sigma_um,
                min_signal=min_signal,
                min_area_um2=min_area_um2,
                max_area_um2=max_area_um2,
            )
            n_unique = int(np.unique(domain_labels[domain_labels > 0]).size)

        count_label.setText(f"{n_unique} domains generated")
        _update_layer(domain_labels)

        state.set(
            "tessellation",
            TessellationResult(
                domain_labels=domain_labels,
                params={
                    "signal_influence": float(signal_influence),
                    "smoothing_sigma_um": float(smoothing_sigma_um),
                    "min_signal": float(min_signal),
                    "min_area_um2": float(min_area_um2),
                    "max_area_um2": float(max_area_um2),
                },
            ),
        )

    def _on_run_clicked() -> None:
        if state.image is None or state.nuclei is None or state.seeds is None:
            return
        run_button.setEnabled(False)
        count_label.setText("Running tessellation...")
        try:
            _recompute()
        finally:
            run_button.setEnabled(True)

    run_button.clicked.connect(_on_run_clicked)

    def _on_seeds_changed() -> None:
        if state.seeds is None:
            content.setEnabled(False)
            # Nuclei was re-run — old tessellation is meaningless. Drop the
            # layer entirely.
            if viewer is not None:
                for layer in list(viewer.layers):
                    if layer.name == _LAYER_NAME:
                        viewer.layers.remove(layer)
                        break
            count_label.setText("0 domains generated")
            return
        content.setEnabled(True)
        # New (or refreshed) seeds: prompt the user to click Run. Don't
        # auto-compute — the watershed can be slow on large images. We keep
        # any existing "Domains" layer visible (stale) so the user can
        # compare against the previous tessellation.
        n_seeds = int(state.seeds.kept_indices.size)
        verb = "re-tessellate" if any(
            getattr(layer, "name", None) == _LAYER_NAME for layer in viewer.layers
        ) else "tessellate"
        count_label.setText(f"{n_seeds} seeds — click Run to {verb}")

    def _on_image_changed() -> None:
        """Reconfigure the min-signal HistogramSlider from the loaded image's
        in-tissue signal distribution."""
        if state.image is None:
            return
        img = state.image
        sig_in_tissue = img.signal[img.tissue_mask]
        if sig_in_tissue.size == 0:
            return
        upper = float(np.percentile(sig_in_tissue, 99.5))
        if upper <= 0:
            upper = float(sig_in_tissue.max() or 1.0)
        suppress["on"] = True
        try:
            min_signal_slider.set_range(0.0, max(upper, 1.0), step=1.0)
            min_signal_slider.set_data(sig_in_tissue, bins=100)
            min_signal_slider.set_value(0.0)
        finally:
            suppress["on"] = False

    state.subscribe("seeds", _on_seeds_changed)
    state.subscribe("image", _on_image_changed)

    # Handle the case where image was already set before we subscribed.
    if state.image is not None:
        _on_image_changed()

    return section
