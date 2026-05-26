"""Submenu 3 — signal-guided watershed tessellation of object domains.

This widget watches the ``seeds`` slot. Whenever it changes, the widget
runs a seeded watershed on the signal channel masked to the tissue,
using the centroids of the kept seeds as marker positions. The
elevation combines a smoothed signal image (low-elevation = bright =
"inside" an object) and a distance transform from the seed markers,
blended by the ``Signal influence`` slider. The result lands in
``state.tessellation``.
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
from .widgets.common import CollapsibleSection, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


_LAYER_NAME = "Domains"


def _compute_tessellation(
    signal: np.ndarray,
    tissue: np.ndarray,
    seeds_rc: np.ndarray,
    pixel_size_um: float,
    signal_influence: float,
    smoothing_sigma_um: float,
) -> np.ndarray:
    """Run the signal-guided seeded watershed and return int32 domain labels."""
    shape = signal.shape
    if seeds_rc.shape[0] == 0:
        return np.zeros(shape, dtype=np.int32)

    signal_f = signal.astype(np.float32, copy=False)
    sigma_px = max(float(smoothing_sigma_um) / max(float(pixel_size_um), 1e-9), 0.0)
    if sigma_px > 0:
        signal_smooth = gaussian(signal_f, sigma=sigma_px, preserve_range=True).astype(
            np.float32, copy=False
        )
    else:
        signal_smooth = signal_f
    signal_norm = signal_smooth / max(float(signal_smooth.max()), 1.0)

    markers = np.zeros(shape, dtype=np.int32)
    rows = np.clip(np.round(seeds_rc[:, 0]).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.round(seeds_rc[:, 1]).astype(int), 0, shape[1] - 1)
    markers[rows, cols] = np.arange(1, seeds_rc.shape[0] + 1, dtype=np.int32)

    seed_indicator = markers > 0
    dist = ndi.distance_transform_edt(~seed_indicator).astype(np.float32)
    dist_norm = dist / max(float(dist.max()), 1.0)

    si = float(signal_influence)
    elevation = si * (1.0 - signal_norm) + (1.0 - si) * dist_norm

    return watershed(elevation, markers=markers, mask=tissue).astype(np.int32)


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
    run_button = QPushButton("Run tessellation")
    count_label = QLabel("0 domains generated")
    count_label.setAlignment(Qt.AlignLeft)

    content_layout.addWidget(signal_slider)
    content_layout.addWidget(sigma_slider)
    content_layout.addWidget(run_button)
    content_layout.addWidget(count_label)
    section.set_content(content)

    # Disabled until seeds become available.
    content.setEnabled(False)

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

    state.subscribe("seeds", _on_seeds_changed)

    return section
