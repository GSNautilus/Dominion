"""Submenu 3 — GFAP-guided watershed tessellation of astrocyte territories.

This widget watches the ``seeds`` slot. Whenever it changes, the widget
runs a seeded watershed on the GFAP channel masked to the tissue, using
the centroids of the kept seeds as marker positions. The elevation
combines a smoothed GFAP image (low-elevation = bright = "inside" an
astrocyte) and a distance transform from the seed markers, blended by
the ``GFAP influence`` slider. The result lands in ``state.tessellation``.
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


_LAYER_NAME = "Astrocyte territories"


def _compute_tessellation(
    gfap: np.ndarray,
    tissue: np.ndarray,
    seeds_rc: np.ndarray,
    pixel_size_um: float,
    gfap_influence: float,
    smoothing_sigma_um: float,
) -> np.ndarray:
    """Run the GFAP-guided seeded watershed and return int32 territory labels."""
    shape = gfap.shape
    if seeds_rc.shape[0] == 0:
        return np.zeros(shape, dtype=np.int32)

    gfap_f = gfap.astype(np.float32, copy=False)
    sigma_px = max(float(smoothing_sigma_um) / max(float(pixel_size_um), 1e-9), 0.0)
    if sigma_px > 0:
        gfap_smooth = gaussian(gfap_f, sigma=sigma_px, preserve_range=True).astype(
            np.float32, copy=False
        )
    else:
        gfap_smooth = gfap_f
    gfap_norm = gfap_smooth / max(float(gfap_smooth.max()), 1.0)

    markers = np.zeros(shape, dtype=np.int32)
    rows = np.clip(np.round(seeds_rc[:, 0]).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.round(seeds_rc[:, 1]).astype(int), 0, shape[1] - 1)
    markers[rows, cols] = np.arange(1, seeds_rc.shape[0] + 1, dtype=np.int32)

    seed_indicator = markers > 0
    dist = ndi.distance_transform_edt(~seed_indicator).astype(np.float32)
    dist_norm = dist / max(float(dist.max()), 1.0)

    gi = float(gfap_influence)
    elevation = gi * (1.0 - gfap_norm) + (1.0 - gi) * dist_norm

    return watershed(elevation, markers=markers, mask=tissue).astype(np.int32)


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Build the tessellation submenu widget."""
    section = CollapsibleSection("Tessellation")

    content = QWidget()
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(4)

    gfap_slider = NumericSlider("GFAP influence", 0.0, 1.0, step=0.01, value=0.5)
    sigma_slider = NumericSlider(
        "Smoothing σ (µm)", 0.0, 10.0, step=0.1, value=2.0, decimals=1
    )
    run_button = QPushButton("Run tessellation")
    count_label = QLabel("0 territories generated")
    count_label.setAlignment(Qt.AlignLeft)

    content_layout.addWidget(gfap_slider)
    content_layout.addWidget(sigma_slider)
    content_layout.addWidget(run_button)
    content_layout.addWidget(count_label)
    section.set_content(content)

    # Disabled until seeds become available.
    content.setEnabled(False)

    def _update_layer(territory_labels: np.ndarray) -> None:
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
                territory_labels,
                name=_LAYER_NAME,
                scale=scale,
                opacity=0.5,
            )
        else:
            existing.data = territory_labels
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
        gfap_influence = gfap_slider.value()
        smoothing_sigma_um = sigma_slider.value()

        if kept.size == 0:
            territory_labels = np.zeros(state.image.gfap.shape, dtype=np.int32)
            n_unique = 0
        else:
            seeds_rc = centroids[kept]
            territory_labels = _compute_tessellation(
                gfap=state.image.gfap,
                tissue=state.image.tissue_mask,
                seeds_rc=seeds_rc,
                pixel_size_um=state.image.pixel_size_um,
                gfap_influence=gfap_influence,
                smoothing_sigma_um=smoothing_sigma_um,
            )
            n_unique = int(np.unique(territory_labels[territory_labels > 0]).size)

        count_label.setText(f"{n_unique} territories generated")
        _update_layer(territory_labels)

        state.set(
            "tessellation",
            TessellationResult(
                territory_labels=territory_labels,
                params={
                    "gfap_influence": float(gfap_influence),
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
            count_label.setText("0 territories generated")
            return
        content.setEnabled(True)
        # New (or refreshed) seeds: prompt the user to click Run. Don't
        # auto-compute — the watershed can be slow on large images. We keep
        # any existing "Astrocyte territories" layer visible (stale) so the
        # user can compare against the previous tessellation.
        n_seeds = int(state.seeds.kept_indices.size)
        verb = "re-tessellate" if any(
            getattr(layer, "name", None) == _LAYER_NAME for layer in viewer.layers
        ) else "tessellate"
        count_label.setText(f"{n_seeds} seeds — click Run to {verb}")

    state.subscribe("seeds", _on_seeds_changed)

    return section
