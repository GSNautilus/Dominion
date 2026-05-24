"""Submenu A — GFAP-only astrocyte seed-finding.

The alternative to the DAPI+classification pair (submenus 1 + 2). Reads
only the GFAP channel and produces a set of seeds directly, by either:

* finding local maxima of smoothed GFAP, or
* finding peaks of the distance transform of thresholded GFAP.

The user picks the algorithm from a combobox. Three sliders drive the
detection (smoothing, GFAP threshold, minimum peak distance) and are
gated on a Run button. A fourth slider — peak strength — stays live as
a post-Run re-filter on the cached peak list.

State flow: on Run, the filtered peaks are written into ``state.nuclei``
as a synthetic :class:`NucleiResult` (label_mask is empty; centroids are
the peak coords) and immediately published as a
:class:`AstrocyteSeedsResult` with ``kept_indices = arange(N)``. The
existing submenu 3 (tessellation) consumes those unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .seedfind import find_seeds_dist_transform_peaks, find_seeds_local_max
from .state import AppState
from .types import AstrocyteSeedsResult, NucleiResult
from .widgets.common import CollapsibleSection, HistogramSlider, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


_SEEDS_LAYER_NAME = "Astrocyte seeds"
_DEFAULT_SIZE_FACTOR = 6.0  # matches submenu 2 so napari UI behaves the same

_METHOD_LOCAL_MAX = "Local maxima of smoothed GFAP"
_METHOD_DIST_PEAKS = "Distance-transform peaks of thresholded GFAP"
_METHODS = [_METHOD_LOCAL_MAX, _METHOD_DIST_PEAKS]


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the GFAP-only seed-finding submenu."""
    section = CollapsibleSection("GFAP seed-finding")

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    # --- Method selector ---
    method_row = QWidget()
    method_layout = QHBoxLayout(method_row)
    method_layout.setContentsMargins(0, 0, 0, 0)
    method_layout.addWidget(QLabel("Method:"))
    method_combo = QComboBox()
    for m in _METHODS:
        method_combo.addItem(m)
    method_layout.addWidget(method_combo, 1)

    # --- Run-gated sliders ---
    sigma_slider = NumericSlider(
        "Smoothing σ (µm)", 0.0, 20.0, step=0.1, value=5.0, decimals=1
    )
    threshold_slider = HistogramSlider(
        "GFAP threshold", 0.0, 1.0, step=1.0, value=0.0, decimals=1
    )
    min_dist_slider = NumericSlider(
        "Min peak distance (µm)", 1.0, 100.0, step=0.5, value=15.0, decimals=1
    )

    run_button = QPushButton("Run seed-finding")

    # --- Live filter (active after first Run) ---
    peak_strength_slider = HistogramSlider(
        "Min peak intensity", 0.0, 1.0, step=0.001, value=0.0, decimals=3
    )

    summary_label = QLabel("No image yet.")

    layout.addWidget(method_row)
    layout.addWidget(sigma_slider)
    layout.addWidget(threshold_slider)
    layout.addWidget(min_dist_slider)
    layout.addWidget(run_button)
    layout.addWidget(peak_strength_slider)
    layout.addWidget(summary_label)

    section.set_content(content)
    content.setEnabled(False)
    peak_strength_slider.setEnabled(False)

    # Cache for results between Run and live filter
    cache: dict = {"peaks": None, "values": None, "method": None}
    suppress: dict = {"on": False}

    # --- Helpers --------------------------------------------------------

    def _get_layer():
        try:
            return viewer.layers[_SEEDS_LAYER_NAME]
        except (KeyError, IndexError):
            return None

    def _replace_layer(peaks: np.ndarray, pixel_size_um: float) -> None:
        """Drop+re-add the seeds Points layer to dodge napari's stale per-point
        property arrays when N changes (same trick as submenu 2)."""
        layer = _get_layer()
        if layer is not None:
            try:
                viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
        # All-yellow (every peak is a kept seed in this mode).
        n = peaks.shape[0]
        face = np.tile(np.array([1.0, 1.0, 0.0, 1.0]), (n, 1))
        viewer.add_points(
            peaks,
            name=_SEEDS_LAYER_NAME,
            face_color=face,
            size=_DEFAULT_SIZE_FACTOR * pixel_size_um,
            scale=(pixel_size_um, pixel_size_um),
            border_color="transparent",
        )

    def _drop_layer() -> None:
        layer = _get_layer()
        if layer is not None:
            try:
                viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass

    def _publish(filtered_peaks: np.ndarray, filtered_values: np.ndarray) -> None:
        """Push results into AppState as synthetic Nuclei + Seeds."""
        if state.image is None:
            return
        img = state.image
        n = filtered_peaks.shape[0]
        nuclei = NucleiResult(
            label_mask=np.zeros(img.gfap.shape, dtype=np.int32),
            centroids=filtered_peaks,
            params={"source": "gfap_seedfind", "method": cache["method"]},
        )
        # Setting nuclei clears seeds & tessellation in AppState; that's fine
        # — we set seeds right after.
        state.set("nuclei", nuclei)
        seeds = AstrocyteSeedsResult(
            kept_indices=np.arange(n, dtype=np.int64),
            scores=filtered_values.astype(np.float64, copy=False),
            params={
                "method": cache["method"],
                "sigma_um": float(sigma_slider.value()),
                "gfap_threshold": float(threshold_slider.value()),
                "min_distance_um": float(min_dist_slider.value()),
                "peak_strength_threshold": float(peak_strength_slider.value()),
            },
        )
        state.set("seeds", seeds)
        _replace_layer(filtered_peaks, img.pixel_size_um)

    def _apply_peak_strength_filter() -> None:
        """Filter cached peaks by peak_strength_slider and publish."""
        peaks = cache["peaks"]
        values = cache["values"]
        if peaks is None or values is None:
            return
        cutoff = float(peak_strength_slider.value())
        mask = values >= cutoff
        kept_peaks = peaks[mask]
        kept_values = values[mask]
        _publish(kept_peaks, kept_values)
        summary_label.setText(
            f"{len(peaks)} peaks → {len(kept_peaks)} seeds "
            f"(strength ≥ {cutoff:.3g})"
        )

    # --- Run handler ----------------------------------------------------

    def _on_run_clicked() -> None:
        if state.image is None:
            return
        img = state.image
        sigma_px = float(sigma_slider.value()) / max(float(img.pixel_size_um), 1e-9)
        min_dist_px = max(1, int(round(
            float(min_dist_slider.value()) / max(float(img.pixel_size_um), 1e-9)
        )))
        threshold = float(threshold_slider.value())
        method = method_combo.currentText()

        run_button.setEnabled(False)
        summary_label.setText("Running seed-finding...")
        try:
            if method == _METHOD_LOCAL_MAX:
                peaks, values = find_seeds_local_max(
                    img.gfap,
                    img.tissue_mask,
                    sigma_px=sigma_px,
                    threshold_abs=threshold,
                    min_distance_px=min_dist_px,
                )
            else:  # _METHOD_DIST_PEAKS
                peaks, values = find_seeds_dist_transform_peaks(
                    img.gfap,
                    img.tissue_mask,
                    sigma_px=sigma_px,
                    threshold_abs=threshold,
                    min_distance_px=min_dist_px,
                )
        finally:
            run_button.setEnabled(True)

        cache["peaks"] = peaks
        cache["values"] = values
        cache["method"] = method

        # Reconfigure peak-strength slider from the new value distribution.
        suppress["on"] = True
        try:
            if values.size == 0:
                peak_strength_slider.set_range(0.0, 1.0, step=0.001)
                peak_strength_slider.set_data(np.zeros(0, dtype=np.float32), bins=20)
                peak_strength_slider.set_value(0.0)
                peak_strength_slider.setEnabled(False)
            else:
                vmax = float(values.max())
                if vmax <= 0.0:
                    vmax = 1.0
                step = max(vmax / 1000.0, 1e-6)
                peak_strength_slider.set_range(0.0, vmax, step=step)
                peak_strength_slider.set_data(values, bins=80)
                peak_strength_slider.set_value(0.0)
                peak_strength_slider.setEnabled(True)
            # Update the slider's label to match the method semantics.
            _set_strength_label_for_method(method)
        finally:
            suppress["on"] = False

        _apply_peak_strength_filter()

    # --- Live filter handler --------------------------------------------

    def _on_peak_strength_changed(_value: float) -> None:
        if suppress["on"]:
            return
        if cache["peaks"] is None:
            return
        _apply_peak_strength_filter()

    # --- Subscriptions --------------------------------------------------

    def _on_image_changed() -> None:
        # Reset everything on a new image.
        cache["peaks"] = None
        cache["values"] = None
        cache["method"] = None
        _drop_layer()
        peak_strength_slider.setEnabled(False)
        if state.image is None:
            content.setEnabled(False)
            summary_label.setText("No image yet.")
            return
        content.setEnabled(True)

        # Configure the GFAP threshold slider from GFAP-in-tissue distribution.
        img = state.image
        gfap_in_tissue = img.gfap[img.tissue_mask]
        suppress["on"] = True
        try:
            if gfap_in_tissue.size == 0:
                t_upper = float(img.gfap.max() or 1.0)
                t_default = 0.0
            else:
                t_upper = float(np.percentile(gfap_in_tissue, 99.5))
                if t_upper <= 0:
                    t_upper = float(gfap_in_tissue.max() or 1.0)
                nonzero = gfap_in_tissue[gfap_in_tissue > 0]
                t_default = float(np.median(nonzero)) if nonzero.size else 0.0
            threshold_slider.set_range(0.0, max(t_upper, 1.0), step=1.0)
            threshold_slider.set_data(gfap_in_tissue, bins=100)
            threshold_slider.set_value(min(t_default, t_upper))
        finally:
            suppress["on"] = False

        summary_label.setText("Click Run to find astrocyte seeds.")

    def _set_strength_label_for_method(method: str) -> None:
        """Update the peak-strength slider's label to reflect what 'strength'
        means for the currently selected method."""
        if method == _METHOD_DIST_PEAKS:
            new_text = "Min peak depth (px)"
        else:
            new_text = "Min peak intensity"
        # Find the label inside HistogramSlider and update it.
        for lb in peak_strength_slider.findChildren(QLabel):
            text = lb.text()
            if (
                text.startswith("Min peak intensity")
                or text.startswith("Min peak depth")
            ):
                lb.setText(new_text)
                break

    # --- Wiring ---------------------------------------------------------

    method_combo.currentTextChanged.connect(_set_strength_label_for_method)
    peak_strength_slider.valueChanged.connect(_on_peak_strength_changed)
    run_button.clicked.connect(_on_run_clicked)
    state.subscribe("image", _on_image_changed)

    # Handle the case where image was already set before we subscribed.
    if state.image is not None:
        _on_image_changed()

    return section
