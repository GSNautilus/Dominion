"""Submenu 2 — object classification / seed selection.

For each candidate nucleus, compute a signal-based "object-likeness"
score over a disc of radius ``R`` around its centroid, then keep nuclei
whose score exceeds a user-tunable threshold ``theta``.

The score for nucleus *i* is

    score_i = sum_{p in P_i} ( alpha * signal[p]
                              + (1 - alpha) * (R_px - dist(p, c_i)) )

where ``P_i`` is the set of pixels within ``R_px`` of the centroid that
also lie in the tissue mask and exceed the signal threshold ``T``.

Four sliders drive the computation:

* ``T``      — signal intensity threshold (recompute scores)
* ``R``      — search radius in microns (recompute scores)
* ``alpha``  — intensity-vs-distance weighting in [0, 1] (recompute scores)
* ``theta``  — score threshold for keeping a seed (filter only)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .state import AppState
from .types import SeedsResult
from .widgets.common import CollapsibleSection, HistogramSlider, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


_SEEDS_LAYER_NAME = "Object seeds"
# Default per-point size at layer creation. Uniform across kept/rejected so
# napari's built-in "point size" UI slider can control all of them. After
# creation we never re-assign `layer.size`, so user resizing via the UI
# persists across Run / theta updates (until N changes, which forces a
# fresh layer). Fixed value (data units / pixels) so it doesn't shrink to
# invisibly small on high-resolution images.
_DEFAULT_POINT_SIZE = 24.0


def _compute_scores(
    signal: np.ndarray,
    tissue_mask: np.ndarray,
    centroids: np.ndarray,
    T: float,
    R_px: float,
    alpha: float,
) -> np.ndarray:
    """Per-nucleus object-likeness score (see module docstring)."""
    n = centroids.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    if n == 0 or R_px <= 0:
        return scores

    H, W = signal.shape
    r_ceil = int(np.ceil(R_px))
    r2 = R_px * R_px

    # Pre-cast for speed.
    signal_f = signal.astype(np.float64, copy=False)

    for i in range(n):
        cy, cx = float(centroids[i, 0]), float(centroids[i, 1])
        y0 = max(0, int(np.floor(cy)) - r_ceil)
        y1 = min(H, int(np.floor(cy)) + r_ceil + 1)
        x0 = max(0, int(np.floor(cx)) - r_ceil)
        x1 = min(W, int(np.floor(cx)) + r_ceil + 1)
        if y1 <= y0 or x1 <= x0:
            continue

        ys = np.arange(y0, y1, dtype=np.float64) - cy
        xs = np.arange(x0, x1, dtype=np.float64) - cx
        dy2 = ys[:, None] ** 2
        dx2 = xs[None, :] ** 2
        d2 = dy2 + dx2

        crop_signal = signal_f[y0:y1, x0:x1]
        crop_tissue = tissue_mask[y0:y1, x0:x1]

        mask = (d2 <= r2) & crop_tissue & (crop_signal >= T)
        if not mask.any():
            continue

        # Sum the score contributions over positive pixels only.
        dist = np.sqrt(d2[mask])
        intens = crop_signal[mask]
        scores[i] = float(
            alpha * intens.sum() + (1.0 - alpha) * (R_px - dist).sum()
        )

    return scores


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the object-classification submenu."""
    section = CollapsibleSection("Object classification")

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    t_slider = HistogramSlider(
        "Signal threshold T", 0.0, 1.0, step=1.0, value=0.0, decimals=1
    )
    r_slider = NumericSlider(
        "Search radius R (µm)", 1.0, 50.0, step=0.5, value=10.0, decimals=1
    )
    alpha_slider = NumericSlider(
        "Intensity ↔ distance weight α", 0.0, 1.0, step=0.01, value=0.5, decimals=2
    )
    theta_slider = HistogramSlider(
        "Seed threshold θ", 0.0, 1.0, step=0.001, value=0.0, decimals=3
    )
    run_button = QPushButton("Run classification")
    summary_label = QLabel("No nuclei yet.")

    layout.addWidget(t_slider)
    layout.addWidget(r_slider)
    layout.addWidget(alpha_slider)
    layout.addWidget(theta_slider)
    layout.addWidget(run_button)
    layout.addWidget(summary_label)

    section.set_content(content)
    content.setEnabled(False)

    # Mutable boxes so nested callbacks can share state without `nonlocal`
    # gymnastics across many handlers.
    cache: dict = {"scores": None, "centroids": None}
    suppress: dict = {"on": False}

    def _get_layer():
        try:
            return viewer.layers[_SEEDS_LAYER_NAME]
        except (KeyError, IndexError):
            return None

    def _ensure_layer(centroids: np.ndarray, pixel_size_um: float):
        layer = _get_layer()
        n = centroids.shape[0]
        # napari's Points layer keeps per-point face/border/size arrays of the
        # current length; assigning `layer.data` to a different-length array
        # triggers an internal resize that can misindex the stale property
        # arrays. Full-reset when N changes to avoid that.
        if layer is not None and len(layer.data) != n:
            try:
                viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass
            layer = None

        face = np.tile(np.array([0.5, 0.5, 0.5, 0.6]), (n, 1))
        if layer is None:
            layer = viewer.add_points(
                centroids,
                name=_SEEDS_LAYER_NAME,
                face_color=face,
                size=_DEFAULT_POINT_SIZE,
                scale=(pixel_size_um, pixel_size_um),
                border_color="transparent",
            )
        else:
            layer.data = centroids
            layer.scale = (pixel_size_um, pixel_size_um)
            layer.face_color = face
            # Intentionally NOT touching layer.size — that's the user's via napari's UI.
        return layer

    def _refresh_layer(kept_indices: np.ndarray):
        if state.image is None or state.nuclei is None:
            return
        layer = _get_layer()
        if layer is None:
            return
        n = state.nuclei.centroids.shape[0]
        face = np.tile(np.array([0.5, 0.5, 0.5, 0.6]), (n, 1))
        if kept_indices.size:
            face[kept_indices] = np.array([1.0, 1.0, 0.0, 1.0])
        layer.face_color = face
        # Intentionally NOT touching layer.size — that's the user's via napari's UI.

    def _publish_seeds(scores: np.ndarray, theta: float) -> np.ndarray:
        kept = np.where(scores >= theta)[0].astype(np.int64)
        params = {
            "T": float(t_slider.value()),
            "R_um": float(r_slider.value()),
            "alpha": float(alpha_slider.value()),
            "theta": float(theta),
        }
        # Publishing 'seeds' would clear downstream — that's the intended
        # AppState semantics, but we set without triggering nuclei-clearing.
        state.set(
            "seeds",
            SeedsResult(
                kept_indices=kept, scores=scores.copy(), params=params
            ),
        )
        summary_label.setText(
            f"{scores.size} nuclei → {kept.size} kept as object seeds"
        )
        _refresh_layer(kept)
        return kept

    def _recompute_scores():
        if state.image is None or state.nuclei is None:
            return
        img = state.image
        centroids = state.nuclei.centroids
        R_px = float(r_slider.value()) / float(img.pixel_size_um)
        T = float(t_slider.value())
        alpha = float(alpha_slider.value())

        scores = _compute_scores(
            img.signal, img.tissue_mask, centroids, T, R_px, alpha
        )
        cache["scores"] = scores
        cache["centroids"] = centroids

        # Reset theta range/histogram from new scores.
        smax = float(scores.max()) if scores.size else 1.0
        if smax <= 0.0:
            smax = 1.0
        # Use p99.5 as the visible upper bound so a few outliers don't
        # squash the histogram, but clamp slider range to true max so the
        # user can still threshold above all scores if they want.
        finite = scores[np.isfinite(scores)]
        upper = float(np.percentile(finite, 99.5)) if finite.size else smax
        if upper <= 0.0:
            upper = smax
        step = max(smax / 1000.0, 1e-6)
        suppress["on"] = True
        try:
            theta_slider.set_range(0.0, smax, step=step)
            theta_slider.set_data(scores, bins=80)
            # Default theta: median of nonzero scores, falling back to 0.
            nonzero = scores[scores > 0]
            default_theta = float(np.median(nonzero)) if nonzero.size else 0.0
            theta_slider.set_value(default_theta)
        finally:
            suppress["on"] = False
        # Now actually publish using the (possibly clamped) slider value.
        _publish_seeds(scores, float(theta_slider.value()))

    def _on_theta_changed(_value: float):
        if suppress["on"]:
            return
        if cache["scores"] is None:
            return
        _publish_seeds(cache["scores"], float(theta_slider.value()))

    def _on_run_clicked():
        # T/R/α are gated on this button; θ stays live as a re-filter.
        if state.image is None or state.nuclei is None:
            return
        run_button.setEnabled(False)
        summary_label.setText("Running classification...")
        try:
            _recompute_scores()
        finally:
            run_button.setEnabled(True)

    theta_slider.valueChanged.connect(_on_theta_changed)
    run_button.clicked.connect(_on_run_clicked)

    def _on_nuclei_changed():
        if state.nuclei is None or state.image is None:
            content.setEnabled(False)
            summary_label.setText("No nuclei yet.")
            return
        content.setEnabled(True)

        # Configure T slider from signal-in-tissue distribution.
        img = state.image
        signal_in_tissue = img.signal[img.tissue_mask]
        if signal_in_tissue.size == 0:
            t_upper = float(img.signal.max() or 1.0)
            t_default = 0.0
        else:
            t_upper = float(np.percentile(signal_in_tissue, 99.5))
            if t_upper <= 0:
                t_upper = float(signal_in_tissue.max() or 1.0)
            nonzero = signal_in_tissue[signal_in_tissue > 0]
            t_default = (
                float(np.median(nonzero)) if nonzero.size else 0.0
            )
        suppress["on"] = True
        try:
            t_slider.set_range(0.0, max(t_upper, 1.0), step=1.0)
            t_slider.set_data(signal_in_tissue, bins=100)
            t_slider.set_value(min(t_default, t_upper))
        finally:
            suppress["on"] = False

        # Build/refresh the points layer with all centroids (all gray until
        # the user clicks Run and we have scores to color by).
        _ensure_layer(state.nuclei.centroids, img.pixel_size_um)

        # Stale: any cached scores are for old nuclei. Wait for Run.
        cache["scores"] = None
        cache["centroids"] = None
        n = int(state.nuclei.centroids.shape[0])
        summary_label.setText(f"{n} nuclei — click Run to classify")

    def _on_image_changed():
        # New image clears nuclei/seeds; just disable until nuclei re-set.
        content.setEnabled(False)
        cache["scores"] = None
        cache["centroids"] = None
        summary_label.setText("No nuclei yet.")
        # Drop any stale seeds layer.
        layer = _get_layer()
        if layer is not None:
            try:
                viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass

    state.subscribe("nuclei", _on_nuclei_changed)
    state.subscribe("image", _on_image_changed)

    def _get_settings() -> dict:
        return {
            "T": float(t_slider.value()),
            "R_um": float(r_slider.value()),
            "alpha": float(alpha_slider.value()),
            "theta": float(theta_slider.value()),
        }

    def _set_widened(slider, value: float) -> None:
        slider.set_value(value)
        if abs(slider.value() - value) > 1e-6:
            slider.set_range(0.0, max(value * 1.2, 1.0), step=max(value / 1000, 1e-6))
            slider.set_value(value)

    def _apply_settings(s: dict) -> None:
        suppress["on"] = True
        try:
            if "T" in s:
                _set_widened(t_slider, float(s["T"]))
            if "R_um" in s:
                r_slider.set_value(float(s["R_um"]))
            if "alpha" in s:
                alpha_slider.set_value(float(s["alpha"]))
            if "theta" in s:
                _set_widened(theta_slider, float(s["theta"]))
        finally:
            suppress["on"] = False

    state.register_settings("object_classification", _get_settings, _apply_settings)

    # Handle the case where nuclei were already set before we subscribed.
    if state.nuclei is not None and state.image is not None:
        _on_nuclei_changed()

    return section
