"""Submenu — channel assignment.

Two dropdowns that pick which channel of a multi-channel TIFF is used
as the "signal" (for tessellation / seed-finding / skeletons / Sholl)
and which is used as the "nuclei" channel (for nuclei-mode seeding).

Publishes a new :class:`ImageData` via ``state.set("image", ...)`` on
every change, which cascades downstream invalidations exactly like
loading a fresh image. Each downstream submenu already re-configures
its histograms on image change, so swapping channels reruns all the
per-image UI setup automatically.

This submenu also owns the napari image layers named ``Signal``,
``Nuclei``, and each ``channel_<n>`` extra — updating their ``data`` and
name-visibility when the assignment changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .state import AppState
from .widgets.common import CollapsibleSection

if TYPE_CHECKING:
    import napari  # noqa: F401


_SIGNAL_LAYER = "Signal"
_NUCLEI_LAYER = "Nuclei"
_NONE_LABEL = "(none)"
_EXTRA_COLORMAPS = ("magenta", "yellow", "cyan", "red", "gray")


def _percentile_limits(arr: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(arr, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the Channels submenu."""
    section = CollapsibleSection("Channels", collapsed=True)

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    def _row(label_text: str, combo: QComboBox) -> QWidget:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        lb = QLabel(label_text)
        lb.setMinimumWidth(60)
        rl.addWidget(lb)
        rl.addWidget(combo, 1)
        return row

    signal_combo = QComboBox()
    nuclei_combo = QComboBox()
    layout.addWidget(_row("Signal:", signal_combo))
    layout.addWidget(_row("Nuclei:", nuclei_combo))
    section.set_content(content)
    content.setEnabled(False)

    suppress: dict = {"on": False}

    # --- Layer management ------------------------------------------------

    def _refresh_layers() -> None:
        """Rebuild the Signal / Nuclei / extra-channel image layers to
        match the current channel assignment."""
        img = state.image
        if img is None:
            return
        scale = (img.pixel_size_um, img.pixel_size_um)

        # 1. Signal layer — always exists after image is loaded.
        if _SIGNAL_LAYER in viewer.layers:
            layer = viewer.layers[_SIGNAL_LAYER]
            layer.data = img.signal
            layer.contrast_limits = _percentile_limits(img.signal)
            layer.scale = scale
        else:
            viewer.add_image(
                img.signal,
                name=_SIGNAL_LAYER,
                colormap="green",
                blending="additive",
                contrast_limits=_percentile_limits(img.signal),
                scale=scale,
            )

        # 2. Nuclei layer — exists only when an image has a designated
        #    nuclei channel. Otherwise drop the layer.
        if img.nuclei is not None:
            if _NUCLEI_LAYER in viewer.layers:
                layer = viewer.layers[_NUCLEI_LAYER]
                layer.data = img.nuclei
                layer.contrast_limits = _percentile_limits(img.nuclei)
                layer.scale = scale
            else:
                viewer.add_image(
                    img.nuclei,
                    name=_NUCLEI_LAYER,
                    colormap="blue",
                    blending="additive",
                    contrast_limits=_percentile_limits(img.nuclei),
                    scale=scale,
                )
        else:
            if _NUCLEI_LAYER in viewer.layers:
                viewer.layers.remove(_NUCLEI_LAYER)

        # 3. Extra channels — drop any that are no longer extras, add /
        #    refresh those that are.
        wanted = set(img.extra_channels)
        for existing in list(viewer.layers):
            name = getattr(existing, "name", "")
            if name.startswith("channel_") and name not in wanted:
                viewer.layers.remove(name)
        for i, (name, arr) in enumerate(img.extra_channels.items()):
            if name in viewer.layers:
                layer = viewer.layers[name]
                layer.data = arr
                layer.contrast_limits = _percentile_limits(arr)
                layer.scale = scale
            else:
                viewer.add_image(
                    arr,
                    name=name,
                    colormap=_EXTRA_COLORMAPS[i % len(_EXTRA_COLORMAPS)],
                    blending="additive",
                    contrast_limits=_percentile_limits(arr),
                    scale=scale,
                    visible=False,
                )

    # --- Combobox population + change handler ---------------------------

    def _populate_combos() -> None:
        """Fill both combos from the current image's channel list, and
        select the currently-designated assignment."""
        img = state.image
        suppress["on"] = True
        try:
            signal_combo.clear()
            nuclei_combo.clear()
            if img is None:
                return
            channels = list(img.all_channels.keys())
            signal_combo.addItems(channels)
            nuclei_combo.addItem(_NONE_LABEL)
            nuclei_combo.addItems(channels)
            # Select current designation.
            si = signal_combo.findText(img.signal_channel_name)
            if si >= 0:
                signal_combo.setCurrentIndex(si)
            if img.nuclei_channel_name is None:
                nuclei_combo.setCurrentIndex(0)
            else:
                ni = nuclei_combo.findText(img.nuclei_channel_name)
                if ni >= 0:
                    nuclei_combo.setCurrentIndex(ni)
        finally:
            suppress["on"] = False

    def _on_combo_changed(_: str) -> None:
        if suppress["on"] or state.image is None:
            return
        signal_name = signal_combo.currentText()
        nuclei_text = nuclei_combo.currentText()
        nuclei_name = None if nuclei_text == _NONE_LABEL else nuclei_text
        if signal_name == nuclei_name:
            # Ignore — the user picked the same channel for both. Revert the
            # nuclei combo to (none).
            suppress["on"] = True
            try:
                nuclei_combo.setCurrentIndex(0)
            finally:
                suppress["on"] = False
            nuclei_name = None
        try:
            new_image = state.image.with_channel_assignment(signal_name, nuclei_name)
        except ValueError:
            return
        state.set("image", new_image)

    signal_combo.currentTextChanged.connect(_on_combo_changed)
    nuclei_combo.currentTextChanged.connect(_on_combo_changed)

    # --- Image subscription ----------------------------------------------

    def _on_image_changed() -> None:
        content.setEnabled(state.image is not None)
        _populate_combos()
        _refresh_layers()

    state.subscribe("image", _on_image_changed)

    # --- Settings round-trip ---------------------------------------------

    def _get_settings() -> dict:
        img = state.image
        if img is None:
            return {}
        return {
            "signal_channel_name": img.signal_channel_name,
            "nuclei_channel_name": img.nuclei_channel_name,
        }

    def _apply_settings(s: dict) -> None:
        img = state.image
        if img is None:
            return
        signal_name = s.get("signal_channel_name")
        nuclei_name = s.get("nuclei_channel_name")
        if signal_name is None or signal_name not in img.all_channels:
            return  # settings from a different image with different channels
        if nuclei_name is not None and nuclei_name not in img.all_channels:
            nuclei_name = None
        try:
            new_image = img.with_channel_assignment(signal_name, nuclei_name)
        except ValueError:
            return
        state.set("image", new_image)

    state.register_settings("channels", _get_settings, _apply_settings)

    # Handle the case where the image was set before we subscribed.
    if state.image is not None:
        _on_image_changed()

    return section
