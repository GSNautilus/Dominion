"""Submenu — region-of-interest (ROI) restriction.

Owns a napari ``Shapes`` layer named ``ROI``. The user draws a polygon
or rectangle using napari's built-in tools; clicking Apply rasterises
the **last** drawn shape into a bool mask and publishes a new
``ImageData`` where ``tissue_mask`` is the intersection of the
signal-positive footprint with the ROI. Downstream stages then only
see pixels inside the ROI.

Pick-one semantics: only the last shape counts. If the user draws
three polygons and clicks Apply, only shape #3 becomes the ROI. This
keeps the UI simple; a union / invert story can come later.

ROIs are session-only — not saved by the Batch/Save-settings flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .state import AppState
from .widgets.common import CollapsibleSection

if TYPE_CHECKING:
    import napari  # noqa: F401


_ROI_LAYER = "ROI"


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the ROI submenu."""
    section = CollapsibleSection("ROI", collapsed=True)

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    hint = QLabel(
        "Select the ROI layer in napari and draw a polygon or rectangle. "
        "Click Apply to restrict analysis. Only the last drawn shape is used."
    )
    hint.setWordWrap(True)
    hint.setAlignment(Qt.AlignLeft)

    button_row = QWidget()
    row_layout = QHBoxLayout(button_row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    apply_button = QPushButton("Apply ROI")
    clear_button = QPushButton("Clear ROI")
    row_layout.addWidget(apply_button)
    row_layout.addWidget(clear_button)

    status_label = QLabel("No ROI (whole tissue).")
    status_label.setWordWrap(True)

    layout.addWidget(hint)
    layout.addWidget(button_row)
    layout.addWidget(status_label)
    section.set_content(content)
    content.setEnabled(False)

    def _get_or_create_shapes_layer():
        if _ROI_LAYER in viewer.layers:
            return viewer.layers[_ROI_LAYER]
        if state.image is None:
            return None
        pixel_size_um = state.image.pixel_size_um
        return viewer.add_shapes(
            name=_ROI_LAYER,
            scale=(pixel_size_um, pixel_size_um),
            face_color="transparent",
            edge_color="yellow",
            edge_width=3,
        )

    def _on_apply_clicked() -> None:
        if state.image is None:
            return
        img = state.image
        shapes_layer = viewer.layers[_ROI_LAYER] if _ROI_LAYER in viewer.layers else None
        if shapes_layer is None or len(shapes_layer.data) == 0:
            status_label.setText("Draw a shape on the ROI layer first.")
            return
        # Rasterise every shape and keep the last one only.
        masks = shapes_layer.to_masks(mask_shape=img.signal.shape)
        if masks.shape[0] == 0:
            status_label.setText("Draw a shape on the ROI layer first.")
            return
        roi_mask = np.asarray(masks[-1], dtype=bool)
        area_px = int(roi_mask.sum())
        if area_px == 0:
            status_label.setText("Last shape has zero area; nothing applied.")
            return
        state.set("image", img.with_roi_mask(roi_mask))
        # Tissue mask after intersection may be smaller than ROI (regions
        # outside signal + nuclei don't count).
        eligible_px = int(state.image.tissue_mask.sum())
        area_um2 = area_px * (img.pixel_size_um ** 2)
        status_label.setText(
            f"ROI applied: {area_um2:.0f} µm² drawn, "
            f"{eligible_px} tissue pixels inside "
            f"(out of {int(roi_mask.sum())} ROI pixels)."
        )

    def _on_clear_clicked() -> None:
        if state.image is None:
            return
        if state.image.roi_mask is None:
            # Nothing to clear on state, but tidy up the shapes layer too.
            pass
        else:
            state.set("image", state.image.with_roi_mask(None))
        # Remove any shapes so a fresh draw isn't obscured by old ones.
        if _ROI_LAYER in viewer.layers:
            shapes_layer = viewer.layers[_ROI_LAYER]
            if len(shapes_layer.data) > 0:
                shapes_layer.data = []
        status_label.setText("No ROI (whole tissue).")

    apply_button.clicked.connect(_on_apply_clicked)
    clear_button.clicked.connect(_on_clear_clicked)

    def _on_image_changed() -> None:
        if state.image is None:
            content.setEnabled(False)
            status_label.setText("No image loaded.")
            return
        content.setEnabled(True)
        # Make sure a ROI Shapes layer exists so the user can start drawing.
        _get_or_create_shapes_layer()
        if state.image.roi_mask is None:
            status_label.setText("No ROI (whole tissue).")
        else:
            area_um2 = (
                int(state.image.roi_mask.sum())
                * (state.image.pixel_size_um ** 2)
            )
            eligible_px = int(state.image.tissue_mask.sum())
            status_label.setText(
                f"ROI active: {area_um2:.0f} µm² drawn, "
                f"{eligible_px} tissue pixels inside."
            )

    state.subscribe("image", _on_image_changed)

    if state.image is not None:
        _on_image_changed()

    return section
