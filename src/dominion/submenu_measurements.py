"""Submenu — per-domain measurements.

After a tessellation, this widget computes per-domain regionprops for
every available channel (signal + nuclei + any extra channels). On Run,
it produces a :class:`MeasurementsResult` and pushes it into AppState
for downstream consumers (e.g. CSV export — Feature 4).

The measurements use the ``TessellationResult.domain_labels`` array
directly, so any min-signal carving done in the tessellation step is
already reflected — pixels outside the effective tessellation space
have label 0 and contribute to nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import regionprops_table

from .state import AppState
from .types import MeasurementsResult
from .widgets.common import CollapsibleSection

if TYPE_CHECKING:
    import napari  # noqa: F401


_INTENSITY_PROPS = (
    "label",
    "intensity_mean",
    "intensity_max",
    "intensity_min",
    "image_intensity",  # placeholder; replaced via extra_properties below
)


def _channels_in_image(image) -> dict[str, np.ndarray]:
    """Return all per-channel arrays present in the image, keyed by display name."""
    channels: dict[str, np.ndarray] = {"signal": image.signal}
    if image.nuclei is not None:
        channels["nuclei"] = image.nuclei
    channels.update(image.extra_channels)
    return channels


def _measure(
    domain_labels: np.ndarray,
    channels: dict[str, np.ndarray],
    pixel_size_um: float,
    selected: set[str],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Compute regionprops for every selected channel + shared morphology.

    Returns ``(domain_ids, per_channel, morphology)``.
    """
    # Morphology props are channel-independent — compute once.
    morph_table = regionprops_table(
        domain_labels,
        properties=(
            "label",
            "area",
            "centroid",
            "eccentricity",
            "solidity",
            "perimeter",
            "equivalent_diameter_area",
            "axis_major_length",
            "axis_minor_length",
        ),
    )
    domain_ids = morph_table["label"].astype(np.int32, copy=False)
    px2 = float(pixel_size_um) ** 2
    morphology: dict[str, np.ndarray] = {
        "area_um2": morph_table["area"].astype(np.float64) * px2,
        "centroid_y_px": morph_table["centroid-0"].astype(np.float64),
        "centroid_x_px": morph_table["centroid-1"].astype(np.float64),
        "eccentricity": morph_table["eccentricity"].astype(np.float64),
        "solidity": morph_table["solidity"].astype(np.float64),
        "perimeter_um": morph_table["perimeter"].astype(np.float64)
        * float(pixel_size_um),
        "equivalent_diameter_um": morph_table["equivalent_diameter_area"].astype(
            np.float64
        )
        * float(pixel_size_um),
        "axis_major_um": morph_table["axis_major_length"].astype(np.float64)
        * float(pixel_size_um),
        "axis_minor_um": morph_table["axis_minor_length"].astype(np.float64)
        * float(pixel_size_um),
    }

    per_channel: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in channels.items():
        if name not in selected:
            continue
        # regionprops needs intensity_image; we also fetch the std via an
        # extra_properties hook so the column is in the same table.
        def _std(_region_mask, intensities):
            return float(np.std(intensities))

        def _median(_region_mask, intensities):
            return float(np.median(intensities))

        def _sum(_region_mask, intensities):
            return float(np.sum(intensities))

        table = regionprops_table(
            domain_labels,
            intensity_image=arr,
            properties=(
                "label",
                "intensity_mean",
                "intensity_max",
                "intensity_min",
            ),
            extra_properties=(_std, _median, _sum),
        )
        # Align order with domain_ids (regionprops returns label-sorted, same as morph)
        per_channel[name] = {
            "mean": table["intensity_mean"].astype(np.float64),
            "max": table["intensity_max"].astype(np.float64),
            "min": table["intensity_min"].astype(np.float64),
            "std": table["_std"].astype(np.float64),
            "median": table["_median"].astype(np.float64),
            "sum": table["_sum"].astype(np.float64),
        }
    return domain_ids, per_channel, morphology


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the measurements submenu."""
    section = CollapsibleSection("Measurements")

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    hint = QLabel("Channels to measure:")
    layout.addWidget(hint)

    # Channels list lives in a scrollable container so a 6-channel image
    # doesn't stretch the dock.
    channels_container = QWidget()
    channels_layout = QVBoxLayout(channels_container)
    channels_layout.setContentsMargins(0, 0, 0, 0)
    channels_layout.setSpacing(2)
    scroll = QScrollArea()
    scroll.setWidget(channels_container)
    scroll.setWidgetResizable(True)
    scroll.setMaximumHeight(120)
    layout.addWidget(scroll)

    run_button = QPushButton("Run measurements")
    summary_label = QLabel("No tessellation yet.")
    summary_label.setAlignment(Qt.AlignLeft)
    summary_label.setWordWrap(True)

    layout.addWidget(run_button)
    layout.addWidget(summary_label)
    section.set_content(content)
    content.setEnabled(False)

    checkboxes: dict[str, QCheckBox] = {}

    def _clear_channels():
        # Remove every existing checkbox from the layout.
        while channels_layout.count():
            item = channels_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        checkboxes.clear()

    def _populate_channels():
        _clear_channels()
        if state.image is None:
            return
        for name in _channels_in_image(state.image).keys():
            cb = QCheckBox(name)
            cb.setChecked(True)  # everything on by default
            channels_layout.addWidget(cb)
            checkboxes[name] = cb
        channels_layout.addStretch(1)

    def _on_run_clicked():
        if state.image is None or state.tessellation is None:
            return
        domain_labels = state.tessellation.domain_labels
        if int(domain_labels.max()) == 0:
            summary_label.setText("No domains in tessellation.")
            return
        channels = _channels_in_image(state.image)
        selected = {name for name, cb in checkboxes.items() if cb.isChecked()}
        if not selected:
            summary_label.setText("No channels selected.")
            return

        run_button.setEnabled(False)
        summary_label.setText("Running measurements...")
        try:
            domain_ids, per_channel, morphology = _measure(
                domain_labels,
                channels,
                state.image.pixel_size_um,
                selected,
            )
        finally:
            run_button.setEnabled(True)

        state.set(
            "measurements",
            MeasurementsResult(
                domain_ids=domain_ids,
                per_channel=per_channel,
                morphology=morphology,
                params={"channels_measured": sorted(selected)},
            ),
        )
        summary_label.setText(
            f"{domain_ids.size} domains × {len(per_channel)} channels measured."
        )

    run_button.clicked.connect(_on_run_clicked)

    def _on_image_changed():
        _populate_channels()
        summary_label.setText(
            "No tessellation yet."
            if state.tessellation is None
            else f"{int(state.tessellation.domain_labels.max())} domains — click Run."
        )

    def _on_tessellation_changed():
        if state.tessellation is None:
            content.setEnabled(False)
            summary_label.setText("No tessellation yet.")
            return
        content.setEnabled(True)
        n = int(state.tessellation.domain_labels.max())
        summary_label.setText(f"{n} domains — click Run.")

    state.subscribe("image", _on_image_changed)
    state.subscribe("tessellation", _on_tessellation_changed)

    if state.image is not None:
        _on_image_changed()
    if state.tessellation is not None:
        _on_tessellation_changed()

    return section
