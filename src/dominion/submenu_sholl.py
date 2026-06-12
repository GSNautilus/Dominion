"""Submenu — Sholl analysis on per-cell skeletons (Feature 3).

After ``Skeletons`` has produced a SkeletonsResult, this widget computes
per-cell Sholl intersection profiles using concentric rings spaced
``ring_spacing_um`` µm apart. Results land in ``state.sholl`` and are
included in the Measurements CSV export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .sholl import sholl_for_skeletons
from .state import AppState
from .types import SholResult
from .widgets.common import CollapsibleSection, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the Sholl-analysis submenu."""
    section = CollapsibleSection("Sholl analysis", collapsed=True)

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    spacing_slider = NumericSlider(
        "Ring spacing (µm)", 0.5, 50.0, step=0.5, value=5.0, decimals=1
    )
    run_button = QPushButton("Run Sholl analysis")
    summary_label = QLabel("No skeletons yet.")
    summary_label.setAlignment(Qt.AlignLeft)
    summary_label.setWordWrap(True)

    layout.addWidget(spacing_slider)
    layout.addWidget(run_button)
    layout.addWidget(summary_label)
    section.set_content(content)
    content.setEnabled(False)

    def _on_run_clicked() -> None:
        if state.image is None or state.skeletons is None:
            return
        sk = state.skeletons
        if not sk.per_domain:
            summary_label.setText("No usable skeletons.")
            return

        spacing = float(spacing_slider.value())
        run_button.setEnabled(False)
        summary_label.setText("Running Sholl analysis...")
        try:
            per_domain = sholl_for_skeletons(
                sk.skeleton_label_image,
                sk.per_domain,
                state.image.pixel_size_um,
                spacing,
            )
        finally:
            run_button.setEnabled(True)

        state.set(
            "sholl",
            SholResult(
                per_domain=per_domain,
                params={"ring_spacing_um": spacing},
            ),
        )

        if not per_domain:
            summary_label.setText("0 Sholl profiles produced.")
            return

        peaks = np.array(
            [d["peak_intersections"] for d in per_domain.values()], dtype=np.int32
        )
        peak_radii = np.array(
            [d["peak_radius_um"] for d in per_domain.values()], dtype=np.float64
        )
        max_radii = np.array(
            [d["max_radius_um"] for d in per_domain.values()], dtype=np.float64
        )
        summary_label.setText(
            f"{len(per_domain)} cells analyzed (spacing {spacing:.1f} µm).\n"
            f"Peak intersections: median {int(np.median(peaks))}, "
            f"max {int(peaks.max())}\n"
            f"Peak radius µm: median {np.median(peak_radii):.1f}, "
            f"range [{peak_radii.min():.1f}, {peak_radii.max():.1f}]\n"
            f"Max radius µm: median {np.median(max_radii):.1f}, "
            f"max {max_radii.max():.1f}"
        )

    run_button.clicked.connect(_on_run_clicked)

    def _on_skeletons_changed() -> None:
        if state.skeletons is None:
            content.setEnabled(False)
            summary_label.setText("No skeletons yet.")
            return
        content.setEnabled(True)
        n = len(state.skeletons.per_domain)
        summary_label.setText(f"{n} skeletons ready — click Run.")

    state.subscribe("skeletons", _on_skeletons_changed)

    def _get_settings() -> dict:
        return {"ring_spacing_um": float(spacing_slider.value())}

    def _apply_settings(s: dict) -> None:
        if "ring_spacing_um" in s:
            spacing_slider.set_value(float(s["ring_spacing_um"]))

    state.register_settings("sholl_analysis", _get_settings, _apply_settings)

    if state.skeletons is not None:
        _on_skeletons_changed()

    return section
