"""Submenu — per-domain skeleton extraction (Feature 2).

After a tessellation, this widget skeletonizes each domain mask, roots
the resulting tree at the corresponding seed, and stashes per-domain
branch info in ``state.skeletons``. A napari Labels layer named
"Skeletons" is added or refreshed so you can see the tracings overlaid
on the signal.

The hard problem (per-cell separation) is already solved by the
tessellation step, so this is a straightforward skimage.skeletonize
inside each domain mask followed by skan for branch decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .skeletonize import skeletonize_domains
from .state import AppState
from .types import SkeletonsResult
from .widgets.common import CollapsibleSection, HistogramSlider, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


_LAYER_NAME = "Skeletons"


def _seed_positions_from_state(state: AppState) -> dict[int, tuple[float, float]]:
    """Recover the (row, col) seed coordinate for each surviving domain ID.

    Tessellation labels are dense (1..N at the time the watershed ran).
    A seed can disappear from the final domain_labels if min-area pruning
    dropped it or if min-signal carving left it with zero pixels. So we
    only return seeds for IDs that actually appear in ``domain_labels``.
    """
    if state.nuclei is None or state.seeds is None or state.tessellation is None:
        return {}
    centroids = state.nuclei.centroids
    kept = np.asarray(state.seeds.kept_indices)
    if kept.size == 0:
        return {}
    present = set(int(x) for x in np.unique(state.tessellation.domain_labels) if x > 0)
    out: dict[int, tuple[float, float]] = {}
    for k_minus_1, idx in enumerate(kept):
        k = k_minus_1 + 1  # labels are 1-indexed
        if k not in present:
            continue
        c = centroids[int(idx)]
        out[k] = (float(c[0]), float(c[1]))
    return out


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the skeleton-extraction submenu."""
    section = CollapsibleSection("Skeletons", collapsed=True)

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    # Optional signal floor restricts what gets traced inside each
    # domain. 0 = trace the full domain (default). Higher = trace only
    # brighter pixels (sparser tracing of soma + main processes).
    signal_trace_slider = HistogramSlider(
        "Min signal for tracing", 0.0, 1.0, step=1.0, value=0.0, decimals=1
    )
    # Geometric twig pruning. 0 = disabled.
    min_branch_slider = NumericSlider(
        "Min branch length (µm)", 0.0, 20.0, step=0.1, value=0.0, decimals=1
    )
    # Intensity-aware twig pruning. A twig is pruned only when both this
    # AND the length criterion fail — so a short bright branch survives.
    # 0 = disabled.
    min_branch_signal_slider = HistogramSlider(
        "Min branch signal", 0.0, 1.0, step=1.0, value=0.0, decimals=1
    )
    # Tree-topology enforcement. Astrocyte filaments are tree-like — when
    # ON, post-skeletonize we break cycles by dropping the dimmest pixel
    # per cycle.
    force_tree_checkbox = QCheckBox("Force tree topology (no loops)")
    force_tree_checkbox.setChecked(True)

    run_button = QPushButton("Run skeletonization")
    summary_label = QLabel("No tessellation yet.")
    summary_label.setAlignment(Qt.AlignLeft)
    summary_label.setWordWrap(True)

    layout.addWidget(signal_trace_slider)
    layout.addWidget(min_branch_slider)
    layout.addWidget(min_branch_signal_slider)
    layout.addWidget(force_tree_checkbox)
    layout.addWidget(run_button)
    layout.addWidget(summary_label)
    section.set_content(content)
    content.setEnabled(False)

    suppress: dict = {"on": False}

    def _update_layer(skeleton_label_image: np.ndarray) -> None:
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
                skeleton_label_image, name=_LAYER_NAME, scale=scale, opacity=0.9
            )
        else:
            existing.data = skeleton_label_image
            try:
                existing.scale = scale
            except Exception:
                pass
            existing.opacity = 0.9

    def _drop_layer() -> None:
        if viewer is None:
            return
        for layer in list(viewer.layers):
            if layer.name == _LAYER_NAME:
                viewer.layers.remove(layer)
                break

    def _on_run_clicked() -> None:
        if state.image is None or state.tessellation is None:
            return
        seeds = _seed_positions_from_state(state)
        if not seeds:
            summary_label.setText("No usable seeds for the current tessellation.")
            return

        signal_threshold = float(signal_trace_slider.value())
        min_branch_length_um = float(min_branch_slider.value())
        min_branch_signal = float(min_branch_signal_slider.value())
        force_tree = bool(force_tree_checkbox.isChecked())
        run_button.setEnabled(False)
        summary_label.setText("Running skeletonization...")
        try:
            per_domain, skel_label_image = skeletonize_domains(
                state.tessellation.domain_labels,
                seeds,
                state.image.pixel_size_um,
                # Always pass the signal — force_tree + min_branch_signal
                # both consume it, not just signal_threshold > 0.
                signal=state.image.signal,
                signal_threshold=signal_threshold,
                min_branch_length_um=min_branch_length_um,
                min_branch_signal=min_branch_signal,
                force_tree=force_tree,
            )
        finally:
            run_button.setEnabled(True)

        state.set(
            "skeletons",
            SkeletonsResult(
                per_domain=per_domain,
                skeleton_label_image=skel_label_image,
                params={
                    "signal_threshold": signal_threshold,
                    "min_branch_length_um": min_branch_length_um,
                    "min_branch_signal": min_branch_signal,
                    "force_tree": force_tree,
                },
            ),
        )

        if not per_domain:
            summary_label.setText("0 skeletons extracted.")
            return

        totals = np.array(
            [d["total_length_um"] for d in per_domain.values()], dtype=np.float64
        )
        branches = np.array(
            [d["n_branches"] for d in per_domain.values()], dtype=np.int32
        )
        endpoints = np.array(
            [d["n_endpoints"] for d in per_domain.values()], dtype=np.int32
        )
        summary_label.setText(
            f"{len(per_domain)} skeletons extracted.\n"
            f"Total length µm: median {np.median(totals):.0f}, "
            f"range [{totals.min():.0f}, {totals.max():.0f}]\n"
            f"Branches/cell: median {int(np.median(branches))}, "
            f"max {int(branches.max())}\n"
            f"Endpoints/cell: median {int(np.median(endpoints))}, "
            f"max {int(endpoints.max())}"
        )
        _update_layer(skel_label_image)

    run_button.clicked.connect(_on_run_clicked)

    def _on_tessellation_changed() -> None:
        if state.tessellation is None:
            content.setEnabled(False)
            _drop_layer()
            summary_label.setText("No tessellation yet.")
            return
        content.setEnabled(True)
        n = int(state.tessellation.domain_labels.max())
        summary_label.setText(f"{n} domains — click Run.")

    def _on_image_changed() -> None:
        """Reconfigure the signal-trace HistogramSlider from the loaded
        image's in-tissue signal distribution (same convention as the
        tessellation submenu's Min-signal slider)."""
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
            signal_trace_slider.set_range(0.0, max(upper, 1.0), step=1.0)
            signal_trace_slider.set_data(sig_in_tissue, bins=100)
            signal_trace_slider.set_value(0.0)
            # The branch-signal slider uses the same in-tissue distribution.
            min_branch_signal_slider.set_range(0.0, max(upper, 1.0), step=1.0)
            min_branch_signal_slider.set_data(sig_in_tissue, bins=100)
            min_branch_signal_slider.set_value(0.0)
        finally:
            suppress["on"] = False

    state.subscribe("tessellation", _on_tessellation_changed)
    state.subscribe("image", _on_image_changed)

    if state.image is not None:
        _on_image_changed()
    if state.tessellation is not None:
        _on_tessellation_changed()

    return section
