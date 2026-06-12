"""Submenu — per-domain measurements + CSV export (Features 1 + 4).

After a tessellation, this widget computes per-domain regionprops for
every available channel (signal + nuclei + any extra channels). On Run,
it produces a :class:`MeasurementsResult` and pushes it into AppState.
A second "Export CSV" button writes the per-domain table and a one-row
per-image summary to disk alongside the source image.

The measurements use the ``TessellationResult.domain_labels`` array
directly, so any min-signal carving done in the tessellation step is
already reflected — pixels outside the effective tessellation space
have label 0 and contribute to nothing.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
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


def _render_summary(m: MeasurementsResult) -> str:
    """One-screen text summary for the dock label after a Run."""
    lines = [
        f"{m.domain_ids.size} domains × {len(m.per_channel)} channels measured."
    ]
    if "area_um2" in m.morphology and m.domain_ids.size > 0:
        a = m.morphology["area_um2"]
        lines.append(
            f"Area µm²: median {np.median(a):.0f}, range [{a.min():.0f}, {a.max():.0f}]"
        )
    for ch in sorted(m.per_channel):
        means = m.per_channel[ch]["mean"]
        if means.size:
            lines.append(
                f"{ch} mean intensity: median {np.median(means):.1f}, "
                f"max {means.max():.0f}"
            )
    return "\n".join(lines)


_SKELETON_COLS = (
    "skel_total_length_um",
    "skel_n_branches",
    "skel_n_endpoints",
    "skel_n_branchpoints",
)

_SHOLL_COLS = (
    "sholl_peak_intersections",
    "sholl_peak_radius_um",
    "sholl_max_radius_um",
    "sholl_critical_radius_um",
    "sholl_auc",
    "sholl_ramification_index",
)


def _skel_row(skeletons, did: int) -> list[object]:
    if skeletons is None:
        return ["" for _ in _SKELETON_COLS]
    info = skeletons.per_domain.get(int(did))
    if info is None:
        return ["" for _ in _SKELETON_COLS]
    return [
        float(info["total_length_um"]),
        int(info["n_branches"]),
        int(info["n_endpoints"]),
        int(info["n_branchpoints"]),
    ]


def _sholl_row(sholl, did: int) -> list[object]:
    if sholl is None:
        return ["" for _ in _SHOLL_COLS]
    info = sholl.per_domain.get(int(did))
    if info is None:
        return ["" for _ in _SHOLL_COLS]
    return [
        int(info["peak_intersections"]),
        float(info["peak_radius_um"]),
        float(info["max_radius_um"]),
        float(info["critical_radius_um"]),
        int(info["auc"]),
        float(info["ramification_index"]),
    ]


def _write_domains_csv(
    path: Path,
    m: MeasurementsResult,
    pixel_size_um: float,
    skeletons=None,
    sholl=None,
) -> None:
    """One row per domain. Columns: morphology + (channel_stat) + optional
    skeleton + Sholl summary columns when their state slots are set."""
    morph_cols = list(m.morphology.keys())
    channel_stats: list[tuple[str, str]] = []
    for ch in sorted(m.per_channel):
        for stat in sorted(m.per_channel[ch]):
            channel_stats.append((ch, stat))

    header = (
        ["domain_id"]
        + morph_cols
        + [f"{ch}_{stat}" for ch, stat in channel_stats]
        + list(_SKELETON_COLS)
        + list(_SHOLL_COLS)
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, did in enumerate(m.domain_ids):
            row: list[object] = [int(did)]
            row.extend(float(m.morphology[k][i]) for k in morph_cols)
            row.extend(float(m.per_channel[ch][stat][i]) for ch, stat in channel_stats)
            row.extend(_skel_row(skeletons, int(did)))
            row.extend(_sholl_row(sholl, int(did)))
            w.writerow(row)


def _write_summary_csv(
    path: Path,
    m: MeasurementsResult,
    image,
    tessellation,
    seeds,
    nuclei,
    skeletons=None,
    sholl=None,
) -> None:
    """One row per image: population-level stats + provenance / params."""
    rows: list[tuple[str, object]] = []

    rows.append(("source_path", str(image.source_path)))
    rows.append(("pixel_size_um", float(image.pixel_size_um)))
    rows.append(("image_h_px", int(image.signal.shape[0])))
    rows.append(("image_w_px", int(image.signal.shape[1])))
    rows.append(
        (
            "tissue_area_um2",
            float(int(image.tissue_mask.sum()) * (image.pixel_size_um ** 2)),
        )
    )
    rows.append(("n_domains", int(m.domain_ids.size)))
    if "area_um2" in m.morphology and m.domain_ids.size > 0:
        a = m.morphology["area_um2"]
        rows.append(("domain_area_mean_um2", float(np.mean(a))))
        rows.append(("domain_area_median_um2", float(np.median(a))))
        rows.append(("domain_area_std_um2", float(np.std(a))))
        rows.append(("domain_area_p25_um2", float(np.percentile(a, 25))))
        rows.append(("domain_area_p75_um2", float(np.percentile(a, 75))))
        rows.append(("domain_area_min_um2", float(a.min())))
        rows.append(("domain_area_max_um2", float(a.max())))
        tissue_mm2 = float(int(image.tissue_mask.sum()) * (image.pixel_size_um ** 2)) / 1e6
        if tissue_mm2 > 0:
            rows.append(("domain_density_per_mm2", float(m.domain_ids.size / tissue_mm2)))
    for ch in sorted(m.per_channel):
        means = m.per_channel[ch]["mean"]
        if means.size:
            rows.append((f"{ch}_mean_intensity_median", float(np.median(means))))
            rows.append((f"{ch}_mean_intensity_mean", float(np.mean(means))))

    if skeletons is not None and skeletons.per_domain:
        per = skeletons.per_domain
        totals = np.array([d["total_length_um"] for d in per.values()], dtype=np.float64)
        branches = np.array([d["n_branches"] for d in per.values()], dtype=np.int32)
        endpoints = np.array([d["n_endpoints"] for d in per.values()], dtype=np.int32)
        branchpts = np.array([d["n_branchpoints"] for d in per.values()], dtype=np.int32)
        rows.append(("n_skeletons", int(len(per))))
        rows.append(("skel_total_length_median_um", float(np.median(totals))))
        rows.append(("skel_total_length_mean_um", float(np.mean(totals))))
        rows.append(("skel_branches_median", float(np.median(branches))))
        rows.append(("skel_endpoints_median", float(np.median(endpoints))))
        rows.append(("skel_branchpoints_median", float(np.median(branchpts))))

    if sholl is not None and sholl.per_domain:
        per = sholl.per_domain
        peaks = np.array(
            [d["peak_intersections"] for d in per.values()], dtype=np.int32
        )
        peak_radii = np.array(
            [d["peak_radius_um"] for d in per.values()], dtype=np.float64
        )
        max_radii = np.array(
            [d["max_radius_um"] for d in per.values()], dtype=np.float64
        )
        crit_radii = np.array(
            [d["critical_radius_um"] for d in per.values()], dtype=np.float64
        )
        aucs = np.array([d["auc"] for d in per.values()], dtype=np.int32)
        ramif = np.array(
            [d["ramification_index"] for d in per.values()], dtype=np.float64
        )
        rows.append(("n_sholl", int(len(per))))
        rows.append(("sholl_peak_intersections_median", float(np.median(peaks))))
        rows.append(("sholl_peak_radius_median_um", float(np.median(peak_radii))))
        rows.append(("sholl_max_radius_median_um", float(np.median(max_radii))))
        rows.append(("sholl_critical_radius_median_um", float(np.median(crit_radii))))
        rows.append(("sholl_auc_median", float(np.median(aucs))))
        rows.append(("sholl_ramification_index_median", float(np.median(ramif))))

    # Provenance: params from each upstream slot, flattened with a stage prefix.
    def _flatten(prefix: str, params: dict | None):
        if not params:
            return
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                v = ";".join(map(str, v))
            rows.append((f"{prefix}__{k}", v))

    _flatten("seeds", getattr(seeds, "params", None))
    _flatten("tessellation", getattr(tessellation, "params", None))
    _flatten("measurements", m.params)
    _flatten("nuclei", getattr(nuclei, "params", None))
    _flatten("skeletons", getattr(skeletons, "params", None))
    _flatten("sholl", getattr(sholl, "params", None))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, v])


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the measurements submenu."""
    section = CollapsibleSection("Measurements", collapsed=True)

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
    export_button = QPushButton("Export CSV...")
    summary_label = QLabel("No tessellation yet.")
    summary_label.setAlignment(Qt.AlignLeft)
    summary_label.setWordWrap(True)

    layout.addWidget(run_button)
    layout.addWidget(export_button)
    layout.addWidget(summary_label)
    section.set_content(content)
    content.setEnabled(False)
    export_button.setEnabled(False)

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
        summary_label.setText(_render_summary(state.measurements))
        export_button.setEnabled(True)

    run_button.clicked.connect(_on_run_clicked)

    def _on_export_clicked():
        if state.image is None or state.measurements is None:
            return
        src = Path(state.image.source_path)
        default_path = str(src.with_name(f"{src.stem}_domains.csv"))
        chosen, _ = QFileDialog.getSaveFileName(
            content,
            "Export per-domain CSV",
            default_path,
            "CSV files (*.csv);;All files (*)",
        )
        if not chosen:
            return
        domains_path = Path(chosen)
        summary_path = domains_path.with_name(
            domains_path.stem + "_summary" + domains_path.suffix
        )
        _write_domains_csv(
            domains_path,
            state.measurements,
            state.image.pixel_size_um,
            skeletons=state.skeletons,
            sholl=state.sholl,
        )
        _write_summary_csv(
            summary_path,
            state.measurements,
            state.image,
            state.tessellation,
            state.seeds,
            state.nuclei,
            skeletons=state.skeletons,
            sholl=state.sholl,
        )
        summary_label.setText(
            _render_summary(state.measurements)
            + f"\nSaved:\n  • {domains_path.name}\n  • {summary_path.name}"
        )

    export_button.clicked.connect(_on_export_clicked)

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
            export_button.setEnabled(False)
            summary_label.setText("No tessellation yet.")
            return
        content.setEnabled(True)
        n = int(state.tessellation.domain_labels.max())
        summary_label.setText(f"{n} domains — click Run.")

    def _on_measurements_changed():
        export_button.setEnabled(state.measurements is not None)

    state.subscribe("image", _on_image_changed)
    state.subscribe("tessellation", _on_tessellation_changed)
    state.subscribe("measurements", _on_measurements_changed)

    def _get_settings() -> dict:
        # Save the channel-checkbox state by channel name. Channels that
        # are absent on the next image just won't be re-checked.
        return {
            "channels_checked": [
                name for name, cb in checkboxes.items() if cb.isChecked()
            ],
        }

    def _apply_settings(s: dict) -> None:
        wanted = set(s.get("channels_checked", []) or [])
        if not wanted:
            return
        for name, cb in checkboxes.items():
            cb.setChecked(name in wanted)

    state.register_settings("measurements", _get_settings, _apply_settings)

    if state.image is not None:
        _on_image_changed()
    if state.tessellation is not None:
        _on_tessellation_changed()

    return section
