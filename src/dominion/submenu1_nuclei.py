"""Submenu 1 — StarDist-based nuclei segmentation.

Provides a small Qt panel (two threshold sliders + a Run button + a
status label) that, on Run, executes the pretrained StarDist
``2D_versatile_fluo`` model on the loaded DAPI channel. Results are
cached to disk keyed on (image, params) and stored back into the shared
:class:`AppState` so downstream submenus can consume them. Heavy work
runs on a :func:`napari.qt.thread_worker` to keep the UI responsive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from .cache import cache_path, load_npz, save_npz
from .state import AppState
from .types import NucleiResult
from .widgets.common import CollapsibleSection, NumericSlider

if TYPE_CHECKING:
    import napari  # noqa: F401


_NUCLEI_MASK_LAYER = "Nuclei (mask)"
_NUCLEI_POINTS_LAYER = "Nuclei (centroids)"


def _auto_n_tiles(shape: tuple[int, int], target_tile_px: int = 2000) -> tuple[int, int]:
    """Pick ``(n_y, n_x)`` so each StarDist tile is ~``target_tile_px`` on a side.

    Sized to fit a single conv layer's activations on a 12 GB consumer GPU
    with headroom; smaller GPUs may need a smaller target.
    """
    ny = max(1, -(-shape[0] // target_tile_px))
    nx = max(1, -(-shape[1] // target_tile_px))
    return (ny, nx)


def _run_stardist(
    dapi: np.ndarray, prob_thresh: float, nms_thresh: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Run pretrained StarDist on a 2D DAPI image.

    Returns ``(label_mask, centroids, n_tiles)`` where ``label_mask`` is
    int32, ``centroids`` is an (N, 2) float array of ``(row, col)`` pixel
    coords, and ``n_tiles`` is the tiling actually used.
    """
    # Imports are deferred so building the widget doesn't drag TensorFlow
    # into memory until the user actually clicks Run.
    from csbdeep.utils import normalize
    from skimage.measure import regionprops_table
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    normalized = normalize(dapi, 1, 99.8, axis=(0, 1))
    n_tiles = _auto_n_tiles(normalized.shape)
    labels, _details = model.predict_instances(
        normalized,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=n_tiles,
    )
    label_mask = labels.astype(np.int32, copy=False)

    if label_mask.max() == 0:
        centroids = np.zeros((0, 2), dtype=float)
    else:
        table = regionprops_table(label_mask, properties=("centroid",))
        centroids = np.column_stack([table["centroid-0"], table["centroid-1"]]).astype(
            float, copy=False
        )
    return label_mask, centroids, n_tiles


def _update_or_add_labels(
    viewer: "napari.Viewer",
    name: str,
    data: np.ndarray,
    *,
    scale: tuple[float, float],
    opacity: float,
):
    """Add a Labels layer named ``name`` or, if present, replace its data in place."""
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        layer.scale = scale
        layer.opacity = opacity
        return layer
    return viewer.add_labels(data, name=name, scale=scale, opacity=opacity)


def _add_points_compat(viewer, **kwargs):
    """Call ``viewer.add_points`` accommodating the napari edge/border rename.

    napari renamed ``edge_color`` to ``border_color`` on Points layers in
    ~0.5; we prefer ``border_color`` and fall back if napari rejects it.
    """
    edge_color = kwargs.pop("edge_color", None)
    if edge_color is not None and "border_color" not in kwargs:
        kwargs["border_color"] = edge_color
    try:
        return viewer.add_points(**kwargs)
    except TypeError:
        # Older napari: retry with edge_color.
        border = kwargs.pop("border_color", None)
        if border is not None:
            kwargs["edge_color"] = border
        return viewer.add_points(**kwargs)


def _set_points_colors(layer, *, edge_color: str, face_color: str) -> None:
    """Set border/face colors on a Points layer across napari versions."""
    for attr in ("border_color", "edge_color"):
        if hasattr(layer, attr):
            try:
                setattr(layer, attr, edge_color)
                break
            except Exception:
                continue
    try:
        layer.face_color = face_color
    except Exception:
        pass


def _update_or_add_points(
    viewer: "napari.Viewer",
    name: str,
    data: np.ndarray,
    *,
    scale: tuple[float, float],
    size: float,
    edge_color: str,
    face_color: str,
):
    """Add a Points layer named ``name`` or, if present, replace its data in place."""
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        layer.scale = scale
        layer.size = size
        _set_points_colors(layer, edge_color=edge_color, face_color=face_color)
        return layer
    layer = _add_points_compat(
        viewer,
        data=data,
        name=name,
        scale=scale,
        size=size,
        edge_color=edge_color,
        face_color=face_color,
    )
    return layer


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the nuclei-segmentation submenu, wired to ``state`` and ``viewer``."""
    section = CollapsibleSection("Nuclei segmentation")

    body = QWidget()
    layout = QVBoxLayout(body)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    prob_slider = NumericSlider("prob_thresh", 0.0, 1.0, step=0.01, value=0.5)
    nms_slider = NumericSlider("nms_thresh", 0.0, 1.0, step=0.01, value=0.4)
    run_button = QPushButton("Run StarDist")
    status_label = QLabel("Idle")
    status_label.setWordWrap(False)

    layout.addWidget(prob_slider)
    layout.addWidget(nms_slider)
    layout.addWidget(run_button)
    layout.addWidget(status_label)

    section.set_content(body)

    # Mutable container so nested callbacks can stash the active worker.
    worker_holder: dict[str, object] = {}

    def _set_status(text: str) -> None:
        # Truncate to a single line so long error messages don't blow up
        # the layout.
        text = text.replace("\r", " ").replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        status_label.setText(text)

    def _apply_to_viewer(result: NucleiResult) -> None:
        image = state.image
        if image is None:
            return
        scale = (image.pixel_size_um, image.pixel_size_um)
        _update_or_add_labels(
            viewer,
            _NUCLEI_MASK_LAYER,
            result.label_mask,
            scale=scale,
            opacity=0.3,
        )
        _update_or_add_points(
            viewer,
            _NUCLEI_POINTS_LAYER,
            result.centroids,
            scale=scale,
            size=5.0 * image.pixel_size_um,
            edge_color="white",
            face_color="transparent",
        )

    def _on_run_clicked() -> None:
        image = state.image
        if image is None:
            _set_status("No image loaded")
            return
        if image.dapi is None:
            _set_status("No DAPI channel — load a CYX TIFF or use --mode gfap")
            return

        prob_thresh = float(prob_slider.value())
        nms_thresh = float(nms_slider.value())
        params = {"prob_thresh": prob_thresh, "nms_thresh": nms_thresh}

        path = cache_path(image.source_path, "nuclei", params)

        # Fast path: cache hit. Do it synchronously — it's a single npz load.
        cached = load_npz(path)
        if cached is not None and "label_mask" in cached and "centroids" in cached:
            label_mask = np.asarray(cached["label_mask"], dtype=np.int32)
            centroids = np.asarray(cached["centroids"], dtype=float)
            result = NucleiResult(
                label_mask=label_mask, centroids=centroids, params=params
            )
            state.set("nuclei", result)
            _apply_to_viewer(result)
            _set_status(f"{len(centroids)} nuclei detected (cached)")
            return

        # Slow path: actually run StarDist on a worker thread so the GUI
        # stays responsive.
        run_button.setEnabled(False)
        n_tiles = _auto_n_tiles(image.dapi.shape)
        if n_tiles == (1, 1):
            _set_status("Running...")
        else:
            _set_status(f"Running... ({n_tiles[0]}x{n_tiles[1]} tiles)")

        from napari.qt import thread_worker

        dapi = image.dapi

        @thread_worker
        def _job():
            return _run_stardist(dapi, prob_thresh, nms_thresh)

        worker = _job()

        def _on_returned(value: tuple[np.ndarray, np.ndarray, tuple[int, int]]) -> None:
            try:
                label_mask, centroids, _n_tiles = value
                save_npz(path, label_mask=label_mask, centroids=centroids)
                result = NucleiResult(
                    label_mask=label_mask, centroids=centroids, params=params
                )
                state.set("nuclei", result)
                _apply_to_viewer(result)
                _set_status(f"{len(centroids)} nuclei detected")
            except Exception as exc:  # pragma: no cover — defensive
                _set_status(f"Error: {exc}")

        def _on_errored(exc: BaseException) -> None:
            _set_status(f"Error: {exc}")

        def _on_finished() -> None:
            run_button.setEnabled(True)
            worker_holder.pop("worker", None)

        worker.returned.connect(_on_returned)
        worker.errored.connect(_on_errored)
        worker.finished.connect(_on_finished)
        worker_holder["worker"] = worker
        worker.start()

    run_button.clicked.connect(_on_run_clicked)

    return section
