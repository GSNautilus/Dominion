"""Dock widget composition for the Dominion napari workflow.

The :func:`build_dock_widget` function is the future-plugin entry point:
it consumes an :class:`AppState` and a napari ``Viewer`` and returns the
fully-assembled dock widget. It must not touch the filesystem or
``sys.argv``; all I/O lives in ``scripts/run_dominion.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QWidget

from . import (
    submenu1_nuclei,
    submenu2_seeds,
    submenu3_tessellation,
    submenu_a_gfap_seeds,
)
from .state import AppState

if TYPE_CHECKING:
    import napari  # noqa: F401


MODES = ("dapi", "gfap")


def build_dock_widget(
    state: AppState, viewer: "napari.Viewer", mode: str = "dapi"
) -> QWidget:
    """Return the dock widget for the chosen pipeline ``mode``.

    Two modes are supported:

    * ``"dapi"`` (default): the three-stage DAPI+GFAP pipeline —
      StarDist nuclei → astrocyte classification → tessellation.
    * ``"gfap"``: the two-stage GFAP-only pipeline —
      GFAP seed-finding → tessellation.

    Plugin-portable: a future napari plugin's dock-widget contribution
    should call this same function with its own ``state``, ``viewer``,
    and ``mode``.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(6)

    if mode == "dapi":
        layout.addWidget(submenu1_nuclei.build_widget(state, viewer))
        layout.addWidget(submenu2_seeds.build_widget(state, viewer))
    else:  # mode == "gfap"
        layout.addWidget(submenu_a_gfap_seeds.build_widget(state, viewer))
    layout.addWidget(submenu3_tessellation.build_widget(state, viewer))
    layout.addStretch(1)

    return container
