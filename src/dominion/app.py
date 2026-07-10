"""Dock widget composition for the DOMINION napari workflow.

The :func:`build_dock_widget` function is the future-plugin entry point:
it consumes an :class:`AppState` and a napari ``Viewer`` and returns the
fully-assembled dock widget. It must not touch the filesystem or
``sys.argv``; all I/O lives in ``scripts/run_dominion.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from . import (
    submenu1_nuclei,
    submenu2_seeds,
    submenu3_tessellation,
    submenu_a_signal_seeds,
    submenu_batch,
    submenu_channels,
    submenu_measurements,
    submenu_roi,
    submenu_skeletons,
    submenu_sholl,
)
from .state import AppState

if TYPE_CHECKING:
    import napari  # noqa: F401


MODES = ("signal", "nuclei")


def _build_header() -> QWidget:
    """A small banner at the top of the dock: name + acronym expansion."""
    header = QFrame()
    layout = QVBoxLayout(header)
    layout.setContentsMargins(6, 4, 6, 6)
    layout.setSpacing(2)

    title = QLabel("DOMINION")
    title.setStyleSheet("font-size: 14pt; font-weight: bold;")
    title.setAlignment(Qt.AlignLeft)

    subtitle = QLabel(
        "<b>DOM</b>ain <b>I</b>dentification for <b>N</b>etworks of "
        "<b>I</b>mmunolabeled <b>O</b>bject <b>N</b>eighborhoods"
    )
    subtitle.setTextFormat(Qt.RichText)
    subtitle.setWordWrap(True)
    subtitle.setStyleSheet("font-size: 9pt; color: palette(mid);")

    layout.addWidget(title)
    layout.addWidget(subtitle)
    return header


def build_dock_widget(
    state: AppState, viewer: "napari.Viewer", mode: str = "signal"
) -> QWidget:
    """Return the dock widget for the chosen pipeline ``mode``.

    Two modes are supported:

    * ``"signal"`` (default): the two-stage signal-only pipeline —
      signal seed-finding → tessellation. Works on a single-channel
      image (the signal channel of interest) or the signal channel of
      a 2-channel image.
    * ``"nuclei"``: the three-stage nuclei-guided pipeline —
      StarDist nuclei segmentation → object classification by signal →
      tessellation. Requires both a nuclei channel and a signal channel.

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

    layout.addWidget(_build_header())
    layout.addWidget(submenu_channels.build_widget(state, viewer))
    layout.addWidget(submenu_roi.build_widget(state, viewer))

    if mode == "nuclei":
        layout.addWidget(submenu1_nuclei.build_widget(state, viewer))
        layout.addWidget(submenu2_seeds.build_widget(state, viewer))
    else:  # mode == "signal"
        layout.addWidget(submenu_a_signal_seeds.build_widget(state, viewer))
    layout.addWidget(submenu3_tessellation.build_widget(state, viewer))
    layout.addWidget(submenu_skeletons.build_widget(state, viewer))
    layout.addWidget(submenu_sholl.build_widget(state, viewer))
    layout.addWidget(submenu_measurements.build_widget(state, viewer))
    layout.addWidget(submenu_batch.build_widget(state, viewer))
    layout.addStretch(1)

    return container
