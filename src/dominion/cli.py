"""Command-line / launcher entry point for the DOMINION napari workflow.

This is the only module that touches ``sys.argv`` and the filesystem for
launch purposes; :mod:`dominion.app` stays pure so a future napari plugin
can reuse :func:`dominion.app.build_dock_widget` directly.

Launching with no image path pops a file-open dialog, so the app can be
started from a double-click shortcut without typing a path. If nuclei mode
is requested on a Basic install (no StarDist/TensorFlow), a friendly
message explains how to upgrade instead of crashing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .app import MODES, build_dock_widget
from .capabilities import NUCLEI_UNAVAILABLE_MESSAGE, has_nuclei_support


def _ensure_qapp():
    """Return a running QApplication, creating one if needed.

    A QApplication must exist before any Qt dialog can be shown. napari
    reuses whatever instance already exists, so creating one here is safe.
    """
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _prompt_for_image() -> Optional[Path]:
    """Show an "Open image" dialog and return the chosen path (or ``None``)."""
    from qtpy.QtWidgets import QFileDialog

    _ensure_qapp()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Open an image for DOMINION",
        "",
        "TIFF images (*.tif *.tiff);;All files (*)",
    )
    return Path(path) if path else None


def _warn_nuclei_unavailable() -> None:
    """Tell the user nuclei mode needs the Full install (GUI box + console)."""
    try:
        from qtpy.QtWidgets import QMessageBox

        _ensure_qapp()
        QMessageBox.information(
            None, "DOMINION — Full version required", NUCLEI_UNAVAILABLE_MESSAGE
        )
    except Exception:
        # Headless / no display: fall back to the console.
        print(NUCLEI_UNAVAILABLE_MESSAGE)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the DOMINION napari workflow.")
    parser.add_argument(
        "image_path",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to a 2D single-channel TIFF (treated as the signal channel) "
            "or a 2-channel CYX TIFF (c0=signal, c1=nuclei). If omitted, a "
            "file-open dialog is shown."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="signal",
        help=(
            "Pipeline variant: 'signal' (default) finds object seeds directly "
            "from the signal channel; 'nuclei' uses StarDist on the nuclei "
            "channel + signal-based classification (requires the Full install "
            "and a CYX TIFF with both channels)."
        ),
    )
    args = parser.parse_args(argv)

    # Gate nuclei mode on the Full install before we load anything heavy.
    if args.mode == "nuclei" and not has_nuclei_support():
        _warn_nuclei_unavailable()
        return

    image_path = args.image_path
    if image_path is None:
        image_path = _prompt_for_image()
        if image_path is None:
            return  # user cancelled the dialog

    # Deferred so --help / the mode gate don't pay napari's import cost.
    import napari

    from .io import load_image
    from .state import AppState

    image = load_image(image_path)

    state = AppState()
    state.set("image", image)

    viewer = napari.Viewer()

    # The Channels submenu owns the Signal / Nuclei / extra-channel image
    # layers — they get created (or refreshed) inside its image-change
    # subscription, so a channel swap in the UI keeps the layers in sync.
    dock = build_dock_widget(state, viewer, mode=args.mode)
    viewer.window.add_dock_widget(dock, name=f"DOMINION ({args.mode})", area="right")

    napari.run()


if __name__ == "__main__":
    main()
