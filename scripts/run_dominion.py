"""CLI entry point for the DOMINION napari workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import napari

from dominion.app import MODES, build_dock_widget
from dominion.io import load_image
from dominion.state import AppState


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the DOMINION napari workflow.")
    parser.add_argument(
        "image_path",
        type=Path,
        help=(
            "Path to a 2D single-channel TIFF (treated as the signal channel) "
            "or a 2-channel CYX TIFF (c0=signal, c1=nuclei)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="signal",
        help=(
            "Pipeline variant: 'signal' (default) finds object seeds directly "
            "from the signal channel; 'nuclei' uses StarDist on the nuclei "
            "channel + signal-based classification (requires a CYX TIFF with "
            "both channels)."
        ),
    )
    args = parser.parse_args(argv)

    image = load_image(args.image_path)

    state = AppState()
    state.set("image", image)

    viewer = napari.Viewer()

    # The Channels submenu owns the Signal / Nuclei / extra image layers —
    # they get created (or refreshed) inside its image-change subscription,
    # so a channel swap in the UI keeps the layers in sync.
    dock = build_dock_widget(state, viewer, mode=args.mode)
    viewer.window.add_dock_widget(
        dock, name=f"DOMINION ({args.mode})", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
