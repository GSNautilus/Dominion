"""CLI entry point for the DOMINION napari workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import napari
import numpy as np

from dominion.app import MODES, build_dock_widget
from dominion.io import load_image
from dominion.state import AppState


def _auto_contrast_limits(arr: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(arr, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


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

    scale = (image.pixel_size_um, image.pixel_size_um)

    viewer.add_image(
        image.signal,
        name="Signal",
        colormap="green",
        blending="additive",
        contrast_limits=_auto_contrast_limits(image.signal),
        scale=scale,
    )
    if image.nuclei is not None:
        viewer.add_image(
            image.nuclei,
            name="Nuclei",
            colormap="blue",
            blending="additive",
            contrast_limits=_auto_contrast_limits(image.nuclei),
            scale=scale,
        )
    # Extra channels (beyond signal + nuclei) are cycled through a small
    # palette so they're visually distinguishable in the viewer.
    _extra_colormaps = ("magenta", "yellow", "cyan", "red", "gray")
    for i, (name, arr) in enumerate(image.extra_channels.items()):
        viewer.add_image(
            arr,
            name=name,
            colormap=_extra_colormaps[i % len(_extra_colormaps)],
            blending="additive",
            contrast_limits=_auto_contrast_limits(arr),
            scale=scale,
            visible=False,  # off by default to avoid clutter
        )

    dock = build_dock_widget(state, viewer, mode=args.mode)
    viewer.window.add_dock_widget(
        dock, name=f"DOMINION ({args.mode})", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
