"""CLI entry point for the Dominion napari workflow."""

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
    parser = argparse.ArgumentParser(description="Launch the Dominion napari workflow.")
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to a CYX uint16 TIFF (c1=GFAP, c2=DAPI).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="gfap",
        help=(
            "Pipeline variant: 'gfap' (default) uses the GFAP-only seed-finding "
            "+ tessellation flow; 'dapi' uses the StarDist+classification + "
            "tessellation flow (requires a CYX TIFF with a DAPI channel)."
        ),
    )
    args = parser.parse_args(argv)

    image = load_image(args.image_path)

    state = AppState()
    state.set("image", image)

    viewer = napari.Viewer()

    scale = (image.pixel_size_um, image.pixel_size_um)

    viewer.add_image(
        image.gfap,
        name="GFAP",
        colormap="green",
        blending="additive",
        contrast_limits=_auto_contrast_limits(image.gfap),
        scale=scale,
    )
    if image.dapi is not None:
        viewer.add_image(
            image.dapi,
            name="DAPI",
            colormap="blue",
            blending="additive",
            contrast_limits=_auto_contrast_limits(image.dapi),
            scale=scale,
        )

    dock = build_dock_widget(state, viewer, mode=args.mode)
    viewer.window.add_dock_widget(
        dock, name=f"Dominion ({args.mode})", area="right"
    )

    napari.run()


if __name__ == "__main__":
    main()
