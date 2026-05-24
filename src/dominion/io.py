"""Image loading for Dominion.

Reads 2-channel CYX uint16 TIFFs (channel 0 = GFAP, channel 1 = DAPI) and
extracts the µm/pixel scale from the TIFF resolution metadata.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import tifffile

from .types import ImageData


_MICRON_TOKENS = {"um", "µm", "micron", "microns", "micrometer", "micrometers"}


def _parse_unit_from_description(description: str | None) -> str | None:
    """Pull `unit=...` out of an ImageJ-style ImageDescription string.

    ImageJ commonly writes the µ glyph as the literal six-character
    escape ``\\u00B5`` rather than the unicode character itself. We
    decode that on the way out.
    """
    if not description:
        return None
    for token in description.replace("\r", "\n").split("\n"):
        token = token.strip()
        if token.lower().startswith("unit="):
            value = token.split("=", 1)[1].strip()
            if "\\u" in value:
                try:
                    value = value.encode("ascii", "backslashreplace").decode(
                        "unicode_escape"
                    )
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
            return value
    return None


def load_image(path: Path) -> ImageData:
    """Load a CYX uint16 TIFF and return an :class:`ImageData`.

    The first two channels are interpreted as GFAP and DAPI respectively.
    Pixel size is parsed from the TIFF ``XResolution`` tag using the
    convention ``pixels_per_um = num/den`` (i.e. ImageJ writes resolution
    in pixels-per-unit). If the resolution unit is present and is not
    microns — or if ``XResolution`` is missing — the pixel size falls
    back to 1.0 and a warning is emitted.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        page = tf.pages[0]
        xres_tag = page.tags.get("XResolution", None)
        desc_tag = page.tags.get("ImageDescription", None)
        description = desc_tag.value if desc_tag is not None else None

    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(
            f"Expected CYX TIFF with at least 2 channels, got shape {arr.shape}"
        )

    gfap = arr[0]
    dapi = arr[1]

    # Resolve pixel size.
    pixel_size_um = 1.0
    unit = _parse_unit_from_description(description)
    if unit is not None and unit.lower() not in _MICRON_TOKENS:
        warnings.warn(
            f"TIFF resolution unit is {unit!r}, not microns; "
            "falling back to pixel_size_um=1.0."
        )
    elif xres_tag is None:
        warnings.warn(
            "TIFF has no XResolution tag; falling back to pixel_size_um=1.0."
        )
    else:
        value = xres_tag.value
        try:
            num, den = value
        except (TypeError, ValueError):
            num, den = value, 1
        if num == 0 or den == 0:
            warnings.warn(
                f"TIFF XResolution is degenerate ({value!r}); "
                "falling back to pixel_size_um=1.0."
            )
        else:
            pixels_per_um = float(num) / float(den)
            if pixels_per_um <= 0:
                warnings.warn(
                    f"TIFF XResolution is non-positive ({pixels_per_um}); "
                    "falling back to pixel_size_um=1.0."
                )
            else:
                pixel_size_um = 1.0 / pixels_per_um

    tissue_mask = (gfap > 0) | (dapi > 0)

    return ImageData(
        gfap=gfap,
        dapi=dapi,
        tissue_mask=tissue_mask,
        pixel_size_um=pixel_size_um,
        source_path=path,
    )
