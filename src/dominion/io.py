"""Image loading for DOMINION.

Reads 2D single-channel TIFFs (treated as the signal channel) or
2-channel CYX TIFFs (channel 0 = signal, channel 1 = nuclei) and
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
    """Load a TIFF and return an :class:`ImageData`.

    Accepts either a 2D single-channel image (treated as the signal
    channel, with nuclei set to ``None``) or a 3D CYX stack where
    channel 0 is the signal channel and channel 1 is the nuclei channel.
    Single-channel input works with ``--mode signal``; ``--mode nuclei``
    requires both channels.

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

    if arr.ndim == 2:
        signal = arr
        nuclei = None
    elif arr.ndim == 3 and arr.shape[0] >= 1:
        signal = arr[0]
        nuclei = arr[1] if arr.shape[0] >= 2 else None
    else:
        raise ValueError(
            f"Expected a 2D single-channel TIFF or a 3D CYX TIFF, "
            f"got shape {arr.shape}"
        )

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

    tissue_mask = (
        (signal > 0) if nuclei is None else ((signal > 0) | (nuclei > 0))
    )

    return ImageData(
        signal=signal,
        nuclei=nuclei,
        tissue_mask=tissue_mask,
        pixel_size_um=pixel_size_um,
        source_path=path,
    )
