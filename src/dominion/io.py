"""Image loading for DOMINION.

Reads 2D single-channel TIFFs (treated as the signal channel),
2-channel CYX TIFFs (channel 0 = signal, channel 1 = nuclei), or
N-channel CYX TIFFs where channels beyond the first two are kept as
``extra_channels`` for per-domain measurement. Pixel size is parsed
from the TIFF resolution metadata.
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

    Accepts:

    * 2D single-channel — treated as the signal channel; nuclei is ``None``.
    * 3D CYX with 1 channel — same as 2D single-channel.
    * 3D CYX with 2 channels — channel 0 = signal, channel 1 = nuclei.
    * 3D CYX with N channels (N > 2) — channels 0 and 1 as above; channels
      2..N-1 become ``extra_channels`` keyed as ``"channel_2"`` ... ``"channel_<N-1>"``
      and are available to the Measurements submenu.

    ``--mode signal`` works on any of the above. ``--mode nuclei`` requires
    at least 2 channels.

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

    all_channels: dict[str, np.ndarray] = {}
    if arr.ndim == 2:
        all_channels["channel_0"] = arr
    elif arr.ndim == 3 and arr.shape[0] >= 1:
        for ch_idx in range(arr.shape[0]):
            all_channels[f"channel_{ch_idx}"] = arr[ch_idx]
    else:
        raise ValueError(
            f"Expected a 2D single-channel TIFF or a 3D CYX TIFF, "
            f"got shape {arr.shape}"
        )

    # Default assignment: channel_0 = signal, channel_1 = nuclei (if present).
    signal_name = "channel_0"
    nuclei_name = "channel_1" if "channel_1" in all_channels else None
    signal = all_channels[signal_name]
    nuclei = all_channels[nuclei_name] if nuclei_name is not None else None
    extra_channels = {
        n: v for n, v in all_channels.items()
        if n != signal_name and (nuclei_name is None or n != nuclei_name)
    }

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
        extra_channels=extra_channels,
        all_channels=all_channels,
        signal_channel_name=signal_name,
        nuclei_channel_name=nuclei_name,
    )
