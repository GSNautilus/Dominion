"""Runtime feature detection for optional ("Full") dependencies.

DOMINION ships in two flavours: a **Basic** install (core deps only, the
default signal-mode pipeline) and a **Full** install that adds the
``nuclei`` extra — StarDist + TensorFlow — for the nuclei-guided mode.

These helpers let the UI and CLI check which flavour is installed *without*
importing TensorFlow (which is slow and allocates GPU memory). We only ask
whether the modules can be found on the import path.
"""

from __future__ import annotations

import importlib.util

#: Modules that the ``nuclei`` extra provides. All must be importable for
#: nuclei mode to work.
_NUCLEI_MODULES = ("stardist", "csbdeep", "tensorflow")

#: Shown to the user when they reach nuclei mode on a Basic install.
NUCLEI_UNAVAILABLE_MESSAGE = (
    "Nuclei detection (StarDist) isn't included in this Basic install.\n\n"
    "It needs the Full version, which adds TensorFlow. To add it, run the "
    "Full installer, or in the DOMINION environment run:\n\n"
    '    pip install "dominion[nuclei]"\n\n'
    "The default (signal) mode works without it."
)


def has_nuclei_support() -> bool:
    """Return ``True`` if the StarDist/TensorFlow stack is importable.

    Uses :func:`importlib.util.find_spec`, so it does not actually import
    TensorFlow — it just checks that the packages are present.
    """
    return all(importlib.util.find_spec(name) is not None for name in _NUCLEI_MODULES)
