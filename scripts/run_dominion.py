"""CLI entry point for the DOMINION napari workflow.

Thin wrapper kept for backward compatibility (``python
scripts/run_dominion.py ...``). The real logic lives in
:func:`dominion.cli.main`, which is also exposed as the ``dominion``
console command.
"""

from __future__ import annotations

from dominion.cli import main

if __name__ == "__main__":
    main()
