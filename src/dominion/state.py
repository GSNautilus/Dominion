"""Mutable application state shared across submenus.

Holds the pipeline slots (``image``, ``nuclei``, ``seeds``,
``tessellation``, ``measurements``) and notifies subscribers when slots
change. Setting a slot clears every downstream slot in the fixed chain
``image -> nuclei -> seeds -> tessellation -> measurements``.
"""

from __future__ import annotations

from typing import Callable, Optional

from .types import (
    ImageData,
    MeasurementsResult,
    NucleiResult,
    SeedsResult,
    TessellationResult,
)


_SLOT_ORDER = ("image", "nuclei", "seeds", "tessellation", "measurements")
_SLOT_TYPES = {
    "image": ImageData,
    "nuclei": NucleiResult,
    "seeds": SeedsResult,
    "tessellation": TessellationResult,
    "measurements": MeasurementsResult,
}


class AppState:
    """Container for the pipeline slots with change subscriptions."""

    def __init__(self) -> None:
        self.image: Optional[ImageData] = None
        self.nuclei: Optional[NucleiResult] = None
        self.seeds: Optional[SeedsResult] = None
        self.tessellation: Optional[TessellationResult] = None
        self.measurements: Optional[MeasurementsResult] = None
        self._subscribers: dict[str, list[Callable[[], None]]] = {
            slot: [] for slot in _SLOT_ORDER
        }

    def subscribe(self, slot_name: str, callback: Callable[[], None]) -> None:
        """Register ``callback`` to be invoked whenever ``slot_name`` changes."""
        if slot_name not in self._subscribers:
            raise KeyError(f"Unknown slot {slot_name!r}; expected one of {_SLOT_ORDER}")
        self._subscribers[slot_name].append(callback)

    def set(self, slot_name: str, value) -> None:
        """Set ``slot_name`` to ``value``, clear downstream slots, notify."""
        if slot_name not in _SLOT_TYPES:
            raise KeyError(f"Unknown slot {slot_name!r}; expected one of {_SLOT_ORDER}")
        if value is not None and not isinstance(value, _SLOT_TYPES[slot_name]):
            raise TypeError(
                f"Slot {slot_name!r} expects {_SLOT_TYPES[slot_name].__name__}, "
                f"got {type(value).__name__}"
            )

        idx = _SLOT_ORDER.index(slot_name)
        changed: list[str] = [slot_name]
        setattr(self, slot_name, value)

        # Clear downstream slots — but only ones that actually had something,
        # so we don't fire spurious notifications for empty downstreams.
        for downstream in _SLOT_ORDER[idx + 1 :]:
            if getattr(self, downstream) is not None:
                setattr(self, downstream, None)
                changed.append(downstream)

        for slot in changed:
            for cb in list(self._subscribers[slot]):
                cb()
