"""Mutable application state shared across submenus.

Holds pipeline slots and notifies subscribers when slots change. The
dependency graph (NOT a flat chain anymore) is:

    image -> nuclei -> seeds -> tessellation -> measurements
                                             \\
                                              -> skeletons -> sholl

Setting any slot clears every transitively-downstream slot (BFS over
``_DOWNSTREAM``) and notifies their subscribers. Sibling slots are
independent — running Measurements does not invalidate Skeletons and
vice versa.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from .types import (
    ImageData,
    MeasurementsResult,
    NucleiResult,
    SeedsResult,
    SkeletonsResult,
    TessellationResult,
)


_SLOT_TYPES = {
    "image": ImageData,
    "nuclei": NucleiResult,
    "seeds": SeedsResult,
    "tessellation": TessellationResult,
    "measurements": MeasurementsResult,
    "skeletons": SkeletonsResult,
}

_DOWNSTREAM: dict[str, list[str]] = {
    "image": ["nuclei"],
    "nuclei": ["seeds"],
    "seeds": ["tessellation"],
    "tessellation": ["measurements", "skeletons"],
    "measurements": [],
    "skeletons": [],
}


def _transitively_downstream(slot: str) -> list[str]:
    """Return all slots strictly downstream of ``slot``, in BFS order."""
    out: list[str] = []
    seen: set[str] = set()
    queue: deque[str] = deque(_DOWNSTREAM.get(slot, []))
    while queue:
        s = queue.popleft()
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        queue.extend(_DOWNSTREAM.get(s, []))
    return out


class AppState:
    """Container for pipeline slots with change subscriptions."""

    def __init__(self) -> None:
        self.image: Optional[ImageData] = None
        self.nuclei: Optional[NucleiResult] = None
        self.seeds: Optional[SeedsResult] = None
        self.tessellation: Optional[TessellationResult] = None
        self.measurements: Optional[MeasurementsResult] = None
        self.skeletons: Optional[SkeletonsResult] = None
        self._subscribers: dict[str, list[Callable[[], None]]] = {
            slot: [] for slot in _SLOT_TYPES
        }

    def subscribe(self, slot_name: str, callback: Callable[[], None]) -> None:
        """Register ``callback`` to be invoked whenever ``slot_name`` changes."""
        if slot_name not in self._subscribers:
            raise KeyError(
                f"Unknown slot {slot_name!r}; expected one of {tuple(_SLOT_TYPES)}"
            )
        self._subscribers[slot_name].append(callback)

    def set(self, slot_name: str, value) -> None:
        """Set ``slot_name`` to ``value``, clear transitively-downstream
        slots, then notify every changed slot's subscribers."""
        if slot_name not in _SLOT_TYPES:
            raise KeyError(
                f"Unknown slot {slot_name!r}; expected one of {tuple(_SLOT_TYPES)}"
            )
        if value is not None and not isinstance(value, _SLOT_TYPES[slot_name]):
            raise TypeError(
                f"Slot {slot_name!r} expects {_SLOT_TYPES[slot_name].__name__}, "
                f"got {type(value).__name__}"
            )

        changed: list[str] = [slot_name]
        setattr(self, slot_name, value)

        for downstream in _transitively_downstream(slot_name):
            if getattr(self, downstream) is not None:
                setattr(self, downstream, None)
                changed.append(downstream)

        for slot in changed:
            for cb in list(self._subscribers[slot]):
                cb()
