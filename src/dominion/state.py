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
    SholResult,
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
    "sholl": SholResult,
}

_DOWNSTREAM: dict[str, list[str]] = {
    "image": ["nuclei"],
    "nuclei": ["seeds"],
    "seeds": ["tessellation"],
    "tessellation": ["measurements", "skeletons"],
    "measurements": [],
    "skeletons": ["sholl"],
    "sholl": [],
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
        self.sholl: Optional[SholResult] = None
        self._subscribers: dict[str, list[Callable[[], None]]] = {
            slot: [] for slot in _SLOT_TYPES
        }
        # Settings registry: each submenu can register a (get, apply) pair
        # under a section name so the Batch submenu can round-trip all
        # widget values through a JSON file.
        self._settings_providers: dict[
            str, tuple[Callable[[], dict], Callable[[dict], None]]
        ] = {}

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

    # --- Settings registry --------------------------------------------------

    def register_settings(
        self,
        section: str,
        get_fn: Callable[[], dict],
        apply_fn: Callable[[dict], None],
    ) -> None:
        """Register a (get, apply) pair for the given section name.

        ``get_fn()`` should return a JSON-serialisable dict of the section's
        current widget values. ``apply_fn(d)`` should set widgets from a
        previously-saved dict, tolerating missing keys gracefully.
        """
        self._settings_providers[section] = (get_fn, apply_fn)

    def get_all_settings(self) -> dict:
        """Return a nested dict of every registered section's current settings."""
        return {name: get() for name, (get, _apply) in self._settings_providers.items()}

    def apply_all_settings(self, settings: dict) -> list[str]:
        """Apply a previously-saved settings dict. Returns the list of
        section names that were applied (skipped sections aren't in
        the dict or aren't registered)."""
        applied: list[str] = []
        for name, (_get, apply) in self._settings_providers.items():
            section = settings.get(name)
            if isinstance(section, dict):
                apply(section)
                applied.append(name)
        return applied
