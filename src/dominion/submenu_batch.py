"""Submenu — Batch / settings persistence.

Save and Load buttons that round-trip every other submenu's widget
values through a JSON file. The pipeline itself is NOT run by loading
settings — the user still clicks Run on each section. This is purely a
way to capture a tuned set of parameters and re-apply them on a new
image or a new session.

A directory-batch button is planned for a future commit; for now this
submenu just hosts Save / Load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .state import AppState
from .widgets.common import CollapsibleSection

if TYPE_CHECKING:
    import napari  # noqa: F401


_FORMAT_VERSION = 1


def build_widget(state: AppState, viewer: "napari.Viewer") -> QWidget:
    """Return the Batch / settings-persistence submenu."""
    section = CollapsibleSection("Batch", collapsed=True)

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    button_row = QWidget()
    row_layout = QHBoxLayout(button_row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    save_button = QPushButton("Save settings...")
    load_button = QPushButton("Load settings...")
    row_layout.addWidget(save_button)
    row_layout.addWidget(load_button)

    status_label = QLabel(
        "Save: writes every section's slider values to JSON.\n"
        "Load: reads JSON and applies it to every section."
    )
    status_label.setWordWrap(True)
    status_label.setAlignment(Qt.AlignLeft)

    layout.addWidget(button_row)
    layout.addWidget(status_label)
    section.set_content(content)

    def _default_filename() -> str:
        if state.image is not None:
            src = Path(state.image.source_path)
            return str(src.with_name(f"{src.stem}_settings.json"))
        return "dominion_settings.json"

    def _on_save_clicked() -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            content,
            "Save settings",
            _default_filename(),
            "JSON files (*.json);;All files (*)",
        )
        if not chosen:
            return
        path = Path(chosen)
        payload: dict = {
            "_meta": {
                "format_version": _FORMAT_VERSION,
                "source_image": (
                    str(state.image.source_path) if state.image is not None else None
                ),
            },
            "sections": state.get_all_settings(),
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            status_label.setText(f"Save failed: {exc}")
            return
        n_sections = len(payload["sections"])
        status_label.setText(f"Saved {n_sections} sections to {path.name}.")

    def _on_load_clicked() -> None:
        start_dir = (
            str(Path(state.image.source_path).parent)
            if state.image is not None
            else ""
        )
        chosen, _ = QFileDialog.getOpenFileName(
            content,
            "Load settings",
            start_dir,
            "JSON files (*.json);;All files (*)",
        )
        if not chosen:
            return
        path = Path(chosen)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status_label.setText(f"Load failed: {exc}")
            return
        # Tolerate older (flat) layouts as well as the standard nested form.
        sections = (
            payload.get("sections")
            if isinstance(payload, dict) and "sections" in payload
            else payload
        )
        if not isinstance(sections, dict):
            status_label.setText("Load failed: JSON has no settings sections.")
            return
        applied = state.apply_all_settings(sections)
        if not applied:
            status_label.setText(
                f"Loaded {path.name} but no sections matched. "
                "Settings file may be from a different mode."
            )
            return
        status_label.setText(
            f"Loaded {len(applied)} section(s) from {path.name}: "
            f"{', '.join(applied)}"
        )

    save_button.clicked.connect(_on_save_clicked)
    load_button.clicked.connect(_on_load_clicked)

    return section
