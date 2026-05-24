"""Shared Qt widgets used across the Dominion submenus.

These are intentionally napari-agnostic and can be exercised in a plain
``QApplication`` for tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleSection(QWidget):
    """A titled section whose body can be hidden by clicking the header."""

    def __init__(self, title: str, parent: Optional[QWidget] = None, *, collapsed: bool = False) -> None:
        super().__init__(parent)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self._toggle.setStyleSheet(
            "QToolButton { font-weight: bold; border: none; padding: 4px; text-align: left; }"
        )
        self._toggle.clicked.connect(self._on_toggle)

        self._content_frame = QFrame(self)
        self._content_frame.setFrameShape(QFrame.StyledPanel)
        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(4)
        self._content_widget: Optional[QWidget] = None
        self._content_frame.setVisible(not collapsed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content_frame)

    def set_content(self, widget: QWidget) -> None:
        """Replace the inner widget."""
        if self._content_widget is not None:
            self._content_layout.removeWidget(self._content_widget)
            self._content_widget.setParent(None)
            self._content_widget.deleteLater()
        self._content_widget = widget
        self._content_layout.addWidget(widget)

    def content_layout(self) -> QVBoxLayout:
        """Direct access to the inner layout for downstream widgets."""
        return self._content_layout

    def _on_toggle(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)


class NumericSlider(QWidget):
    """A labeled slider+spinbox combo that emits float ``valueChanged``.

    The underlying :class:`QSlider` works in integer steps; the public API
    is float-valued and quantised by ``step``.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        *,
        decimals: int = 2,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        if maximum < minimum:
            raise ValueError(f"maximum ({maximum}) < minimum ({minimum})")

        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._syncing = False

        self._label = QLabel(label, self)
        self._label.setMinimumWidth(80)

        self._slider = QSlider(Qt.Horizontal, self)
        self._spin = QDoubleSpinBox(self)
        self._spin.setDecimals(self._decimals)

        self._configure_range()
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)

        self.set_value(float(value))

    # -- public API ----------------------------------------------------

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, value: float) -> None:
        value = float(np.clip(value, self._min, self._max))
        self._syncing = True
        try:
            self._spin.setValue(value)
            self._slider.setValue(self._to_int(value))
        finally:
            self._syncing = False
        self.valueChanged.emit(value)

    def set_range(self, minimum: float, maximum: float, step: Optional[float] = None) -> None:
        """Update the slider/spinbox range (and optional step), preserving value if possible."""
        if maximum < minimum:
            raise ValueError(f"maximum ({maximum}) < minimum ({minimum})")
        current = self.value()
        self._min = float(minimum)
        self._max = float(maximum)
        if step is not None:
            if step <= 0:
                raise ValueError(f"step must be positive, got {step}")
            self._step = float(step)
        self._configure_range()
        self.set_value(float(np.clip(current, self._min, self._max)))

    # -- internals -----------------------------------------------------

    def _configure_range(self) -> None:
        n_steps = max(1, int(round((self._max - self._min) / self._step)))
        self._syncing = True
        try:
            self._slider.setRange(0, n_steps)
            self._slider.setSingleStep(1)
            self._spin.setRange(self._min, self._max)
            self._spin.setSingleStep(self._step)
        finally:
            self._syncing = False

    def _to_int(self, value: float) -> int:
        return int(round((value - self._min) / self._step))

    def _from_int(self, ivalue: int) -> float:
        return self._min + ivalue * self._step

    def _on_slider(self, ivalue: int) -> None:
        if self._syncing:
            return
        value = self._from_int(ivalue)
        self._syncing = True
        try:
            self._spin.setValue(value)
        finally:
            self._syncing = False
        self.valueChanged.emit(value)

    def _on_spin(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._slider.setValue(self._to_int(value))
        finally:
            self._syncing = False
        self.valueChanged.emit(value)


class HistogramSlider(QWidget):
    """A :class:`NumericSlider` paired with a histogram plot of supporting data.

    Useful for picking a threshold against the distribution of some
    per-object score. The vertical cutoff line on the histogram mirrors
    the slider value.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        *,
        decimals: int = 2,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._slider = NumericSlider(
            label, minimum, maximum, step, value, decimals=decimals, parent=self
        )
        self._plot = pg.PlotWidget(parent=self)
        self._plot.setMinimumHeight(80)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setMenuEnabled(False)
        self._plot.getPlotItem().showAxis("left", False)
        self._hist_item: Optional[pg.PlotDataItem] = None
        self._cutoff_line = pg.InfiniteLine(
            pos=value, angle=90, movable=False, pen=pg.mkPen("y", width=2)
        )
        self._plot.addItem(self._cutoff_line)

        self._slider.valueChanged.connect(self._on_slider_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider)
        layout.addWidget(self._plot)

    # -- public API ----------------------------------------------------

    def value(self) -> float:
        return self._slider.value()

    def set_value(self, value: float) -> None:
        self._slider.set_value(value)

    def set_range(self, minimum: float, maximum: float, step: Optional[float] = None) -> None:
        self._slider.set_range(minimum, maximum, step)

    def set_data(self, values: np.ndarray, bins: int = 100) -> None:
        """Refresh the histogram from the given 1-D array."""
        values = np.asarray(values).ravel()
        values = values[np.isfinite(values)]
        if self._hist_item is not None:
            self._plot.removeItem(self._hist_item)
            self._hist_item = None
        if values.size == 0:
            return
        counts, edges = np.histogram(values, bins=bins)
        self._hist_item = self._plot.plot(
            edges,
            counts,
            stepMode="center",
            fillLevel=0,
            brush=(120, 120, 200, 150),
            pen=pg.mkPen((80, 80, 160), width=1),
        )
        # Keep the cutoff line on top of the histogram.
        self._plot.removeItem(self._cutoff_line)
        self._plot.addItem(self._cutoff_line)

    # -- internals -----------------------------------------------------

    def _on_slider_changed(self, value: float) -> None:
        self._cutoff_line.setPos(value)
        self.valueChanged.emit(value)
