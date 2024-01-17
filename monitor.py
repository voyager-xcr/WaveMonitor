from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, Qt, Slot
from PySide6.QtGui import QMouseEvent, QShortcut
from PySide6.QtWidgets import QCheckBox, QMenu, QWidgetAction


class Monitor:
    """Keep a pyqtgraph plotWidget and plot waveforms on it."""

    def __init__(self):
        super().__init__()
        self.lines: dict[str, Line] = {}

        plot_widget = pg.plot(title='Wave Monitor - "F" zoom fit "C" clear all')

        plot_item = plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True)
        plot_item.getViewBox().disableAutoRange()

        # Make it hold millions of points.
        plot_item.setDownsampling(auto=True, mode="subsample")
        plot_item.setClipToView(True)
        # ClipToView disables plot_item.autoRange, as well as "View all" in right-click menu.
        QShortcut("F", plot_widget).activated.connect(self.autoscale)
        QShortcut("C", plot_widget).activated.connect(self.clear)

        # Replace the context menu with our own.
        self._filter = RightClickFilter(self)
        plot_item.getViewBox().setMenuEnabled(False)
        # viewport gets the mouseReleaseEvent, See https://blog.csdn.net/theoryll/article/details/110918779
        plot_widget.viewport().installEventFilter(self._filter)

        self.plot_widget = plot_widget
        self.plot_item = plot_item
        self.line_seperation = 2

    @Slot(str, np.ndarray, list, float)
    def add_line(self, name: str, t: np.ndarray, ys: list[np.ndarray]):
        if name in self.lines:
            line = self.lines[name]
            line.update_wfm(t, ys)
        else:
            offset = self.line_seperation * len(self.lines)
            self.lines[name] = Line(name, t, ys, offset, self.plot_item)

    @Slot(str)
    def remove_line(self, name: str):
        if name in self.lines:
            self.lines[name].remove()
            del self.lines[name]

    @Slot()
    def clear(self):
        for name in list(self.lines.keys()):
            self.remove_line(name)

    @Slot()
    def autoscale(self):
        if self.lines:
            t0 = min(line.t0 for line in self.lines.values())
            t1 = max(line.t1 for line in self.lines.values())
            y0 = min(line.offset for line in self.lines.values()) - 1
            y1 = max(line.offset for line in self.lines.values()) + 1
            self.plot_item.setRange(xRange=(t0, t1), yRange=(y0, y1))

    def show_context_menu(self, pos):
        context_menu = QMenu(self.plot_widget)

        for line_name, line in self.lines.items():
            checkbox = QCheckBox(line_name)
            checkbox.setChecked(line.is_visible())
            checkbox.toggled.connect(line.set_visible)

            action = QWidgetAction(self.plot_widget)
            action.setDefaultWidget(checkbox)
            context_menu.addAction(action)

        context_menu.exec(self.plot_widget.mapToGlobal(pos.toPoint()))


class Line:
    """Container for a waveform."""

    colors = (
        # # Simple RBG
        # (255, 0, 0, 50),
        # (0, 0, 255, 50),
        # (0, 255, 0, 50),
        # "dark_background" in https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        (110, 177, 166, 50),
        (218, 219, 146, 50),
        (158, 154, 183, 50),
        (214, 98, 86, 50),
        (98, 144, 176, 50),
        (217, 147, 69, 50),
        (146, 188, 75, 50),
        (155, 99, 156, 50),
        (170, 200, 163, 50),
        (219, 202, 81, 50),
    )

    def __init__(
        self,
        name: str,
        t: np.ndarray,
        ys: list[np.ndarray],
        offset: float,
        plot_item: pg.PlotItem,
    ):
        self.plot_item = plot_item
        self.offset = offset
        self.t0 = t[0]
        self.t1 = t[-1]
        self.lines = [
            plot_item.plot(
                t, y + offset, pen=color[:-1], fillLevel=offset, fillBrush=color
            )
            for y, color in zip(ys, self.colors)
        ]
        text = pg.TextItem(text=name, anchor=(1, 0.5))
        plot_item.addItem(text)
        self.text = text
        plot_item.sigXRangeChanged.connect(self.update_label_pos)

    def update_wfm(self, t: np.ndarray, ys: list[np.ndarray]):
        offset = self.offset
        self.t0 = t[0]
        self.t1 = t[-1]

        # Update existing lines with new data.
        old_lines = self.lines
        new_lines = []
        for line, y in zip(self.lines, ys):
            line.setData(t, y + offset)
            new_lines.append(line)

        # Remove unused lines.
        if len(ys) < len(old_lines):
            for line in old_lines[len(ys) :]:
                self.plot_item.removeItem(line)

        # Add more lines if needed.
        if len(ys) > len(old_lines):
            for y, color in zip(ys[len(old_lines) :], self.colors[len(old_lines) :]):
                line = self.plot_item.plot(
                    t, y + offset, pen=color[:-1], fillLevel=offset, fillBrush=color
                )
                new_lines.append(line)

        self.lines = new_lines

    def update_offset(self, offset: float):
        self.offset = offset
        for line in self.lines:
            line.setData(y=line.yData + offset)
        self.update_label_pos()

    def remove(self):
        for line in self.lines:
            self.plot_item.removeItem(line)
        self.plot_item.removeItem(self.text)

    def update_label_pos(self):
        viewbox = self.plot_item.getViewBox()
        (x0, x1), (y0, y1) = viewbox.viewRange()
        if x1 <= self.t0:
            pos = self.t0
        elif x1 <= self.t1:
            pos = x1
        else:
            pos = self.t1
        self.text.setPos(pos, self.offset)

    def set_visible(self, visible: bool):
        for line in self.lines:
            line.setVisible(visible)
        self.text.setVisible(visible)

    @property
    def is_visible(self) -> Callable[[], bool]:
        return self.text.isVisible


class RightClickFilter(QObject):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.mouse_press_pos = None

    def eventFilter(self, watched, event: QMouseEvent):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:
                self.mouse_press_pos = event.position()
        if event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.RightButton:
                if self.mouse_press_pos is not None:
                    if (event.position() - self.mouse_press_pos).manhattanLength() < 5:
                        # If the mouse moved more than 10 pixels, then it's not a right-click.
                        self.monitor.show_context_menu(event.position())
        return super().eventFilter(watched, event)


if __name__ == "__main__":
    t = np.linspace(0, 1, 1_00_001)  # 1m pts ~= 1ms for 1GSa/s.
    n = 20  # Okay with 20.
    i_waves = [np.cos(2 * np.pi * f * t) for f in range(1, n + 1)]
    q_waves = [np.sin(2 * np.pi * f * t) for f in range(1, n + 1)]
    
    monitor = Monitor()
    for i, (i_wave, q_wave) in enumerate(zip(i_waves, q_waves)):
        monitor.add_line(f"wave_{i}", t, [i_wave, q_wave])
    monitor.autoscale()
    pg.exec()
