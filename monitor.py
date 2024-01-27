import logging
import sys
import warnings
from typing import Callable

import msgpack
import msgpack_numpy
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QMouseEvent,
    QPalette,
    QShortcut,
)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
)

PIPE_NAME = "wave_monitor"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class WaveMonitor:
    """Wrapper to operate Monitor in a separate process.

    Based on QLocalSocket (like named pipe). Messages are serialized with msgpack.

    Note:
        The wrapper is not intend for Qt application, which means no event loop,
        also no signals or slots.
    """

    def __init__(self) -> None:
        self.sock = QLocalSocket()
        self.sock.connectToServer(PIPE_NAME)

    def add_line(
        self, name: str, t: np.ndarray, ys: list[np.ndarray], offset: float
    ) -> None:
        warnings.warn("Use add_wfm instead.", DeprecationWarning)
        self.add_wfm(name, t, ys)

    def add_wfm(self, name: str, t: np.ndarray, ys: list[np.ndarray]) -> None:
        self.send_msg(dict(_type="add_wfm", name=name, t=t, ys=ys))

    def remove_wfm(self, name: str) -> None:
        self.send_msg(dict(_type="remove_wfm", name=name))

    def clear(self) -> None:
        self.send_msg(dict(_type="clear"))

    def autoscale(self) -> None:
        self.send_msg(dict(_type="autoscale"))

    def send_msg(self, msg: dict) -> None:
        msg = msgpack.packb(msg, default=msgpack_numpy.encode)
        self.sock.write(msg)

    def close(self) -> None:
        # Seems not necessary, because the server only listen to the newest connection.
        self.sock.disconnectFromServer()
        if self.sock.state() == QLocalServer.ConnectedState:
            if not self.sock.waitForDisconnected():
                warnings.warn("Could not disconnect from server")

    def hello(self) -> bytes:
        self.send_msg(dict(_type="are_you_there"))
        if self.sock.waitForReadyRead(100):  # timeout in ms
            msg = self.sock.readAll().data().strip()
        else:
            msg = None
        if msg != b"yes":
            raise RuntimeError("WaveViewer is not responding.")
        return msg


class DataSource(QLocalServer):
    """Receive messages from MonitorWrapper and emit signals to trigger operation on monitor."""

    add_wfm = Signal(str, np.ndarray, list)
    remove_wfm = Signal(str)
    clear = Signal()
    autoscale = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.newConnection.connect(self.handle_new_connection)

    def handle_new_connection(self):
        # Only listening the newest client connection, igonring all previous.
        self.client_connection = self.nextPendingConnection()
        self.client_connection.readyRead.connect(self.read_client_and_emit)

    def read_client_and_emit(self):
        msg = self.client_connection.readAll().data().strip()
        msg = msgpack.unpackb(msg, object_hook=msgpack_numpy.decode)
        logger.debug(f"Received: {msg}")

        if msg["_type"] == "add_wfm":
            self.add_wfm.emit(msg["name"], msg["t"], msg["ys"])
        elif msg["_type"] == "remove_wfm":
            self.remove_wfm.emit(msg["name"])
        elif msg["_type"] == "clear":
            self.clear.emit()
        elif msg["_type"] == "autoscale":
            self.autoscale.emit()
        elif msg["_type"] == "are_you_there":
            self.client_connection.write(b"yes")
        else:
            raise ValueError(f"Unknown message type: {msg['_type']}")


class MonitorWindow:
    """Keep some widgets and plot waveforms with them."""

    def __init__(self, wfm_seperation: float = 2):
        """Construct widgets."""
        window = MainWindow()
        window.setWindowTitle("Wave Monitor")
        window.setWindowIcon(QIcon("osci3.png"))
        QShortcut("F", window).activated.connect(self.autoscale)
        QShortcut("C", window).activated.connect(self.clear)
        QShortcut("R", window).activated.connect(self.refresh_plots)
        QShortcut("Shift+A", window).activated.connect(self._add_test_wfm)
        QShortcut("Shift+1", window).activated.connect(self._add_test_wfm1)

        plot_widget = pg.plot(parent=window)
        window.setCentralWidget(plot_widget)

        plot_item = plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True)
        # Make it hold millions of points.
        plot_item.setDownsampling(auto=True, mode="subsample")
        plot_item.setClipToView(True)
        # ClipToView disables plot_item.autoRange, as well as "View all" in right-click menu.
        plot_item.getViewBox().disableAutoRange()

        # Custom context menu.
        plot_item.getViewBox().setMenuEnabled(False)  # Disable the menu by pyqtgraph.
        _filter = RightClickFilter(self.show_context_menu)
        # viewport gets the mouseReleaseEvent, See https://blog.csdn.net/theoryll/article/details/110918779
        plot_widget.viewport().installEventFilter(_filter)
        self._right_click_filter = _filter

        dock_widget = QDockWidget("visible wfms⪅30", window)
        list_widget = QListWidget()
        list_widget.setDragDropMode(QListWidget.InternalMove)
        dock_widget.setWidget(list_widget)
        dock_widget.setFloating(False)
        dock_widget.setStyleSheet(
            "QScrollBar:vertical {width: 10px;}" "QScrollBar:horizontal {height: 10px;}"
        )
        window.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
        font_metrics = dock_widget.fontMetrics()
        initial_width = font_metrics.horizontalAdvance("X") * 15  # 15 chars wide.
        window.resizeDocks([dock_widget], [initial_width], Qt.Horizontal)

        server = DataSource()
        server.add_wfm.connect(self.add_wfm)
        server.remove_wfm.connect(self.remove_wfm)
        server.clear.connect(self.clear)
        server.autoscale.connect(self.autoscale)
        # server.removeServer(PIPE_NAME)  # Remove previous instance.
        # Interesting, read https://doc.qt.io/qtforpython-6/PySide6/QtNetwork/QLocalServer.html#PySide6.QtNetwork.PySide6.QtNetwork.QLocalServer.listen
        server.listen(PIPE_NAME)
        window.closing.connect(server.close)

        window.show()
        self.wfms: dict[str, "Waveform"] = {}
        self.window = window
        self.plot_widget = plot_widget
        self.plot_item = plot_item
        self.dock_widget = dock_widget
        self.list_widget = list_widget
        self.server = server
        self.wfm_seperation = wfm_seperation

    def add_wfm(self, name: str, t: np.ndarray, ys: list[np.ndarray]):
        if name in self.wfms:
            wfm = self.wfms[name]
            wfm.update_wfm(t, ys)
        else:
            visible_wfms = self.visible_wfms
            offset = self.wfm_seperation * len(visible_wfms)
            wfm = Waveform(name, t, ys, offset, self.plot_item, self.list_widget)
            if len(visible_wfms) >= 20:
                wfm.set_visible(False)
            self.wfms[name] = wfm

    def remove_wfm(self, name: str):
        if name in self.wfms:
            self.wfms[name].remove()
            del self.wfms[name]

    def clear(self):
        for name in list(self.wfms.keys()):
            self.remove_wfm(name)

    def autoscale(self):
        if self.wfms:
            t0 = min(wfm.t0 for wfm in self.wfms.values())
            t1 = max(wfm.t1 for wfm in self.wfms.values())
            y0 = min(wfm.offset for wfm in self.wfms.values()) - 1
            y1 = max(wfm.offset for wfm in self.wfms.values()) + 1
            self.plot_item.setRange(xRange=(t0, t1), yRange=(y0, y1))

    def refresh_plots(self):
        for i, wfm in enumerate(self.visible_wfms):
            wfm.update_offset(self.wfm_seperation * i)

    @property
    def visible_wfms(self) -> list["Waveform"]:
        """Return a list of visible wfms, sorted as in list_widget."""
        list_wfms = []
        for name in self.list_names:
            wfm = self.wfms[name]
            if wfm.is_visible():
                list_wfms.append(wfm)
        return list_wfms

    @property
    def list_names(self) -> list[str]:
        """Return list of item names in list_widget, should be names of wfms."""
        list_widget = self.list_widget
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def restore_dock(self):
        if not self.dock_widget.isVisible():
            self.dock_widget.show()

    def show_context_menu(self, pos: QPointF):
        context_menu = QMenu(self.plot_widget)

        zoom_fit_action = QAction("Zoom fit (F)", self.window)
        zoom_fit_action.triggered.connect(self.autoscale)
        context_menu.addAction(zoom_fit_action)

        arrange_action = QAction("Refresh plots (R)", self.window)
        arrange_action.triggered.connect(self.refresh_plots)
        context_menu.addAction(arrange_action)

        clear_action = QAction("Clear all (C)", self.window)
        clear_action.triggered.connect(self.clear)
        context_menu.addAction(clear_action)

        dock_restore_action = QAction('Open "visible wfms" list', self.window)
        dock_restore_action.triggered.connect(self.restore_dock)
        context_menu.addAction(dock_restore_action)

        # Not working. But anyway, it is slow.
        export_action = QAction("PyQtGraph Export (csv slow!)", self.window)
        export_action.triggered.connect(self.plot_widget.sceneObj.showExportDialog)

        context_menu.exec(self.plot_widget.mapToGlobal(pos.toPoint()))

    def _add_test_wfm(self):
        i = len(self.wfms)
        t = np.linspace(0, 1, 100_001)
        i_wave = np.cos(2 * np.pi * i * t)
        q_wave = np.sin(2 * np.pi * i * t)
        self.add_wfm(f"test_wfm_{i}", t, [i_wave, q_wave])

    def _add_test_wfm1(self):
        t = np.linspace(0, 1, 10_001)
        i_wave = np.random.rand(t.size)
        q_wave = np.random.rand(t.size)
        z_wave = np.random.rand(t.size)
        self.add_wfm(f"test_wfm_random", t, [i_wave, q_wave, z_wave])

class Waveform:
    """Container for all assets of a waveform."""

    colors = (
        # # Simple RBG
        # (255, 0, 0, 50),
        # (0, 0, 255, 50),
        # (0, 255, 0, 50),
        # "dark_background" in https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        (214, 98, 86, 80),
        (98, 144, 176, 80),
        (217, 147, 69, 80),
        (146, 188, 75, 80),
        (155, 99, 156, 80),
        (170, 200, 163, 80),
        (219, 202, 81, 80),
        (110, 177, 166, 80),
        (218, 219, 146, 80),
        (158, 154, 183, 80),
    )

    def __init__(
        self,
        name: str,
        t: np.ndarray,
        ys: list[np.ndarray],
        offset: float,
        plot_item: pg.PlotItem,
        list_widget: QListWidget,
    ):
        """Add line plot to plot_item, add checkbox to list_widget."""
        lines: list[pg.PlotDataItem] = [
            plot_item.plot(
                t, y + offset, pen=color[:-1], fillLevel=offset, fillBrush=color
            )
            for y, color in zip(ys, self.colors)
        ]

        text = pg.TextItem(text=name, anchor=(1, 0.5))
        plot_item.addItem(text)
        plot_item.sigXRangeChanged.connect(self.update_label_pos)

        list_item = QListWidgetItem(name)
        list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)  # Add checkbox.
        list_item.setCheckState(Qt.Checked)
        # QListWidgetItem is not a QObject, so it can't emit signals.
        # The checkbox state change is emitted by QListWidget.
        list_widget.itemChanged.connect(self.handel_checkbox_change)
        list_widget.addItem(list_item)

        self.offset = offset
        self.t0 = t[0]
        self.t1 = t[-1]
        self.plot_item = plot_item
        self.lines = lines
        self.text = text
        self.update_label_pos()
        self.list_item = list_item
        self.list_widget = list_widget

    def update_wfm(self, t: np.ndarray, ys: list[np.ndarray]):
        # Update existing lines with new data.
        old_lines = self.lines
        new_lines = []
        for line, y in zip(self.lines, ys):
            line.setData(t, y + self.offset)
            new_lines.append(line)

        # Remove unused lines.
        if len(ys) < len(old_lines):
            for line in old_lines[len(ys) :]:
                self.plot_item.removeItem(line)

        # Add more lines if needed.
        if len(ys) > len(old_lines):
            for y, color in zip(ys[len(old_lines) :], self.colors[len(old_lines) :]):
                line = self.plot_item.plot(
                    t,
                    y + self.offset,
                    pen=color[:-1],
                    fillLevel=self.offset,
                    fillBrush=color,
                )
                new_lines.append(line)

        self.t0 = t[0]
        self.t1 = t[-1]
        self.lines = new_lines

    def update_offset(self, offset: float):
        old_offset = self.offset
        new_offset = offset
        for line in self.lines:
            t, y = line.getData()
            line.setData(t, y - old_offset + new_offset)
            line.setFillLevel(new_offset)
        self.offset = new_offset
        self.update_label_pos()

    def remove(self):
        for line in self.lines:
            self.plot_item.removeItem(line)

        self.plot_item.removeItem(self.text)
        self.plot_item.sigXRangeChanged.disconnect(self.update_label_pos)

        row = self.list_widget.row(self.list_item)
        self.list_widget.takeItem(row)

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

        # Change checkbox state without triggering handel_checkbox_change.
        self.list_widget.itemChanged.disconnect(self.handel_checkbox_change)
        self.list_item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        self.list_widget.itemChanged.connect(self.handel_checkbox_change)

    def handel_checkbox_change(self, item: QListWidgetItem):
        """Triggered when the checkbox is clicked."""
        if item is self.list_item:
            self.set_visible(item.checkState() == Qt.Checked)

    def is_visible(self) -> bool:
        return self.text.isVisible()


class RightClickFilter(QObject):
    def __init__(self, show_ctx_menu: Callable[[QPointF], None]):
        super().__init__()
        self.show_ctx_menu = show_ctx_menu
        self.mouse_press_pos = None

    def eventFilter(self, watched, event: QMouseEvent):
        # Filter the right-click instead dragging.
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:
                self.mouse_press_pos = event.position()
        if event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.RightButton:
                if self.mouse_press_pos is not None:
                    if (event.position() - self.mouse_press_pos).manhattanLength() < 5:
                        self.show_ctx_menu(event.position())
        return super().eventFilter(watched, event)


class MainWindow(QMainWindow):
    closing = Signal()

    def closeEvent(self, event):
        self.closing.emit()
        return super().closeEvent(event)


def setup_window_style(app: QApplication) -> None:
    """Set the window style to Dark Fusion and use Segoe UI font."""
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    setup_window_style(app)
    monitor = MonitorWindow()

    # t = np.linspace(0, 1, 1_00_001)  # 1m pts ~= 1ms for 1GSa/s.
    # n = 10
    # i_waves = [np.cos(2 * np.pi * f * t) for f in range(1, n + 1)]
    # q_waves = [np.sin(2 * np.pi * f * t) for f in range(1, n + 1)]
    # for i, (i_wave, q_wave) in enumerate(zip(i_waves, q_waves)):
    #     monitor.add_wfm(f"wave_{i}", t, [i_wave, q_wave])
    # monitor.autoscale()

    sys.exit(app.exec())
