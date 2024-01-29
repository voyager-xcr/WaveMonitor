"""A simple GUI for monitoring waveforms.

Usage:
    monitor = WaveMonitor()
    monitor.run_monitor_window()
"""

import logging
import subprocess
import sys
import time
import warnings
from typing import Any, Callable, Literal

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
    QMessageBox,
)

PIPE_NAME = "wave_monitor"
__version__ = "0.0.1"
about_message = (
    f"<b>Wave Monitor</b> v{__version__}<br><br>"
    "A simple GUI for monitoring waveforms.<br><br>"
    "by Jiawei Qiu"
)
logger = logging.getLogger(__name__)


class WaveMonitor:
    """Wrapper to operate Monitor in a separate process.

    Before using it, start a new monitor window by either of following methods:

    1. Call monitor.run_monitor_window(). This creates a new process for app event loop.

    2. Run this script, which blocks the process for app event loop.

    Based on QLocalSocket (like named pipe). Messages are serialized with msgpack.

    Note:
        The wrapper is not intend for Qt application, which means no event loop,
        also no signals or slots.
    """

    logger = logger.getChild("WaveMonitor")

    def __init__(self) -> None:
        self.sock = QLocalSocket()
        self.sock.connectToServer(PIPE_NAME)

    def add_line(
        self, name: str, t: np.ndarray, ys: list[np.ndarray], offset: float
    ) -> None:
        # For compatibility, TODO: remove this.
        warnings.warn(
            "Use add_wfm instead. This will be removed by v0.1",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_wfm(name, t, ys)

    def add_wfm(self, name: str, t: np.ndarray, ys: list[np.ndarray]) -> None:
        self.write(dict(_type="add_wfm", name=name, t=t, ys=ys))

    def remove_wfm(self, name: str) -> None:
        self.write(dict(_type="remove_wfm", name=name))

    def clear(self) -> None:
        self.write(dict(_type="clear"))

    def autoscale(self) -> None:
        self.write(dict(_type="autoscale"))

    def write(self, msg: dict) -> None:
        if self.sock.state() != QLocalSocket.ConnectedState:
            raise RuntimeError("Socket not connected")

        msg = dump(msg) + b"\n"  # Add a newline to indicate the end of message.
        self.sock.write(len(msg).to_bytes(9) + b"\n")  # first 10 bytes is msg length
        self.sock.waitForBytesWritten()
        self.sock.write(msg)  # Send the message

    def query(self, msg: dict, timeout_ms: int = 1000) -> bytes:
        # BUG: timeout if too much previous data waitting to send. Maybe flush hleps.
        self.write(msg)
        self.sock.waitForBytesWritten()  # Make sure bytes written.
        if self.sock.waitForReadyRead(timeout_ms):
            msg = self.sock.readAll().data().strip()  # Could be empty.
        else:
            msg = b""
        return msg

    def query_and_decode(self, msg: dict, timeout_ms: int = 1000) -> Any:
        reply = self.query(msg, timeout_ms)
        if reply:
            return load(reply)
        else:
            return None

    def disconnect(self) -> None:
        self.sock.disconnectFromServer()
        if self.sock.state() == QLocalSocket.ConnectedState:
            if not self.sock.waitForDisconnected():
                raise RuntimeError("Could not disconnect from server")

    def confirm_connect(self, timeout_ms: int = 100) -> bool:
        """Connect to server and returns success status."""
        self.disconnect()  # Refresh the state, otherwise the state is still connected.
        self.sock.connectToServer(PIPE_NAME)
        result = self.sock.waitForConnected(timeout_ms)
        return result

    def run_monitor_window(
        self,
        log_level: Literal["WARNING", "INFO", "DEBUG"] = "INFO",
        aviod_multiple: bool = True,
        timeout_s: float = 10,
    ) -> None:
        """Connect to existing monitor_window or create one in new process.

        Blocks until server is listening.
        """
        cmd = ["cmd", "/c", "start", sys.executable, __file__, f"--log={log_level}"]

        if not self.confirm_connect(timeout_ms=100):
            subprocess.run(cmd)
        elif aviod_multiple:
            self.logger.info("Monitor is already running, not starting a new one.")
            return None
        else:
            subprocess.run(cmd)
            warnings.warn("Monitor is already running, starting a duplicate one.")

        start_time = time.time()
        while not self.confirm_connect(timeout_ms=100):
            if time.time() - start_time > timeout_s:
                raise RuntimeError("Timeout waiting for server to start listening.")
            time.sleep(0.1)

    def echo(self) -> bytes:
        """Check if the server is responding, for testing purpose."""
        reply = self.query(dict(_type="are_you_there"))
        if reply != b"yes":
            raise RuntimeError("Server is not responding.")
        return reply


class DataSource(QLocalServer):
    """Receive messages from MonitorWrapper and emit signals to trigger operation on monitor."""

    add_wfm = Signal(str, np.ndarray, list)
    remove_wfm = Signal(str)
    clear = Signal()
    autoscale = Signal()
    logger = logger.getChild("DataSource")

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.partial_msg: bytes = b""
        self.expected_msg_length: int = None

        self.newConnection.connect(self.handle_new_connection)
        QApplication.instance().aboutToQuit.connect(self.close)
        # Remove previous instance. see https://doc.qt.io/qtforpython-6/PySide6/QtNetwork/QLocalServer.html#PySide6.QtNetwork.PySide6.QtNetwork.QLocalServer.removeServer
        # self.removeServer(PIPE_NAME)  # Remove previous instance.
        self.listen(PIPE_NAME)
        self.logger.info('Listening on "%s".', PIPE_NAME)

    def handle_new_connection(self):
        self.close_client_connection()
        self.client_connection = self.nextPendingConnection()
        self.client_connection.readyRead.connect(self.assmeble_message)
        self.client_connection.disconnected.connect(
            lambda: self.logger.info("Client disconnected.")
        )
        self.logger.info("New client connected.")

    def assmeble_message(self):
        # One readyRead signal may contain multiple messages.
        while self.client_connection.canReadLine():
            # Read the msg length.
            if self.expected_msg_length is None:
                if self.client_connection.bytesAvailable() < 10:
                    continue
                line = self.client_connection.read(10).data()
                try:
                    self.expected_msg_length = int.from_bytes(line[:-1])
                    logger.debug(f"Expecting {self.expected_msg_length} bytes for msg.")
                except:
                    logger.exception(f"Failed to parse msg length: {line}")
                    continue

            # Read the msg.
            if self.client_connection.bytesAvailable() < self.expected_msg_length:
                continue

            msg = self.client_connection.read(self.expected_msg_length).data()
            logger.debug(f"Received {len(msg)} bytes.")
            self.partial_msg += msg
            self.expected_msg_length -= len(msg)

            if len(self.partial_msg) < self.expected_msg_length:
                self.logger.debug(
                    "msg len: %s, expected: %s",
                    len(self.partial_msg),
                    self.expected_msg_length,
                )
                continue

            # Process the message
            msg = load(self.partial_msg[:-1])
            self.partial_msg = b""
            self.expected_msg_length = None

            self.logger.debug(f"Received: {msg}")
            self.emit_signals(msg)

    def emit_signals(self, msg: dict):
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

    def close_client_connection(self):
        if hasattr(self, "client_connection"):
            self.client_connection.readyRead.disconnect(self.assmeble_message)
            self.client_connection.close()  # Not working, because client not in qt event loop.

    def close(self):
        self.close_client_connection()
        self.logger.info('Closing server "%s".', PIPE_NAME)
        super().close()


def dump(payload: Any) -> bytes:
    """Serialize payload with msgpack."""
    return msgpack.packb(payload, default=msgpack_numpy.encode)

def load(payload: bytes) -> Any:
    """Deserialize payload with msgpack."""
    return msgpack.unpackb(payload, object_hook=msgpack_numpy.decode)


class MonitorWindow:
    """Keep some widgets and plot waveforms with them."""

    logger = logger.getChild("MonitorWindow")

    def __init__(self, wfm_seperation: float = 2):
        """Construct widgets."""
        MonitorWindow.setup_app_style(QApplication.instance())
        window = QMainWindow()
        window.setWindowTitle("Wave Monitor")
        window.setWindowIcon(QIcon("osci3.png"))
        QShortcut("F", window).activated.connect(self.autoscale)
        QShortcut("C", window).activated.connect(self.confirm_clear)
        QShortcut("R", window).activated.connect(self.refresh_plots)
        QShortcut("H", window).activated.connect(self.hide_all)
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

        dock_widget = QDockWidget("wfms⪅30", window)
        list_widget = QListWidget()
        list_widget.setDragDropMode(QListWidget.InternalMove)
        _filter = DeleteEventFilter(self.remove_wfm, list_widget)
        list_widget.installEventFilter(_filter)
        self._delete_event_filter = _filter
        dock_widget.setWidget(list_widget)
        dock_widget.setFloating(False)
        dock_widget.setStyleSheet(
            "QScrollBar:vertical {width: 10px;}" "QScrollBar:horizontal {height: 10px;}"
        )
        window.addDockWidget(Qt.RightDockWidgetArea, dock_widget)
        font_metrics = dock_widget.fontMetrics()
        initial_width = font_metrics.horizontalAdvance("X") * 15  # 15 chars wide.
        window.resizeDocks([dock_widget], [initial_width], Qt.Horizontal)

        server = DataSource(window)
        server.add_wfm.connect(self.add_wfm)
        server.remove_wfm.connect(self.remove_wfm)
        server.clear.connect(self.clear)
        server.autoscale.connect(self.autoscale)

        window.show()
        self.logger.info("Ready. Right-click to show menu.")
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

    def hide_all(self):
        for wfm in self.wfms.values():
            wfm.set_visible(False)

    def remove_wfm(self, name: str):
        if name in self.wfms:
            self.wfms[name].remove()
            del self.wfms[name]
        else:
            self.logger.debug(f"Waveform {name} not found, nothing removed.")

    def clear(self):
        for name in list(self.wfms.keys()):
            self.remove_wfm(name)

    def confirm_clear(self):
        """Ask user to confirm before clearing all wfms."""
        reply = QMessageBox.question(
            self.window,
            "Clear all waveforms?",
            "Are you sure to clear all waveforms?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.clear()

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

        refresh_action = QAction("Refresh plots (R)", self.window)
        refresh_action.triggered.connect(self.refresh_plots)
        context_menu.addAction(refresh_action)

        hide_all_action = QAction("Hide all (H)", self.window)
        hide_all_action.triggered.connect(self.hide_all)
        context_menu.addAction(hide_all_action)

        clear_action = QAction("Clear all (C)", self.window)
        clear_action.triggered.connect(self.confirm_clear)
        context_menu.addAction(clear_action)

        sort_action = QAction('Sort "wfms" list', self.window)
        sort_action.triggered.connect(self.list_widget.sortItems)
        context_menu.addAction(sort_action)

        dock_restore_action = QAction('Restore "wfms" list', self.window)
        dock_restore_action.triggered.connect(self.restore_dock)
        context_menu.addAction(dock_restore_action)

        # # Not working. But anyway, it is slow.
        # export_action = QAction("PyQtGraph Export (csv slow!)", self.window)
        # export_action.triggered.connect(self.plot_widget.sceneObj.showExportDialog)

        context_menu.addSeparator()

        about_action = QAction("About", self.window)
        about_action.triggered.connect(self.show_about_dialog)
        context_menu.addAction(about_action)

        context_menu.exec(self.plot_widget.mapToGlobal(pos.toPoint()))

    def show_about_dialog(self):
        QMessageBox.about(self.window, "About Wave Monitor", about_message)

    @staticmethod
    def setup_app_style(app: QApplication) -> None:
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

    def _add_test_wfm(self):
        i = len(self.wfms)
        t = np.linspace(0, 1, 100_001)
        i_wave = np.cos(2 * np.pi * i * t)
        q_wave = np.sin(2 * np.pi * i * t)
        self.add_wfm(f"test_wfm_{i}", t, [i_wave, q_wave])

    def _add_test_wfm1(self):
        t = np.linspace(0, 1, 10_001)
        f = np.random.randint(3, 100)
        i_wave = np.cos(2 * np.pi * f * t)
        q_wave = np.sin(2 * np.pi * f * t)
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


class DeleteEventFilter(QObject):
    def __init__(self, remove_wfm: Callable[[str], None], list_widget: QListWidget):
        super().__init__()
        self.remove_wfm = remove_wfm
        self.list_widget = list_widget

    def eventFilter(self, source, event):
        if (
            source is self.list_widget
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Delete
        ):
            current_item = self.list_widget.currentItem()
            if current_item is not None:
                self.remove_wfm(current_item.text())
            return True
        return super().eventFilter(source, event)


def config_log(dafault_loglevel="INFO"):
    # Get the log level from command line arguments, find pattern like "=-log=DEBUG"
    loglevel = next(
        (arg.split("=")[1] for arg in sys.argv if arg.startswith("--log=")),
        dafault_loglevel,
    )
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    config_log()
    app = QApplication(sys.argv)
    monitor = MonitorWindow()
    sys.exit(app.exec())
