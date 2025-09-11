import logging
import subprocess
import time
import warnings
from typing import Literal

import msgpack
import msgpack_numpy
import numpy as np
from PySide6.QtNetwork import QLocalSocket
from typing_extensions import deprecated

from .constants import CHUNK_SIZE, HEAD_LENGTH, PIPE_NAME

logger = logging.getLogger(__name__)


class WaveMonitor:
    """Wrapper to operate Monitor in a separate process.

    Before using it, start a new monitor window by either of following methods:

    1. Call monitor.find_or_create_window().

    2. Run this script, which blocks the process for app event loop.

    Note:
        The wrapper is not intend for Qt application, which means neither event loop,
        no receiving/emiting signals or slots.
    """

    logger = logger.getChild("WaveMonitor")

    def __init__(self, create_window: bool = True) -> None:
        self.sock = QLocalSocket()
        self.sock.connectToServer(PIPE_NAME)

        self._last_wfm_time = {}

        if create_window:
            try:
                self.find_or_create_window()
            except Exception:
                self.logger.exception("Failed to connect to server.")

    @deprecated("offset will be ignored. Use add_wfm instead.")
    def add_line(self, name: str, t: np.ndarray, ys: list[np.ndarray], offset) -> None:
        self.add_wfm(name, t, ys)

    def add_wfm(self, name: str, t: np.ndarray, ys: list[np.ndarray]) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(t, np.ndarray):
            raise TypeError("t must be a numpy array")
        if not isinstance(ys, list):
            raise TypeError("ys must be a list")
        if t.ndim != 1:
            raise ValueError("t must be 1D")
        for y in ys:
            if not isinstance(y, np.ndarray):
                raise TypeError("ys must be a list of numpy arrays")
            if y.ndim != 1:
                raise ValueError("ys must be a list of 1D numpy arrays")
            if y.shape != t.shape:
                raise ValueError("ys must have the same shape as t")

        now = time.time()
        server_interval = self.get_wfm_interval()
        last_time = self._last_wfm_time.get(name, 0)
        if (now - last_time) < server_interval:
            self.logger.debug(
                "Skipping adding waveform '%s' due to interval limit: %.3f seconds.",
                name,
                server_interval,
            )
            return None
        self._last_wfm_time[name] = now

        self.logger.debug("Adding waveform '%s'", name)
        self.write(dict(_type="add_wfm", name=name, t=t, ys=ys))

    def get_wfm_interval(self):
        try:
            reply = self.query(dict(_type="get_wfm_interval"), timeout_ms=200)
            if reply:
                data = msgpack.unpackb(reply, object_hook=msgpack_numpy.decode)
                self.logger.debug("msg received: %r", data)
                if isinstance(data, dict) and data.get("_type") == "wfm_interval":
                    return float(data.get("interval", 0.0) or 0.0)
        except Exception:
            self.logger.debug("get_wfm_interval: query failed or timed out")

        return 0.0

    def remove_wfm(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        self.write(dict(_type="remove_wfm", name=name))

    def clear(self) -> None:
        """Set all waveforms to zero.

        Note: This does not remove the waveforms, right click on the window to remove them.
        """
        self.write(dict(_type="clear"))

    def autoscale(self) -> None:
        self.write(dict(_type="autoscale"))

    def add_note(self, name: str, note: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(note, str):
            raise TypeError("note must be a string")

        self.write(dict(_type="add_note", name=name, note=note))

    def write(self, msg: dict) -> None:
        if self.sock.state() != QLocalSocket.ConnectedState:
            if not self.refresh_connect():
                warnings.warn("Not connected to server.")
                return

        self.logger.debug("msg to send: %r", msg)

        sock = self.sock
        payload = msgpack.packb(msg, default=msgpack_numpy.encode)
        total_length = len(payload)
        sock.write(total_length.to_bytes(HEAD_LENGTH, "big"))
        sock.waitForBytesWritten()

        start = 0
        while start < total_length:
            end = min(start + CHUNK_SIZE, total_length)
            chunk = payload[start:end]
            written_len = sock.write(chunk)
            if written_len == -1:
                raise RuntimeError("Failed to write to socket.")
            sock.waitForBytesWritten()
            start += written_len
            self.logger.debug(
                "Wrote %d bytes, %d bytes remaining.",
                written_len,
                total_length - start,
            )

    def query(self, msg: dict, timeout_ms: int = 1000) -> bytes:
        """Send a message and wait for a response.

        Raise TimeoutError if no response within timeout period.
        """
        if self.sock.state() != QLocalSocket.ConnectedState:
            if not self.refresh_connect():
                warnings.warn("Not connected to server.")
                return

        self.write(msg)
        self.sock.waitForBytesWritten()

        if self.sock.waitForReadyRead(timeout_ms):
            return self.sock.readAll().data().strip()
        else:
            raise TimeoutError(
                "No response from server for timeout=%r, msg=%r.", timeout_ms, msg
            )

    def disconnect(self) -> None:
        self.sock.disconnectFromServer()
        if self.sock.state() == QLocalSocket.ConnectedState:
            if not self.sock.waitForDisconnected():
                raise RuntimeError("Could not disconnect from server")

    def refresh_connect(self, timeout_ms: int = 100) -> bool:
        """Connect to server and returns success status."""
        self.disconnect()  # Refresh the state, otherwise the state is still connected.
        self.sock.connectToServer(PIPE_NAME)
        result = self.sock.waitForConnected(timeout_ms)
        return result

    def find_or_create_window(
        self,
        log_level: Literal["WARNING", "INFO", "DEBUG"] = "INFO",
        aviod_multiple: bool = True,
        timeout_s: float = 10,
    ) -> None:
        """Connect to existing monitor window
        or create one in new process.

        Blocks until server is listening.
        """
        # TODO: --log is not implemented
        cmd = ["start-wave-monitor", f"--log={log_level}"]
        if not self.refresh_connect(timeout_ms=100):
            subprocess.Popen(cmd)
        elif aviod_multiple:
            warnings.warn("Monitor is already running, not starting a new one.")
            return
        else:
            warnings.warn("Monitor is already running, starting a duplicate one.")
            subprocess.Popen(cmd)

        start_time = time.time()
        while not self.refresh_connect(timeout_ms=100):
            self.logger.debug("Waiting for server to start listening.")
            if time.time() - start_time > timeout_s:
                raise TimeoutError("Timeout waiting for server to start listening.")
            time.sleep(0.1)

    def echo(self) -> bytes:
        return self.query(dict(_type="are_you_there"))
