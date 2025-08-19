import logging
import subprocess
import time
import warnings
from multiprocessing.connection import Client as MPClient
from typing import TYPE_CHECKING, Literal

import msgpack
import msgpack_numpy
import numpy as np
from typing_extensions import deprecated

if TYPE_CHECKING:
    from multiprocessing.connection import PipeConnection

PIPE_NAME = "//./pipe/WaveMonitor"
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
        self.conn: "PipeConnection | None" = None
        if create_window:
            try:
                self.find_or_create_window()
            except:
                self.logger.exception("Failed to connect to server.")

    @deprecated(
        "offset will be ignored. Use add_wfm instead. This will be removed by v0.1"
    )
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

        self.write(dict(_type="add_wfm", name=name, t=t, ys=ys))

    def remove_wfm(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        self.write(dict(_type="remove_wfm", name=name))

    def clear(self) -> None:
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
        if not self.conn or self.conn.closed:
            if not self.refresh_connect():
                raise RuntimeError("Socket not connected")

        self.logger.debug(f"msg to send: {msg}")

        msg = msgpack.packb(msg, default=msgpack_numpy.encode)
        self.conn.send_bytes(msg)
        self.logger.debug(f"msg sent: {len(msg)} bytes")

    def query(self, msg: dict, timeout_ms: int = 1000) -> bytes:
        # BUG: timeout if too much previous data waitting to send. Maybe flush helps.
        self.write(msg)
        start_time = time.time()
        while time.time() - start_time < timeout_ms / 1000:
            if self.conn.poll():
                msg = self.conn.recv_bytes()
                return msg
            time.sleep(0.01)  # 10ms
        return b""

    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def refresh_connect(self, timeout_ms: int = 100) -> bool:
        """Connect to server and returns success status."""
        self.disconnect()  # Refresh the state, otherwise the state is still connected.
        try:
            self.conn = MPClient(PIPE_NAME, family="AF_PIPE")
            return True
        except:
            return False

    def find_or_create_window(
        self,
        log_level: Literal["WARNING", "INFO", "DEBUG"] = "INFO",
        aviod_multiple: bool = True,
        timeout_s: float = 10,
    ) -> None:
        """Connect to existing monitor_window or create one in new process.

        Blocks until server is listening.
        """
        # start without blocking
        cmd = ["start-wave-monitor", f"--log={log_level}"]
        if not self.refresh_connect(timeout_ms=100):
            subprocess.Popen(cmd)
        elif aviod_multiple:
            self.logger.info("Monitor is already running, not starting a new one.")
            return
        else:
            subprocess.Popen(cmd)
            warnings.warn("Monitor is already running, starting a duplicate one.")

        start_time = time.time()
        while not self.refresh_connect(timeout_ms=100):
            self.logger.debug("Waiting for server to start listening.")
            if time.time() - start_time > timeout_s:
                raise RuntimeError("Timeout waiting for server to start listening.")
            time.sleep(0.1)

    def echo(self) -> bytes:
        """Check if the server is responding, for testing purpose."""
        reply = self.query(dict(_type="are_you_there"))
        if reply != b"yes":
            raise RuntimeError("Server is not responding.")
        return reply
