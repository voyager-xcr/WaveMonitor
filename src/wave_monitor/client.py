import logging
import queue
import subprocess
import threading
import time
import warnings
from concurrent.futures import Future
from typing import Literal

import numpy as np
from PySide6.QtNetwork import QLocalSocket
from typing_extensions import deprecated

from .constants import CHUNK_SIZE, HEAD_LENGTH, PIPE_NAME
from .proto import decode, encode

logger = logging.getLogger(__name__)


class _IOWorker(threading.Thread):
    """A background I/O worker that owns the QLocalSocket and performs I/O.

    Contracts:
    - Single connection to server (server closes previous connection on new one).
    - All socket access happens in this thread only.
    - Supported ops: write (fire-and-forget), query (request/response),
      connect, disconnect.
    """

    def __init__(self, pipe_name: str, head_len: int, chunk_size: int, logger: logging.Logger):
        super().__init__(name="WaveMonitorIO", daemon=True)
        self._pipe_name = pipe_name
        self._head_len = head_len
        self._chunk_size = chunk_size
        self._logger = logger.getChild("IOWorker")

        # Queue of (operation, payload) items to process in the worker thread
        self._tasks: queue.Queue[tuple[str, dict]] = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        self._sock: QLocalSocket | None = None

    # Public API (thread-safe)
    def submit_write(self, msg: dict) -> None:
        self._tasks.put(("write", {"msg": msg}))

    def submit_query(self, msg: dict, timeout_ms: int) -> Future:
        fut: Future = Future()
        self._tasks.put(("query", {"msg": msg, "timeout_ms": timeout_ms, "future": fut}))
        return fut

    def submit_connect(self, timeout_ms: int) -> Future:
        fut: Future = Future()
        self._tasks.put(("connect", {"timeout_ms": timeout_ms, "future": fut}))
        return fut

    def submit_disconnect(self) -> Future:
        fut: Future = Future()
        self._tasks.put(("disconnect", {"future": fut}))
        return fut

    def stop(self) -> None:
        self._stop_event.set()
        # Add a no-op to unblock queue
        self._tasks.put(("_stop", {}))

    def join(self, timeout: float | None = None) -> None:  # type: ignore[override]
        """Wait for the worker thread to exit."""
        super().join(timeout)

    # Thread run-loop
    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                op, payload = self._tasks.get(timeout=0.1)
            except queue.Empty:
                continue

            if op == "_stop":
                break

            try:
                if op == "write":
                    self._handle_write(payload["msg"])  # fire-and-forget
                elif op == "query":
                    self._handle_query(payload["msg"], payload["timeout_ms"], payload["future"])  # sets result/exception
                elif op == "connect":
                    self._handle_connect(payload["timeout_ms"], payload["future"])  # sets bool
                elif op == "disconnect":
                    self._handle_disconnect(payload["future"])  # sets bool
                else:
                    self._logger.warning("Unknown op: %s", op)
            except Exception as e:
                self._logger.exception("IO worker op failed: %s", op)
                # If there's a future attached, surface the error
                fut = payload.get("future")
                if isinstance(fut, Future) and not fut.done():
                    fut.set_exception(e)

        # Cleanup
        try:
            if self._sock is not None:
                if self._sock.state() == QLocalSocket.ConnectedState:
                    self._sock.disconnectFromServer()
                    self._sock.waitForDisconnected()
        except Exception:
            pass

    # Internal helpers (run in worker thread)
    def _ensure_connected(self, timeout_ms: int = 100) -> bool:
        if self._sock is None:
            # Create socket in this thread to keep Qt thread affinity consistent
            self._sock = QLocalSocket()
        if self._sock.state() == QLocalSocket.ConnectedState:
            return True
        # Always disconnect first to refresh state
        try:
            self._sock.disconnectFromServer()
        except Exception:
            pass
        self._sock.connectToServer(self._pipe_name)
        return bool(self._sock.waitForConnected(timeout_ms))

    def _handle_connect(self, timeout_ms: int, fut: Future) -> None:
        ok = self._ensure_connected(timeout_ms)
        if not fut.done():
            fut.set_result(bool(ok))

    def _handle_disconnect(self, fut: Future) -> None:
        if self._sock is None:
            ok = True
        else:
            try:
                self._sock.disconnectFromServer()
                ok = True
                if self._sock.state() == QLocalSocket.ConnectedState:
                    ok = bool(self._sock.waitForDisconnected())
            except Exception:
                ok = False
        if not fut.done():
            fut.set_result(ok)

    def _write_payload(self, payload: bytes) -> None:
        assert self._sock is not None
        total_length = len(payload)
        self._sock.write(total_length.to_bytes(self._head_len, "big"))
        self._sock.waitForBytesWritten()

        start = 0
        while start < total_length:
            end = min(start + self._chunk_size, total_length)
            chunk = payload[start:end]
            written_len = self._sock.write(chunk)
            if written_len == -1:
                raise RuntimeError("Failed to write to socket.")
            start += written_len
            self._logger.debug(
                "Wrote %d bytes, %d bytes remaining.",
                written_len,
                total_length - start,
            )

    def _handle_write(self, msg: dict) -> None:
        # Try connect briefly; if not connected, warn and drop (match previous behavior)
        if not self._ensure_connected(timeout_ms=100):
            warnings.warn("Not connected to server.", stacklevel=2)
            return
        # Handle heavy pre-processing for specific message types in the background thread
        try:
            if msg.get("_type") == "add_wfm":
                conv_dtype = msg.pop("_convert_dtype", None)
                if conv_dtype is not None:
                    # Perform astype conversions off the main thread
                    t = msg.get("t")
                    ys = msg.get("ys")
                    if t is not None and ys is not None:
                        msg["t"] = t.astype(conv_dtype)
                        msg["ys"] = [y.astype(conv_dtype) for y in ys]
        except Exception:
            self._logger.exception("Failed dtype conversion in I/O worker; sending original arrays.")

        self._logger.debug("msg to send (async): %r", msg)
        payload = encode(msg)
        self._write_payload(payload)

    def _handle_query(self, msg: dict, timeout_ms: int, fut: Future) -> None:
        if not self._ensure_connected(timeout_ms=100):
            # Keep previous semantics: warn and set exception to inform caller
            warnings.warn("Not connected to server.")
            if not fut.done():
                fut.set_exception(ConnectionError("Not connected to server"))
            return
        self._logger.debug("query send: %r", msg)
        payload = encode(msg)
        self._write_payload(payload)

        assert self._sock is not None
        self._sock.waitForBytesWritten()
        if self._sock.waitForReadyRead(timeout_ms):
            data = self._sock.readAll().data().strip()
        else:
            if not fut.done():
                fut.set_exception(TimeoutError(
                    f"No response from server for timeout={timeout_ms!r}, msg={msg!r}."
                ))
            return

        try:
            reply = decode(data)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
            return
        if not fut.done():
            fut.set_result(reply)


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
    WFM_INTERVAL_CACHE_DURATION = 3.0  # seconds

    def __init__(self, create_window: bool = True) -> None:
        # Background I/O worker
        self._io = _IOWorker(PIPE_NAME, HEAD_LENGTH, CHUNK_SIZE, logger)
        self._io.start()

        self._last_wfm_time = {}
        self._wfm_interval = 0.0
        self._last_interval_check = 0.0

        if create_window:
            try:
                self.find_or_create_window()
            except Exception:
                self.logger.exception("Failed to connect to server.")

    @deprecated("offset will be ignored. Use add_wfm instead.")
    def add_line(self, name: str, t: np.ndarray, ys: list[np.ndarray], offset) -> None:
        self.add_wfm(name, t, ys)

    def add_wfm(
        self,
        name: str,
        t: np.ndarray,
        ys: list[np.ndarray],
        dtype: np.float16 | np.float32 | np.float64 | None = np.int16,
    ) -> None:
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
        # Defer dtype conversion to I/O worker to avoid blocking main thread
        convert_dtype = None
        if dtype in (np.float16, np.float32, np.float64):
            convert_dtype = dtype
        self.write(dict(_type="add_wfm", name=name, t=t, ys=ys, _convert_dtype=convert_dtype))

    def get_wfm_interval(self):
        if time.time() - self._last_interval_check < self.WFM_INTERVAL_CACHE_DURATION:
            return self._wfm_interval

        try:
            msg = self.query(dict(_type="get_wfm_interval"), timeout_ms=200)
            # Tests monkeypatch query to return packed bytes; accept both forms
            if isinstance(msg, (bytes, bytearray)):
                msg = decode(msg)
            self._wfm_interval = float(msg["interval"])
            self._last_interval_check = time.time()
        except Exception:
            self.logger.debug("get_wfm_interval: query failed or timed out")

        return self._wfm_interval

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
        """Asynchronously send a message via the background I/O thread.

        Returns immediately. Errors (like not connected) will be logged/warned
        by the background worker.
        """
        self._io.submit_write(msg)

    def query(self, msg: dict, timeout_ms: int = 1000):
        """Send a message and wait for a response (synchronous API).

        This delegates to the background I/O worker so the main thread doesn't
        block on large writes. Raises TimeoutError on no response.
        """
        fut = self._io.submit_query(msg, timeout_ms)
        try:
            reply = fut.result(timeout=timeout_ms / 1000 + 1)
        except Exception as e:
            # Surface TimeoutError or connection errors
            raise e
        self.logger.debug("query received: %r", reply)
        return reply

    def disconnect(self) -> None:
        fut = self._io.submit_disconnect()
        ok = fut.result(timeout=1.0)
        if not ok:
            raise RuntimeError("Could not disconnect from server")

    def close(self) -> None:
        """Stop background I/O worker."""
        try:
            self._io.stop()
        except Exception:
            pass
        try:
            self._io.join(timeout=1.0)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def join(self, timeout: float | None = None) -> None:
        """Convenience wrapper to wait for the I/O worker to terminate."""
        try:
            self._io.join(timeout)
        except Exception:
            pass

    def refresh_connect(self, timeout_ms: int = 100) -> bool:
        """Connect to server via the background worker and return success status."""
        fut = self._io.submit_connect(timeout_ms)
        return bool(fut.result(timeout=max(1.0, timeout_ms / 1000 + 0.1)))

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
