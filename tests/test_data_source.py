
import msgpack
import msgpack_numpy
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from wave_monitor.window import DataSource


class DummyParent(QObject):
    """Minimal QObject parent with attributes used by DataSource."""

    def __init__(self):
        super().__init__()
        self.wfm_interval = 0.5


def test_emit_signals_get_wfm_interval():
    app = QApplication.instance() or QApplication([])
    parent = DummyParent()
    server = DataSource(parent)

    # Simulate a client connection object with write method.
    class FakeConn:
        def __init__(self):
            self.written = b""

        class _Signal:
            def disconnect(self, *args, **kwargs):
                return None

        def write(self, b):
            # QLocalSocket.write may return int; our fake should accept bytes
            self.written += b

        def close(self):
            return None

        # Provide a readyRead-like object with disconnect method used in close_client_connection
        @property
        def readyRead(self):
            return FakeConn._Signal()

    fake = FakeConn()
    server.client_connection = fake

    # send get_wfm_interval message
    msg = {"_type": "get_wfm_interval"}
    server.handle_client_message(msg)

    # The server should write a msgpack reply to the connection
    assert fake.written != b""
    unpacked = msgpack.unpackb(fake.written, object_hook=msgpack_numpy.decode)
    assert isinstance(unpacked, dict)
    assert unpacked.get("_type") == "wfm_interval"
    server.close()
