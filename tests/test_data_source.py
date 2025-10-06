from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from wave_monitor.window import DataSource, SharedServerState


class DummyParent(QObject):
    """Minimal QObject parent with attributes used by DataSource. Needs state attribute now."""

    def __init__(self):
        super().__init__()
        self.state = SharedServerState()


def test_emit_signals_removed_wfm_interval():
    """
    Note: The get_wfm_interval message handler has been removed in favor of shared memory.
    wfm_interval is now shared via multiprocessing.shared_memory.ShareableList instead
    of message-based communication, which is much faster.
    
    This test verifies that the get_wfm_interval message type is no longer handled.
    """
    app = QApplication.instance() or QApplication([])
    parent = DummyParent()
    server = DataSource(parent)
    
    # Test that get_wfm_interval message type is no longer handled
    class FakeConn:
        def __init__(self):
            self.written = b""
        
        def write(self, b):
            self.written += b
        
        def close(self):
            return None
        
        class _Signal:
            def disconnect(self, *args, **kwargs):
                return None
        
        @property
        def readyRead(self):
            return FakeConn._Signal()
    
    fake = FakeConn()
    server.client_connection = fake
    
    # The get_wfm_interval message should now be treated as unknown message type
    # We expect no response to be written since it's an unknown message
    msg = {"_type": "get_wfm_interval"}
    server.handle_client_message(msg)
    
    # No response should be written for unknown message types
    assert fake.written == b""
    
    server.close()
