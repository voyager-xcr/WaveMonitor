import logging

import msgpack
import msgpack_numpy
import numpy as np
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

PIPE_NAME = "wave_monitor_test"
HEAD_LENGTH = 4  # Allows 4,294,967,295 bytes per message.
CHUNK_SIZE = 100_000_000  # 100MB, too large will cause socket closed.

def write_msg(sock: QLocalSocket, payload: bytes) -> None:
    """Write a message to the socket in chunks."""
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
        print(f"Wrote {written_len:,} bytes, {total_length - start:,} bytes remaining.")

def client():
    sock = QLocalSocket()
    sock.connectToServer(PIPE_NAME)
    sock.state()

    t = np.linspace(0, 1, 100_000_001)
    ys = [np.sin(2 * np.pi * f * t) for f in [1,2]]
    msg = dict(_type="add_wfm", name="wave", t=t, ys=ys)
    msg = msgpack.packb(msg, default=msgpack_numpy.encode)
    f"{len(msg):,} bytes"

    write_msg(sock, msg)
    write_msg(sock, msgpack.packb('abc', default=msgpack_numpy.encode))
    write_msg(sock, b"asdhc")  # Illegal msg.
    write_msg(sock, msgpack.packb('abc', default=msgpack_numpy.encode))

    sock.disconnectFromServer()

class DataSource(QLocalServer):
    logger = logger.getChild("DataSource")

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.frame_buffer: bytes = b""
        self.frame_length: int | None = None

        self.newConnection.connect(self.handle_new_connection)
        QApplication.instance().aboutToQuit.connect(self.close)

        # Remove previous instance. see https://doc.qt.io/qtforpython-6/PySide6/QtNetwork/QLocalServer.html#PySide6.QtNetwork.PySide6.QtNetwork.QLocalServer.removeServer
        # self.removeServer(PIPE_NAME)  # Remove previous instance.
        self.listen(PIPE_NAME)

        self.logger.info('Listening on "%s".', self.fullServerName())

    def handle_new_connection(self):
        self.close_client_connection()  # Close previous connection.
        self.client_connection = self.nextPendingConnection()
        self.client_connection.readyRead.connect(self.read_frame)  # Trigger on buffer update.
        self.client_connection.disconnected.connect(
            lambda: self.logger.info("Client disconnected.")
        )
        self.logger.info("New client connected.")

    def read_frame(self):
        while self.client_connection.bytesAvailable():
            self.logger.debug(">>> Buffer updated")
            if self.frame_length is None:
                if self.client_connection.bytesAvailable() < HEAD_LENGTH:
                    self.logger.debug("Not enough data to read frame length.")
                    return  # Wait for next buffer update.
                data = self.client_connection.read(HEAD_LENGTH).data()
                try:
                    self.frame_length = int.from_bytes(data, "big")
                    self.logger.info(f"{self.frame_length=:,}")
                except Exception:
                    self.frame_length = None
                    self.logger.exception("Failed to parse frame length: %r", data)
                    continue  # Try again.

            expected = min(self.frame_length - len(self.frame_buffer), CHUNK_SIZE)
            available = self.client_connection.bytesAvailable()
            if available < expected:
                self.logger.debug(f"{expected=:,}, {available=:,}")
                return  # Wait for next buffer update.

            data = self.client_connection.read(expected).data()
            self.logger.debug(f"Received {len(data):,} bytes.")
            self.frame_buffer += data
    
            if len(self.frame_buffer) < self.frame_length:
                continue  # Proceed to read the rest.

            try:
                data = msgpack.unpackb(self.frame_buffer, object_hook=msgpack_numpy.decode)
            except Exception:
                data = None
                self.logger.exception("Failed to parse msg: %r", self.frame_buffer)

            self.frame_buffer = b""
            self.frame_length = None
            self.logger.info("<<< Received: %r", data)

            continue  # There might be more data.

    def close_client_connection(self):
        if hasattr(self, "client_connection"):
            self.client_connection.readyRead.disconnect(self.read_frame)
            self.client_connection.close()  # Not working, because client not in qt event loop.
            self.frame_buffer = b""
            self.remaining_length = None

    def close(self):
        self.close_client_connection()
        self.logger.info('Closing server "%s".', PIPE_NAME)
        super().close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication([])
    server = DataSource(app)
    app.exec()
