PIPE_NAME = "wave_monitor"
HEAD_LENGTH = 4  # Allows 4,294,967,295 bytes per message.
CHUNK_SIZE = 100_000_000  # 100MB, too large will cause socket closed.
