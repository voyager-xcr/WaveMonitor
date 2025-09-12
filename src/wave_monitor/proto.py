import msgpack
import msgpack_numpy


def decode(data: bytes):
    return msgpack.unpackb(data, object_hook=msgpack_numpy.decode)

def encode(obj) -> bytes:
    return msgpack.packb(obj, default=msgpack_numpy.encode)
