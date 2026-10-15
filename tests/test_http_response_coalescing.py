"""Exercise the real adapter with an in-memory socket; no files or AWS calls.

The old adapter sent headers and body in separate socket writes. On Linux,
Nagle plus delayed ACK can stall the second small write. This test detects
that wire pattern without making a timing assertion on a different OS.
"""
import importlib.util
import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch


class MemorySocket:
    def __init__(self, body):
        self.input = io.BytesIO(
            b"POST /encode HTTP/1.1\r\nHost: test\r\nContent-Length: "
            + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        self.writes = []
        self.options = []

    def makefile(self, mode, buffering=None):
        if mode == "rb":
            return self.input
        return io.BufferedWriter(OutputSocket(self), buffer_size=buffering)

    def sendall(self, data):
        self.writes.append(bytes(data))

    def setsockopt(self, *args):
        self.options.append(args)


class OutputSocket(io.RawIOBase):
    def __init__(self, owner):
        self.owner = owner

    def writable(self):
        return True

    def write(self, data):
        self.owner.sendall(data)
        return len(data)


class ModelStub:
    def __init__(self, *args, **kwargs):
        self.identity = {"model": "diagnostic"}

    def handle(self, request):
        return {"vectors": "A" * 2732, "shape": [1, 512],
                "telemetry": {"encode_seconds": 0.0003}}


class ResponseCoalescingTests(unittest.TestCase):
    def test_real_adapter_flushes_one_complete_response_and_disables_nagle(self):
        path = Path(__file__).resolve().parents[1] / "scripts/isolated_model_http.py"
        spec = importlib.util.spec_from_file_location("diagnostic_http_adapter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sock = MemorySocket(json.dumps({"operation": "encode"}).encode())

        class ServerStub:
            def __init__(self, address, handler):
                self.handler = handler

            def serve_forever(self):
                self.handler(sock, ("127.0.0.1", 12345), self)

        argv = [str(path), "--root", ".", "--model", "potion_retrieval",
                "--resource-id", "diagnostic", "--bind", "127.0.0.1"]
        with patch.object(module, "ModelServer", ModelStub), \
             patch.object(module, "HTTPServer", ServerStub), \
             patch.object(module.sys, "argv", argv):
            module.main()
        self.assertEqual(len(sock.writes), 1, "Headers/body must not be separate small writes")
        self.assertIn(b"\r\n\r\n", sock.writes[0])
        self.assertIn(b'"vectors":', sock.writes[0])
        self.assertIn((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1), sock.options)


if __name__ == "__main__":
    unittest.main()
