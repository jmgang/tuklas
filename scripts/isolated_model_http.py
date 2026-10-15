"""Private EC2 serving adapter. Restart the process to switch static models."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pycon_model2vec_demo.isolated_serving import ModelServer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", required=True, choices=("potion_retrieval", "plain_student",
                    "custom_student", "tokenlearn_custom_student"))
    p.add_argument("--resource-id", required=True)
    p.add_argument("--bind", required=True, help="Bind to the EC2 private IP only")
    p.add_argument("--port", type=int, default=8080)
    a = p.parse_args()
    model = ModelServer(a.root, a.model, a.resource_id)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Coalesce headers and body before flushing. The previous unbuffered
        # adapter sent two small writes and was vulnerable to Linux delayed
        # ACK/Nagle stalls on persistent connections. These are serving-only
        # changes for FUTURE runs; existing signed results are not replaced.
        wbufsize = 64 * 1024
        disable_nagle_algorithm = True

        def do_POST(self):
            try:
                if self.path != "/encode":
                    raise ValueError("Unknown path")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 6 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                value = model.handle(data)
                status = 200
            except (ValueError, KeyError, TypeError) as error:
                value, status = {"error": str(error)}, 400
            encoded = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
            self.close_connection = False

        def log_message(self, *_):
            pass  # Avoid request logging overhead in timed phases.

    print(json.dumps({"stage": "server-ready", "identity": model.identity}), flush=True)
    HTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
