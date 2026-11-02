"""Scripted endpoint that pauses the second model reply for interruption tests."""
import argparse
import json
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from stateframe.scripted_endpoint import Script, make_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--pause-at", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=30)
    parser.add_argument("--tokenize-off-marker", type=Path)
    args = parser.parse_args()
    parent = make_handler(Script(json.loads(args.script.read_text())))

    class Handler(parent):
        calls = 0

        def do_POST(self):
            if (args.tokenize_off_marker is not None and args.tokenize_off_marker.exists()
                    and self.path.rstrip("/").endswith("/tokenize")):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self._send(404, {"error": {"message": "tokenizer unavailable"}})
                return
            if self.path.rstrip("/") == "/v1/chat/completions":
                Handler.calls += 1
                if Handler.calls == args.pause_at:
                    args.marker.write_text(f"request {args.pause_at} reached endpoint\n")
                    time.sleep(args.pause_seconds)
            return super().do_POST()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(f"http://127.0.0.1:{server.server_address[1]}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
