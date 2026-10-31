"""A scripted endpoint that sends a stale tool name to exercise engine refusal."""
import argparse
import json
from http.server import ThreadingHTTPServer
from pathlib import Path

from stateframe.scripted_endpoint import Script, make_handler


class ForcedScript(Script):
    def next_call(self, offered):
        if self.position < len(self.steps):
            offered = [*offered, self.steps[self.position]["tool"]]
        return super().next_call(offered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(ForcedScript(json.loads(args.script.read_text()))))
    print(f"http://127.0.0.1:{server.server_address[1]}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
