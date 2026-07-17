"""Minimal stdlib local web server for the PGIXS Tab1 UI prototype."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pgixs_python.ui_backend.api import startup_payload, status_payload, workflows_payload


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PGIXS Python Tab1</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1f2933; }
    main { max-width: 960px; margin: 0 auto; }
    pre { background: #f4f7f9; border: 1px solid #d9e2ec; padding: 1rem; overflow: auto; }
    .status { color: #334e68; }
  </style>
</head>
<body>
  <main>
    <h1>PGIXS Python Tab1</h1>
    <p class="status">Local web UI scaffold is running.</p>
    <p>API endpoints: <code>/api/status</code>, <code>/api/workflows</code>, <code>/api/startup</code></p>
    <pre id="startup">Loading startup payload...</pre>
  </main>
  <script>
    fetch("/api/startup")
      .then((response) => response.json())
      .then((payload) => {
        document.getElementById("startup").textContent = JSON.stringify(payload, null, 2);
      })
      .catch((error) => {
        document.getElementById("startup").textContent = String(error);
      });
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PGIXS Tab1 local web UI scaffold.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local UI server.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local UI server.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, int(args.port)), PGIXSRequestHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"PGIXS Tab1 local UI scaffold: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPGIXS local UI server stopped.")
    finally:
        server.server_close()
    return 0


class PGIXSRequestHandler(BaseHTTPRequestHandler):
    server_version = "PGIXSLocalUI/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
        if self.path in {"/", "/index.html"}:
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if self.path == "/api/status":
            self._send_json(status_payload())
            return
        if self.path == "/api/workflows":
            self._send_json(workflows_payload())
            return
        if self.path == "/api/startup":
            self._send_json(startup_payload())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    raise SystemExit(main())

