from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from vercel_api import build_scale_response  # noqa: E402


def _read_json(request: BaseHTTPRequestHandler) -> dict:
    length = int(request.headers.get("content-length", "0") or 0)
    if length <= 0:
        return {}
    body = request.rfile.read(length).decode("utf-8")
    return json.loads(body or "{}")


def _read_query(request: BaseHTTPRequestHandler) -> dict:
    parsed = urlparse(request.path)
    values = parse_qs(parsed.query)
    return {key: value[-1] for key, value in values.items()}


def _send_json(request: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    request.send_response(status)
    request.send_header("Content-Type", "application/json")
    request.send_header("Access-Control-Allow-Origin", "*")
    request.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    request.send_header("Access-Control-Allow-Headers", "Content-Type")
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        _send_json(self, 204, {})

    def do_GET(self) -> None:
        try:
            _send_json(self, 200, build_scale_response(_read_query(self)))
        except Exception as exc:
            _send_json(self, 500, {"detail": str(exc)})

    def do_POST(self) -> None:
        try:
            _send_json(self, 200, build_scale_response(_read_json(self)))
        except Exception as exc:
            _send_json(self, 500, {"detail": str(exc)})
