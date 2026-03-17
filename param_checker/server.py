from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

import torch

from .loader import TensorMeta, tensor_meta, tensor_to_view, tensor_view_2d


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


class ParamCheckerHandler(BaseHTTPRequestHandler):
    tensors: Dict[str, "torch.Tensor"] = {}
    cache: Dict[str, "TensorCacheEntry"] = {}

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, payload: str, status: int = 200) -> None:
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _get_cache_entry(self, key: str) -> "TensorCacheEntry | None":
        tensor = self.tensors.get(key)
        if tensor is None:
            return None
        entry = self.cache.get(key)
        if entry is None:
            entry = TensorCacheEntry(key=key, tensor=tensor)
            self.cache[key] = entry
        return entry

    @staticmethod
    def _parse_int(value: str | None) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _serve_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_text("Not found", 404)
            return

        content = path.read_bytes()
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None:
            mime = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/keys":
            keys = sorted(self.tensors.keys())
            self._send_json({"keys": keys})
            return

        if parsed.path == "/api/meta":
            params = parse_qs(parsed.query)
            key = params.get("key", [""])[0]
            if not key:
                self._send_text("Missing key", 400)
                return
            entry = self._get_cache_entry(key)
            if entry is None:
                self._send_text("Key not found", 404)
                return
            meta = entry.meta()
            self._send_json({"meta": meta.__dict__})
            return

        if parsed.path == "/api/tile":
            params = parse_qs(parsed.query)
            key = params.get("key", [""])[0]
            if not key:
                self._send_text("Missing key", 400)
                return
            entry = self._get_cache_entry(key)
            if entry is None:
                self._send_text("Key not found", 404)
                return

            r0 = self._parse_int(params.get("r0", [None])[0])
            c0 = self._parse_int(params.get("c0", [None])[0])
            rows = self._parse_int(params.get("rows", [None])[0])
            cols = self._parse_int(params.get("cols", [None])[0])
            stride = self._parse_int(params.get("stride", ["1"])[0]) or 1

            if r0 is None or c0 is None or rows is None or cols is None:
                self._send_text("Missing tile parameters", 400)
                return
            if rows <= 0 or cols <= 0 or stride <= 0:
                self._send_text("Invalid tile parameters", 400)
                return

            view = entry.view()
            max_rows, max_cols = view.shape
            r0 = max(0, min(r0, max_rows - 1))
            c0 = max(0, min(c0, max_cols - 1))
            r1 = max(0, min(r0 + rows, max_rows))
            c1 = max(0, min(c0 + cols, max_cols))

            tile = view[r0:r1:stride, c0:c1:stride]
            meta = entry.meta()
            min_value = meta.min_value
            max_value = meta.max_value

            if max_value == min_value:
                tile_uint8 = torch.full(tile.shape, 127, dtype=torch.uint8)
            else:
                norm = (tile - min_value) / (max_value - min_value)
                norm = norm.clamp(0, 1)
                tile_uint8 = (norm * 255).to(torch.uint8)

            payload: dict = {
                "key": key,
                "r0": r0,
                "c0": c0,
                "rows": r1 - r0,
                "cols": c1 - c0,
                "stride": stride,
                "tile_rows": int(tile_uint8.shape[0]),
                "tile_cols": int(tile_uint8.shape[1]),
            }

            try:
                import numpy as np  # noqa: F401

                data_bytes = tile_uint8.cpu().numpy().tobytes()
                payload["encoding"] = "base64"
                payload["data"] = base64.b64encode(data_bytes).decode("ascii")
            except Exception:
                payload["encoding"] = "list"
                payload["data"] = tile_uint8.cpu().tolist()

            self._send_json(payload)
            return

        if parsed.path == "/api/value":
            params = parse_qs(parsed.query)
            key = params.get("key", [""])[0]
            row = self._parse_int(params.get("row", [None])[0])
            col = self._parse_int(params.get("col", [None])[0])
            if not key or row is None or col is None:
                self._send_text("Missing value parameters", 400)
                return
            entry = self._get_cache_entry(key)
            if entry is None:
                self._send_text("Key not found", 404)
                return
            view = entry.view()
            rows, cols = view.shape
            if row < 0 or col < 0 or row >= rows or col >= cols:
                self._send_text("Index out of range", 400)
                return
            value = float(view[row, col].item())
            self._send_json({"key": key, "row": row, "col": col, "value": value})
            return

        if parsed.path == "/api/tensor":
            params = parse_qs(parsed.query)
            key = params.get("key", [""])[0]
            if not key:
                self._send_text("Missing key", 400)
                return
            entry = self._get_cache_entry(key)
            if entry is None:
                self._send_text("Key not found", 404)
                return
            meta = entry.meta()
            if meta.numel > 262144:
                self._send_json(
                    {"error": "Tensor too large. Use /api/meta and /api/tile.", "meta": meta.__dict__},
                    status=413,
                )
                return
            view = tensor_to_view(key, entry.tensor)
            self._send_json({"tensor": view.__dict__})
            return

        if parsed.path == "/" or parsed.path == "":
            self._serve_static(WEB_ROOT / "index.html")
            return

        safe_path = (WEB_ROOT / parsed.path.lstrip("/")).resolve()
        if not str(safe_path).startswith(str(WEB_ROOT.resolve())):
            self._send_text("Invalid path", 400)
            return
        self._serve_static(safe_path)


def run_server(tensors: Dict[str, "torch.Tensor"], host: str, port: int) -> None:
    ParamCheckerHandler.tensors = tensors
    ParamCheckerHandler.cache = {}
    server = ThreadingHTTPServer((host, port), ParamCheckerHandler)
    print(f"Serving on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@dataclass
class TensorCacheEntry:
    key: str
    tensor: torch.Tensor
    _view: Optional[torch.Tensor] = None
    _meta: Optional[TensorMeta] = None

    def view(self) -> torch.Tensor:
        if self._view is None:
            self._view = tensor_view_2d(self.tensor)
        return self._view

    def meta(self) -> TensorMeta:
        if self._meta is None:
            self._meta = tensor_meta(self.key, self.tensor, view=self.view())
        return self._meta
