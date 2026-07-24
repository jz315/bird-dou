"""Dependency-free HTTP and static-file server for the local React game."""

from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

from .game import GameService, WebGameError
from .guandan import GuandanService
from .guandan.session import validate_card_ids

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATIC_ROOT = REPOSITORY_ROOT / "web" / "dist"


class BirdDouWebServer(ThreadingHTTPServer):
    """HTTP server carrying its game registry and built frontend path."""

    def __init__(self, address: tuple[str, int], static_root: Path) -> None:
        self.game_service = GameService(REPOSITORY_ROOT)
        self.guandan_service = GuandanService()
        self.static_root = static_root.resolve()
        super().__init__(address, BirdDouRequestHandler)


class BirdDouRequestHandler(BaseHTTPRequestHandler):
    """Serve the JSON game API and React build without external dependencies."""

    server: BirdDouWebServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "schemaVersion": 1})
            return
        if path == "/api/config":
            self._json(
                HTTPStatus.OK,
                {"schemaVersion": 1, "aiModes": self.server.game_service.available_modes()},
            )
            return
        parts = _parts(path)
        if len(parts) == 4 and parts[:3] == ("api", "guandan", "games"):
            self._handle(lambda: self.server.guandan_service.get_game(parts[3]))
            return
        if len(parts) == 3 and parts[:2] == ("api", "games"):
            self._handle(lambda: self.server.game_service.get_game(parts[2]))
            return
        if path.startswith("/api/"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            return
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        parts = _parts(path)
        if parts == ("api", "guandan", "games"):
            self._handle(self._create_guandan_game)
            return
        if (
            len(parts) == 5
            and parts[:3] == ("api", "guandan", "games")
            and parts[4] == "actions"
        ):
            self._handle(lambda: self._play_guandan(parts[3]))
            return
        if parts == ("api", "games"):
            self._handle(self._create_game)
            return
        if len(parts) == 4 and parts[:2] == ("api", "games") and parts[3] == "actions":
            self._handle(lambda: self._play(parts[2]))
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _create_game(self) -> dict[str, object]:
        body = self._body()
        seed_value = body.get("seed")
        if seed_value is not None and not isinstance(seed_value, int):
            raise WebGameError("牌局种子必须是整数")
        seat_value = body.get("humanSeat", 0)
        mode_value = body.get("aiMode", "heuristic")
        if not isinstance(seat_value, int) or not isinstance(mode_value, str):
            raise WebGameError("开局参数无效")
        return self.server.game_service.create_game(
            seed=seed_value,
            human_seat=seat_value,
            ai_mode=mode_value,
        )

    def _play(self, game_id: str) -> dict[str, object]:
        action_index = self._body().get("actionIndex")
        if not isinstance(action_index, int):
            raise WebGameError("动作编号必须是整数")
        return self.server.game_service.play(game_id, action_index)

    def _create_guandan_game(self) -> dict[str, object]:
        body = self._body()
        seed = body.get("seed")
        seat = body.get("humanSeat", 0)
        level = body.get("level", 0)
        mode = body.get("aiMode", "heuristic")
        if seed is not None and not isinstance(seed, int):
            raise WebGameError("牌局种子必须是整数")
        if not isinstance(seat, int) or not isinstance(level, int) or not isinstance(mode, str):
            raise WebGameError("掼蛋开局参数无效")
        return self.server.guandan_service.create_game(
            seed=seed,
            human_seat=seat,
            level=level,
            ai_mode=mode,
        )

    def _play_guandan(self, game_id: str) -> dict[str, object]:
        card_ids = validate_card_ids(self._body().get("cardIds"))
        return self.server.guandan_service.play(game_id, card_ids)

    def _body(self) -> Mapping[str, object]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise WebGameError("请求长度无效") from error
        if length < 0 or length > 64 * 1024:
            raise WebGameError("请求内容过大")
        try:
            value = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            raise WebGameError("请求不是有效 JSON") from error
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise WebGameError("请求必须是 JSON 对象")
        return cast(Mapping[str, object], value)

    def _handle(self, operation: object) -> None:
        try:
            if not callable(operation):
                raise TypeError("HTTP operation is not callable")
            payload = operation()
            if not isinstance(payload, dict):
                raise TypeError("HTTP operation returned a non-object")
        except WebGameError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:  # pragma: no cover - final HTTP safety boundary
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"服务器错误：{error}"})
            return
        self._json(HTTPStatus.OK, payload)

    def _static(self, request_path: str) -> None:
        root = self.server.static_root
        if not root.is_dir():
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "前端尚未构建，请先在 web 目录运行 npm run build"},
            )
            return
        relative = unquote(request_path).lstrip("/") or "index.html"
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            self._json(HTTPStatus.FORBIDDEN, {"error": "禁止访问该路径"})
            return
        if not candidate.is_file():
            candidate = root / "index.html"
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            "no-cache" if candidate.name == "index.html" else "public, max-age=3600",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        body = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/") if part)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play BIRD-Dou in a local React browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-root", type=Path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        raise ValueError("port must be in 1..65535")
    server = BirdDouWebServer((arguments.host, arguments.port), arguments.static_root)
    url = f"http://{arguments.host}:{arguments.port}"
    print(f"BIRD-Dou Web 正在运行：{url}")
    if arguments.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBIRD-Dou Web 已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
