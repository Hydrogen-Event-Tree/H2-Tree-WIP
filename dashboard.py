from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run(
    events_path: str = "events.json",
    index_path: str = "index.html",
    host: str = "127.0.0.1",
    port: int = 4000,
    max_tries: int = 5,
) -> None:
    index_html = _read_text(index_path)
    base_dir = Path(index_path).resolve().parent

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, code: int, content_type: str, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _respond_json_file(self, path: Path) -> None:
            try:
                payload = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                self._respond(404, "text/plain; charset=utf-8", f"{path.name} not found.")
            except OSError as exc:
                self._respond(500, "text/plain; charset=utf-8", f"Unable to read {path.name}: {exc}")
            else:
                self._respond(200, "application/json; charset=utf-8", payload)

        def do_GET(self):  # type: ignore[override]
            if self.path in ("/", "/index.html"):
                self._respond(200, "text/html; charset=utf-8", index_html)
                return

            if self.path == "/events.json":
                self._respond_json_file(Path(events_path))
                return

            if self.path.endswith(".json"):
                requested = (base_dir / self.path.lstrip("/")).resolve()
                try:
                    requested.relative_to(base_dir)
                except ValueError:
                    self._respond(403, "text/plain; charset=utf-8", "Forbidden.")
                    return
                self._respond_json_file(requested)
                return

            self._respond(404, "text/plain; charset=utf-8", "Not found.")

        def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
            return  # Silence default logging.

    server = None
    bound_port = port
    last_error: Exception | None = None
    for attempt in range(max_tries):
        try_port = port + attempt
        try:
            server = ReusableHTTPServer((host, try_port), Handler)
            bound_port = try_port
            break
        except OSError as exc:
            last_error = exc
            continue

    if server is None:
        raise last_error if last_error else RuntimeError("Unable to start server.")

    print(f"dashboard available at localhost:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
