"""Local experiment dashboard: run history, curves, logs and rollout videos.

Read-only over every worktree's runs/ directory; loopback only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from flyarm.dashboard.catalog import (
    Root,
    detail,
    discover_roots,
    find_runs,
    gallery,
    live,
    resolve_media,
    summarize,
)

STATIC = Path(__file__).parent / "static"


def create_dashboard_app(repo: Path) -> FastAPI:
    app = FastAPI(title="FlyArm dashboard", docs_url=None, redoc_url=None)

    def roots() -> list[Root]:
        return discover_roots(repo)

    def run_path(label: str, name: str) -> tuple[Root, Path]:
        root = next((r for r in roots() if r.label == label), None)
        if root is None:
            raise HTTPException(404, "unknown root")
        run = (root.path / name).resolve()
        if not run.is_relative_to(root.path) or not run.is_dir():
            raise HTTPException(404, "unknown run")
        return root, run

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text())

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        found = [summarize(root, run) for root in roots() for run in find_runs(root)]
        return {
            "roots": [{"label": r.label, "path": str(r.path)} for r in roots()],
            "runs": sorted(found, key=lambda item: item["updated"], reverse=True),
        }

    @app.get("/api/runs/{label}/{name}")
    def run_detail(label: str, name: str) -> dict[str, Any]:
        root, run = run_path(label, name)
        return detail(root, run)

    @app.get("/api/gallery")
    def videos() -> dict[str, Any]:
        return gallery(roots(), repo / "docs" / "featured-videos.json")

    @app.get("/api/live")
    def live_clips() -> dict[str, Any]:
        return live(roots())

    @app.get("/media/{label}/{relative:path}")
    def media(label: str, relative: str) -> FileResponse:
        try:
            path = resolve_media(roots(), label, relative)
        except PermissionError as error:
            raise HTTPException(403, "not a media file inside the runs directory") from error
        except FileNotFoundError as error:
            raise HTTPException(404, "not found") from error
        return FileResponse(path)

    return app


def serve_dashboard(repo: Path, host: str, port: int) -> None:
    import uvicorn

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("The dashboard only supports loopback hosts")
    uvicorn.run(create_dashboard_app(repo), host=host, port=port, log_level="warning")
