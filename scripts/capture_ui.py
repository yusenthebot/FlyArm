"""Screenshot a running FlyArm live UI mid-episode, through the UI itself.

Headless Chrome (software WebGL) opens the page, the script clicks the page's own buttons
(for example Reset then Run), waits while the timeline fills, and saves a PNG. Usage:

    uv run python scripts/capture_ui.py http://localhost:8771 out.png --clicks Reset Run \
        --seconds 8
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


def _targets(timeout: float = 20.0) -> list[dict]:
    """Chrome's DevTools target list, waiting for the debugging port on a busy machine."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


async def _capture(url: str, output: Path, clicks: list[str], seconds: float) -> None:
    targets = _targets()
    page = next(target for target in targets if target["type"] == "page")
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**28) as socket:
        counter = 0

        async def call(method: str, **params: object) -> dict:
            nonlocal counter
            counter += 1
            await socket.send(json.dumps({"id": counter, "method": method, "params": params}))
            while True:
                message = json.loads(await socket.recv())
                if message.get("id") == counter:
                    if "error" in message:
                        raise RuntimeError(f"{method}: {message['error']}")
                    return message.get("result", {})

        await call("Page.navigate", url=url)
        await asyncio.sleep(6)
        for label in clicks:
            script = (
                "(() => { const b = [...document.querySelectorAll('button')]"
                f".find(x => x.textContent.trim().endsWith({json.dumps(label)}));"
                " if (!b) return false; b.click(); return true; })()"
            )
            result = await call("Runtime.evaluate", expression=script, returnByValue=True)
            if not result.get("result", {}).get("value"):
                raise RuntimeError(f"No button labelled {label!r}")
            await asyncio.sleep(1)
        await asyncio.sleep(seconds)
        shot = await call("Page.captureScreenshot", format="png")
        output.write_bytes(base64.b64decode(shot["data"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--clicks", nargs="*", default=[])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen(
            [
                CHROME,
                "--headless=new",
                f"--remote-debugging-port={PORT}",
                f"--user-data-dir={profile}",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--hide-scrollbars",
                f"--window-size={args.width},{args.height}",
                "--force-device-scale-factor=2",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            asyncio.run(_capture(args.url, args.output, args.clicks, args.seconds))
        finally:
            chrome.terminate()
            try:
                chrome.wait(timeout=10)
            except subprocess.TimeoutExpired:
                chrome.kill()
                chrome.wait(timeout=10)
    print(args.output)


if __name__ == "__main__":
    main()
