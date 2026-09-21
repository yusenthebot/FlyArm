"""Pinned upstream assets. No credentials and no implicit downloads at import time."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import httpx

MENAGERIE_SHA = "822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7"
BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
SOURCES = {
    "annotations": (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        14483314,
        "50a7718770c57220f160ba4f431ab89e",
        "1780494878811468",
    ),
    "weights": (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1051241946,
        "f30e9dcca25cfd021bf1e7b3d975599e",
        "1780494887545976",
    ),
    "neurotransmitters": (
        "body-neurotransmitters-male-cns-v1.0.feather",
        43282834,
        "3d842b12fe5c49eefade528d7dd24a1f",
        "1780894899156750",
    ),
}
SOURCE_SHA256 = {
    "annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "weights": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    "neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
}


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, algorithm).hexdigest()


def fetch_data(destination: Path) -> dict:
    """Download immutable GCS generations, verify size/transport MD5, record SHA256."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"dataset": "MaleCNS v1.0", "license": "CC-BY-4.0", "files": {}}
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        for key, (name, size, md5, generation) in SOURCES.items():
            target = destination / name
            url = f"{BUCKET}/{name}?generation={generation}"
            if not target.exists():
                partial = target.with_suffix(".part")
                written = 0
                with client.stream("GET", url) as response, partial.open("wb") as output:
                    response.raise_for_status()
                    for chunk in response.iter_bytes(2**20):
                        written += len(chunk)
                        if written > size:
                            raise ValueError(f"Download exceeded pinned size: {name}")
                        output.write(chunk)
                if partial.stat().st_size != size or digest_file(partial, "md5") != md5:
                    raise ValueError(f"Size/hash mismatch: {partial}")
                partial.rename(target)
            if target.stat().st_size != size or digest_file(target, "md5") != md5:
                raise ValueError(f"Existing data does not match pinned source: {target}")
            sha256 = digest_file(target)
            if sha256 != SOURCE_SHA256[key]:
                raise ValueError(f"SHA256 mismatch: {name}")
            manifest["files"][key] = {
                "name": name,
                "url": url,
                "bytes": size,
                "sha256": sha256,
            }
    (destination / "sources.lock.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _git(*arguments: str, timeout: int = 180) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is required to fetch and verify Menagerie")
    # Only internal fixed subcommands; paths are absolute and never shell-interpolated.
    result = subprocess.run(  # nosec B603
        [executable, *arguments], check=True, text=True, capture_output=True, timeout=timeout
    )
    return result.stdout.strip()


def fetch_arm(destination: Path) -> Path:
    """Fetch only the upstream Panda asset directory at an exact Git revision."""
    destination = destination.resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth=1",
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            str(destination),
        )
        _git("-C", str(destination), "sparse-checkout", "set", "franka_emika_panda")
        _git("-C", str(destination), "fetch", "--depth=1", "origin", MENAGERIE_SHA)
        _git("-C", str(destination), "checkout", "--detach", MENAGERIE_SHA)
    return verify_arm(destination)


def verify_arm(destination: Path) -> Path:
    """Read-only verification of a clean pinned asset checkout."""
    destination = destination.resolve()
    actual = _git("-C", str(destination), "rev-parse", "HEAD", timeout=30)
    if actual != MENAGERIE_SHA:
        raise ValueError(f"Unexpected Menagerie revision {actual}; expected {MENAGERIE_SHA}")
    dirty = _git("-C", str(destination), "status", "--porcelain", timeout=30)
    if dirty:
        raise ValueError("Menagerie has local changes; use a fresh asset directory")
    scene = destination / "franka_emika_panda" / "scene.xml"
    if not scene.is_file():
        raise FileNotFoundError(f"Incomplete asset checkout: {scene}")
    return scene
