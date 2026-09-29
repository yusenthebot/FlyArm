"""Small shared file helpers for run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def save_json(path: Path, value: dict | list) -> None:
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as file:
        file.write(serialized)
        temporary = Path(file.name)
    temporary.replace(path)
