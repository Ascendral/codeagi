from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class JsonStateStore:
    """Tiny JSON-backed state store used by the connective prototypes."""

    def __init__(self, path: str | Path, manifest_store: Any | None = None, key: str = "") -> None:
        self.path = Path(path)
        self.manifest_store = manifest_store
        self.key = key

    def load(self, default: Any = None) -> Any:
        if not self.path.exists():
            return deepcopy(default)
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return deepcopy(default)

    def save(self, data: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        if self.manifest_store is not None:
            self.manifest_store.record(self.key, self.path)
