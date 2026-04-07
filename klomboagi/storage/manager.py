from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from klomboagi.utils.time import utc_now


@dataclass
class StoragePaths:
    runtime_root: Path
    long_term_root: Path


class ManifestStore:
    """Minimal manifest tracker for JSON state files."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, key: str, value: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, str] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                data = {}
        if key:
            data[key] = str(value)
            self.path.write_text(json.dumps(data, indent=2))


class EventLog:
    """Append-only JSONL event log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": utc_now(),
            "type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


class StorageManager:
    """Minimal storage bootstrap used by the connective tests."""

    def __init__(self, paths: StoragePaths) -> None:
        self.paths = paths
        self.manifest_store = ManifestStore(paths.long_term_root / "manifests" / "objects.json")
        self.event_log = EventLog(paths.runtime_root / "events.jsonl")

    @classmethod
    def bootstrap(cls) -> "StorageManager":
        project_root = Path(__file__).resolve().parents[2]
        runtime_root = Path(os.environ.get("KLOMBOAGI_RUNTIME_ROOT", project_root / "runtime"))
        long_term_root = Path(os.environ.get("KLOMBOAGI_LONG_TERM_ROOT", project_root / "long-term"))
        runtime_root.mkdir(parents=True, exist_ok=True)
        long_term_root.mkdir(parents=True, exist_ok=True)
        return cls(StoragePaths(runtime_root=runtime_root, long_term_root=long_term_root))

    def _json_path(self, name: str) -> Path:
        filename = name if name.endswith(".json") else f"{name}.json"
        return self.paths.long_term_root / "memory" / filename

    def load_json(self, name: str, default: Any = None) -> Any:
        path = self._json_path(name)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return default

    def save_json(self, name: str, data: Any) -> None:
        path = self._json_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        self.manifest_store.record(name, path)
