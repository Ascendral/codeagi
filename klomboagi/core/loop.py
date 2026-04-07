from __future__ import annotations


class _WorkingMemory:
    def __init__(self) -> None:
        self._items: dict[str, dict] = {}

    def update(self, mission_id: str, **kwargs) -> None:
        current = self._items.setdefault(mission_id, {})
        current.update({k: v for k, v in kwargs.items() if v is not None})


class RuntimeLoop:
    """Minimal runtime loop shim so connective.runtime_bridge imports cleanly."""

    def __init__(self, storage) -> None:
        self.storage = storage
        self.working_memory = _WorkingMemory()

    def initialize(self) -> dict[str, object]:
        return {"status": "initialized"}

    def status(self) -> dict[str, object]:
        return {"status": "ready"}

    def run_cycle(self) -> dict[str, object]:
        return {"status": "idle"}
