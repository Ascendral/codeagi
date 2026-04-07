from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Drive(str, Enum):
    CURIOSITY = "curiosity"
    PREDICTION = "prediction"
    COMPETENCE = "competence"


@dataclass
class DriveBank:
    values: dict[Drive, float] = field(default_factory=lambda: {drive: 0.5 for drive in Drive})

    def boost(self, drive: Drive, amount: float) -> None:
        self.values[drive] = min(1.0, self.values.get(drive, 0.5) + amount)

    def satisfy(self, drive: Drive, amount: float) -> None:
        self.values[drive] = max(0.0, self.values.get(drive, 0.5) - amount)

    def to_dict(self) -> dict[str, float]:
        return {drive.value: round(value, 3) for drive, value in self.values.items()}


class AttentionController:
    def __init__(self) -> None:
        self.drives = DriveBank()
        self._concepts: list[dict] = []

    def add_concept(self, name: str, payload: dict, priority: float = 0.5, relevant_drives: list[Drive] | None = None) -> None:
        self._concepts.append(
            {
                "name": name,
                "payload": payload,
                "priority": priority,
                "relevant_drives": [drive.value for drive in (relevant_drives or [])],
            }
        )

    def on_contradiction(self) -> None:
        self.drives.boost(Drive.CURIOSITY, 0.2)

    def on_prediction_correct(self) -> None:
        self.drives.boost(Drive.COMPETENCE, 0.2)

    def on_gap_found(self) -> None:
        self.drives.boost(Drive.CURIOSITY, 0.1)

    def tick(self) -> None:
        for drive in Drive:
            value = self.drives.values.get(drive, 0.5)
            self.drives.values[drive] = max(0.0, min(1.0, value * 0.98))

    def stats(self) -> dict:
        return {
            "drives": self.drives.to_dict(),
            "concepts": len(self._concepts),
        }
