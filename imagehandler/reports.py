from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BBox:
    """Bounding box in left, top, right, bottom pixel coordinates.

    right/bottom are exclusive, matching PIL crop semantics.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def padded(self, padding: int, width: int, height: int) -> "BBox":
        return BBox(
            max(0, self.left - padding),
            max(0, self.top - padding),
            min(width, self.right + padding),
            min(height, self.bottom + padding),
        )

    def expand(self, amount: int, width: int, height: int) -> "BBox":
        return self.padded(amount, width, height)

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.bottom <= other.top
            or other.bottom <= self.top
        )

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def to_list(self) -> list[int]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass
class OperationReport:
    ok: bool
    operation: str
    source: str | None = None
    backend: str | None = None
    mode: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    boxes: list[BBox] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["boxes"] = [box.to_list() for box in self.boxes]
        return data

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
