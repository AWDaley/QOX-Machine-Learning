"""Explicit, JSON-serializable primitive actions for scripted and future learned agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class Kind(StrEnum):
    WAIT = "wait"
    MOVE = "move"
    GRAZE = "graze"
    HARVEST = "harvest"
    PLANT = "plant"
    COLLECT_WASTE = "collect_waste"
    FERTILIZE = "fertilize"
    MILK = "milk"
    EAT = "eat"
    EAT_CARCASS = "eat_carcass"
    DEPOSIT = "deposit"
    REST = "rest"
    ATTACK = "attack"


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True, slots=True)
class Action:
    kind: Kind = Kind.WAIT
    dx: int = 0
    dy: int = 0

    def __post_init__(self) -> None:
        if self.kind == Kind.MOVE:
            if (self.dx, self.dy) not in DIRECTIONS:
                raise ValueError("Move must be exactly one cardinal tile")
        elif self.dx != 0 or self.dy != 0:
            raise ValueError("Only MOVE can have direction")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        return cls(kind=Kind(data["kind"]), dx=int(data.get("dx", 0)), dy=int(data.get("dy", 0)))
