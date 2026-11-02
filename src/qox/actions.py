"""Explicit, JSON-serializable primitive actions for scripted and future learned agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class Kind(StrEnum):
    WAIT = "wait"
    MOVE = "move"
    DASH = "dash"  # Nomad-only two-cell burst.
    SNEAK = "sneak"  # Ravager-only quiet, tiring, half-rate movement.
    CHARGE = "charge"  # Ravager-only committed high-speed directional attack.
    GRAB_CARCASS = "grab_carcass"
    DROP_CARCASS = "drop_carcass"
    HIBERNATE = "hibernate"
    WAKE = "wake"
    FRIENDLY_CALL = "friendly_call"
    SCARY_CALL = "scary_call"
    GRAZE = "graze"
    HARVEST = "harvest"
    PLANT = "plant"
    COLLECT_WASTE = "collect_waste"
    FERTILIZE = "fertilize"
    MILK = "milk"
    EAT = "eat"
    EAT_CARCASS = "eat_carcass"
    DEPOSIT = "deposit"
    ENTER_REFUGE = "enter_refuge"
    REST = "rest"
    ATTACK = "attack"
    CONTRIBUTE_FESTIVAL = "contribute_festival"
    START_FARMER_FESTIVAL = "start_farmer_festival"
    START_NOMAD_FESTIVAL = "start_nomad_festival"
    START_COMBINED_FESTIVAL = "start_combined_festival"


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True, slots=True)
class Action:
    kind: Kind = Kind.WAIT
    dx: int = 0
    dy: int = 0

    def __post_init__(self) -> None:
        if self.kind in (Kind.MOVE, Kind.DASH, Kind.SNEAK, Kind.CHARGE):
            if (self.dx, self.dy) not in DIRECTIONS:
                raise ValueError("Move must be exactly one cardinal tile")
        elif self.dx != 0 or self.dy != 0:
            raise ValueError("Only directional movement/charge actions can have direction")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        return cls(kind=Kind(data["kind"]), dx=int(data.get("dx", 0)), dy=int(data.get("dy", 0)))
