"""Plain state objects; no decision logic lives on an entity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Species(StrEnum):
    FARMER = "farmer"
    COW = "cow"
    NOMAD = "nomad"
    SCAVENGER = "scavenger"
    RAVAGER = "ravager"


@dataclass(slots=True)
class Tile:
    fertility: float
    grass: float
    crop: float = 0.0
    planted: bool = False
    waste: float = 0.0
    cover: float = 0.0
    passable: bool = True


@dataclass(slots=True)
class Actor:
    id: int
    species: Species
    x: int
    y: int
    energy: float = 70.0
    stamina: float = 100.0
    health: float = 100.0
    age: int = 0
    carried_food: float = 0.0
    carried_waste: float = 0.0
    milk: float = 0.0
    dung_pending: float = 0.0
    home_id: int | None = None
    sheltered: bool = False
    friendly_cooldown: int = 0
    scary_cooldown: int = 0

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(slots=True)
class Settlement:
    id: int
    x: int
    y: int
    stored_food: float = 8.0
    refuge_capacity: int = 12


@dataclass(slots=True)
class Carcass:
    id: int
    x: int
    y: int
    meat: float
    origin_species: Species


@dataclass(frozen=True, slots=True)
class Event:
    tick: int
    kind: str
    actor_id: int | None = None
    amount: float = 0.0
    detail: str = ""
    x: int | None = None
    y: int | None = None


@dataclass(frozen=True, slots=True)
class CallSignal:
    """Perceptible local sound. It changes observations, never cow positions."""

    source_id: int
    kind: str
    x: int
    y: int
    tick: int
    radius: int
    intensity: float = 1.0
