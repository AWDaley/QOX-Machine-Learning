"""Replay action traces against their expected world-state hashes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from .actions import Action
from .agents import actions_for
from .config import SimulationConfig
from .world import World


@dataclass
class Trace:
    config: dict
    frames: list[dict] = field(default_factory=list)
    schema_version: int = 1

    def append(self, world: World, orders: dict[int, Action]) -> None:
        self.frames.append({
            "tick": world.tick,
            "actions": {str(aid): action.to_dict() for aid, action in sorted(orders.items())},
            "state_hash": world.state_hash(),
        })

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported trace schema")
        return cls(config=data["config"], frames=data["frames"])


def run(config: SimulationConfig, ticks: int, mode: str = "scripted",
        record: bool = False) -> tuple[World, Trace | None]:
    if ticks < 0:
        raise ValueError("ticks cannot be negative")
    world = World(config)
    trace = Trace(config.as_dict()) if record else None
    for _ in range(ticks):
        orders = actions_for(world, mode)
        world.step(orders)
        if trace:
            trace.append(world, orders)
    return world, trace


def replay(trace: Trace) -> World:
    world = World(SimulationConfig(**trace.config))
    for index, frame in enumerate(trace.frames, start=1):
        if frame["tick"] != index:
            raise ValueError(f"Trace tick mismatch at frame {index}")
        actions = {int(aid): Action.from_dict(value)
                   for aid, value in frame["actions"].items()}
        world.step(actions)
        if world.state_hash() != frame["state_hash"]:
            raise ValueError(f"Replay divergence at tick {index}")
    return world
