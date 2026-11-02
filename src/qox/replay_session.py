"""Hash-verified, seekable QOX action-trace replay with bounded checkpoints.

The world is advanced only by actions already recorded in the trace. Deep-copy
snapshots are an in-memory seek accelerator, not new simulation logic.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

from .actions import Action
from .config import SimulationConfig
from .trace import Trace
from .world import World


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    tick: int
    kind: str
    actor_id: int | None


MAJOR = frozenset({"birth", "death", "charge_hit", "charge_resolved",
                   "festival_started", "festival_success", "festival_failed",
                   "enter_refuge", "friendly_call", "scary_call"})


class ReplayCursor:
    """Validate every replayed transition and allow backward/forward seeking.

    Uses limited checkpoint snapshots rather than attempting to deserialize
    World.snapshot() as a complete live world (that dump excludes RNG state).
    """

    def __init__(self, trace: Trace, *, checkpoint_every: int = 64,
                 max_checkpoints: int = 24):
        if checkpoint_every < 1 or max_checkpoints < 2:
            raise ValueError("Invalid replay cache settings")
        if trace.schema_version != 1:
            raise ValueError("Unsupported trace version")
        self.trace = trace
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints = max_checkpoints
        self.world = World(SimulationConfig(**trace.config))
        self.position = 0
        self.last_actions: dict[int, Action] = {}
        self._cache: OrderedDict[int, World] = OrderedDict([(0, deepcopy(self.world))])
        self._events_by_tick: dict[int, tuple[ReplayEvent, ...]] = {}
        self._all_events_indexed = False

    @property
    def length(self) -> int:
        return len(self.trace.frames)

    def _store(self, world: World) -> None:
        if world.tick % self.checkpoint_every:
            return
        self._cache[world.tick] = deepcopy(world)
        self._cache.move_to_end(world.tick)
        while len(self._cache) > self.max_checkpoints:
            # Keep tick zero, and evict the oldest *other* checkpoint.
            oldest = next((k for k in self._cache if k != 0), None)
            if oldest is None:
                break
            del self._cache[oldest]

    def _advance(self, world: World, position: int) -> dict[int, Action]:
        frame = self.trace.frames[position]
        if frame["tick"] != position + 1:
            raise ValueError(f"Trace tick mismatch at frame {position + 1}")
        actions = {int(aid): Action.from_dict(value)
                   for aid, value in frame["actions"].items()}
        world.step(actions)
        if world.state_hash() != frame["state_hash"]:
            raise ValueError(f"Replay divergence at tick {world.tick}")
        self._events_by_tick[world.tick] = tuple(
            ReplayEvent(world.tick, e.kind, e.actor_id)
            for e in world.last_events if e.kind in MAJOR)
        self._store(world)
        return actions

    def seek(self, tick: int) -> World:
        if not 0 <= tick <= self.length:
            raise IndexError("Replay tick outside the trace")
        if tick == self.position:
            return self.world
        origins = [v for v in self._cache if v <= tick]
        if self.position <= tick:
            origins.append(self.position)
        start = max(origins)
        if start == self.position:
            world = self.world  # fast sequential playback
        else:
            world = deepcopy(self._cache[start])
        for frame_index in range(start, tick):
            self._advance(world, frame_index)
        self.world = world
        self.position = tick
        self.last_actions = ({int(aid): Action.from_dict(value)
                              for aid, value in self.trace.frames[tick - 1]["actions"].items()}
                             if tick else {})
        return world

    def major_events(self) -> tuple[ReplayEvent, ...]:
        if not self._all_events_indexed:
            old_position = self.position
            self.seek(self.length)
            self.seek(old_position)
            self._all_events_indexed = True
        return tuple(ev for tick in sorted(self._events_by_tick)
                     for ev in self._events_by_tick[tick])

    def next_event_tick(self, *, forward: bool = True) -> int | None:
        ticks = sorted({e.tick for e in self.major_events()})
        if forward:
            return next((t for t in ticks if t > self.position), None)
        return next((t for t in reversed(ticks) if t < self.position), None)
