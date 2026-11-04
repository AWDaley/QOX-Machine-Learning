"""M16k: actor-local agricultural event history for opt-in farmer policies.

Only events *performed by the controlled farmer* enter this memory. There is
no global crop inventory, waypoint, scripted work phase or hidden state leak.
The neural policy decides how to use the history; it never commands actions.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..entities import Event
from ..world import World

TASK_EVENTS = ("plant", "harvest", "collect_waste", "fertilize", "deposit")
HISTORY_WINDOW = 32
FARMER_TASK_HISTORY_FEATURES = (
    *(f"recent_{kind}" for kind in TASK_EVENTS),
    "since_harvest", "since_fertilize", "since_deposit",
    *(f"last_{kind}" for kind in TASK_EVENTS),
    "since_last_success",
)
FARMER_HISTORY_MODES = ("none", "masked", "events")


@dataclass
class FarmerTaskHistory:
    """Compact, episode-local and deterministic memory of verified own work."""

    tick: int = 0
    successes: deque[tuple[int, str]] = field(default_factory=deque)
    last_tick: dict[str, int] = field(default_factory=dict)
    last_kind: str | None = None

    @classmethod
    def from_recorded_events(cls, events: Iterable[Event], actor_id: int,
                             through_tick: int) -> FarmerTaskHistory:
        """Rehydrate local memory after a world load from the actor's own log.

        This is history reconstruction, not a read of the world's current
        hidden resources; future events and other actors are excluded.
        """
        if through_tick < 0:
            raise ValueError("World tick must not be negative")
        memory = cls(tick=through_tick)
        for event in events:
            if (event.actor_id != actor_id or event.amount <= 0 or
                    event.kind not in TASK_EVENTS or not 0 <= event.tick <= through_tick):
                continue
            memory.last_tick[event.kind] = event.tick
            memory.last_kind = event.kind
            if through_tick - event.tick < HISTORY_WINDOW:
                memory.successes.append((event.tick, event.kind))
        return memory

    def advance(self, events: Iterable[Event], actor_id: int, *, steps: int = 1) -> None:
        """Advance by observed decisions; ignore other actors' events."""
        if steps < 1:
            raise ValueError("Task history must advance at least one step")
        self.tick += steps
        for event in events:
            if (event.actor_id != actor_id or event.amount <= 0 or
                    event.kind not in TASK_EVENTS):
                continue
            self.successes.append((self.tick, event.kind))
            self.last_tick[event.kind] = self.tick
            self.last_kind = event.kind
        while self.successes and self.tick - self.successes[0][0] >= HISTORY_WINDOW:
            self.successes.popleft()

    def features(self) -> np.ndarray:
        counts = Counter(kind for _, kind in self.successes)

        def age(kind: str | None) -> float:
            if kind is None or kind not in self.last_tick:
                return 1.0
            return min(1.0, (self.tick - self.last_tick[kind]) / HISTORY_WINDOW)

        latest_age = age(self.last_kind)
        values = (
            *(min(1.0, counts[kind] / 4) for kind in TASK_EVENTS),
            age("harvest"), age("fertilize"), age("deposit"),
            *(float(self.last_kind == kind) for kind in TASK_EVENTS),
            latest_age,
        )
        return np.asarray(values, dtype=np.float32)


def farmer_policy_observation(observation: np.ndarray, mode: str,
                              history: FarmerTaskHistory | None = None) -> np.ndarray:
    """Append history only for opt-in v2 policies; v1 input remains exact."""
    if mode not in FARMER_HISTORY_MODES:
        raise ValueError("Unknown farmer task-history mode")
    if mode == "none":
        return observation
    if mode == "events" and history is None:
        raise ValueError("Actor-local history is required for event-memory policy")
    tail = (history.features() if mode == "events" else
            np.zeros(len(FARMER_TASK_HISTORY_FEATURES), dtype=np.float32))
    return np.concatenate((observation, tail))


@dataclass
class FarmerHistoryCache:
    """Keep farmer history consistent in both standalone and desktop worlds.

    One completed world tick is consumed at most once per actor; repeated UI
    probability previews cannot add phantom history events. A new world or
    rewind resets the cache, never reuses a previous world's private memory.
    """

    world_identity: int | None = None
    histories: dict[int, FarmerTaskHistory] = field(default_factory=dict)
    last_ticks: dict[int, int] = field(default_factory=dict)

    def for_actor(self, world: World, actor_id: int) -> FarmerTaskHistory:
        identity = id(world)
        if identity != self.world_identity:
            self.histories.clear()
            self.last_ticks.clear()
            self.world_identity = identity
        if actor_id not in self.histories or world.tick < self.last_ticks[actor_id]:
            self.histories[actor_id] = FarmerTaskHistory.from_recorded_events(
                world.events, actor_id, world.tick)
            self.last_ticks[actor_id] = world.tick
        previous = self.last_ticks[actor_id]
        if world.tick > previous:
            self.histories[actor_id].advance(world.last_events, actor_id,
                                              steps=world.tick - previous)
            self.last_ticks[actor_id] = world.tick
        return self.histories[actor_id]
