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
