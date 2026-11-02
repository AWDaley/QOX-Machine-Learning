"""Read-only, bounded presentation of confirmed QOX world events.

No policy, physics, RNG, or actor state is changed by this module.  Actions
represent *requests*; only emitted World events prove outcomes.  Rendering
may run at any rate without changing the authoritative simulation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from .actions import Action, Kind
from .entities import Event, Species

RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ActorStamp:
    x: int
    y: int
    species: Species


@dataclass(frozen=True, slots=True)
class DisplayEvent:
    tick: int
    kind: str
    label: str
    x: int | None
    y: int | None
    actor_id: int | None
    species: Species | None
    amount: float
    color: RGB
    effect: str
    lifetime: int
    major: bool
    start: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ActionFeedback:
    kind: Kind
    result: str  # confirmed, unconfirmed, or state (no event expected)
    tick: int


@dataclass(frozen=True, slots=True)
class EventStyle:
    label: str
    color: RGB
    effect: str = "spark"
    lifetime: int = 4
    major: bool = False


GREEN = (135, 231, 139)
BLUE = (112, 199, 249)
GOLD = (255, 215, 107)
RED = (254, 120, 115)
PURPLE = (192, 143, 242)
CREAM = (241, 241, 209)
ORANGE = (255, 160, 93)

# Only actual simulator events can create success effects. Nonvisual background
# work (e.g. every metabolism or carcass decay tick) is deliberately excluded.
STYLES: dict[str, EventStyle] = {
    "charge_windup": EventStyle("Winding up charge", ORANGE, "ring", 3, True),
    "charge_hit": EventStyle("Charge hit", RED, "impact", 5, True),
    "charge_resolved": EventStyle("Charge resolved", ORANGE, "trail", 3, True),
    "attack": EventStyle("Ravager attack", RED, "impact", 4, True),
    "nomad_attack": EventStyle("Nomad defended", BLUE, "impact", 4, True),
    "friendly_call": EventStyle("Friendly cattle call", BLUE, "ring", 4, True),
    "scary_call": EventStyle("Scary cattle call", ORANGE, "ring", 4, True),
    "graze": EventStyle("Grazed", GREEN, "leaf", 3),
    "plant": EventStyle("Planted crops", GREEN, "leaf", 4),
    "harvest": EventStyle("Harvested crops", GOLD, "spark", 4, True),
    "collect_waste": EventStyle("Collected manure", PURPLE, "spark", 4),
    "fertilize": EventStyle("Fertilized field", GREEN, "spark", 5, True),
    "deposit": EventStyle("Stored food", GOLD, "spark", 3),
    "milk": EventStyle("Collected milk", CREAM, "spark", 4),
    "eat": EventStyle("Ate food", GREEN, "leaf", 3),
    "eat_waste": EventStyle("Ate waste", PURPLE, "leaf", 3),
    "eat_carcass": EventStyle("Ate carrion", PURPLE, "leaf", 3),
    "grab_carcass": EventStyle("Grabbed carcass", ORANGE, "spark", 4, True),
    "drag_carcass": EventStyle("Dragging carcass", ORANGE, "trail", 3),
    "drop_carcass": EventStyle("Dropped carcass", ORANGE, "spark", 4),
    "enter_refuge": EventStyle("Entered settlement", BLUE, "ring", 5, True),
    "exit_refuge": EventStyle("Left settlement", BLUE, "spark", 4),
    "hibernate": EventStyle("Hibernating", PURPLE, "ring", 5, True),
    "wake": EventStyle("Woke up", PURPLE, "spark", 4),
    "death": EventStyle("Died", RED, "burst", 7, True),
    "birth": EventStyle("Newborn", GREEN, "burst", 7, True),
    "festival_contribution": EventStyle("Festival contribution", GOLD, "spark", 4, True),
    "festival_started": EventStyle("Festival began", GOLD, "ring", 6, True),
    "festival_success": EventStyle("Festival succeeded", GOLD, "burst", 9, True),
    "festival_failed": EventStyle("Festival failed", RED, "burst", 9, True),
}

# One actor may emit multiple events in a tick; each action is confirmed only
# by its matching event(s). CHARGE means wind-up, not a future impact.
CONFIRMING_EVENTS: dict[Kind, frozenset[str]] = {
    Kind.MOVE: frozenset({"move"}),
    Kind.DASH: frozenset({"dash"}),
    Kind.SNEAK: frozenset({"sneak"}),
    Kind.CHARGE: frozenset({"charge_windup"}),
    Kind.GRAB_CARCASS: frozenset({"grab_carcass"}),
    Kind.DROP_CARCASS: frozenset({"drop_carcass"}),
    Kind.HIBERNATE: frozenset({"hibernate"}),
    Kind.WAKE: frozenset({"wake"}),
    Kind.FRIENDLY_CALL: frozenset({"friendly_call"}),
    Kind.SCARY_CALL: frozenset({"scary_call"}),
    Kind.GRAZE: frozenset({"graze"}),
    Kind.HARVEST: frozenset({"harvest"}),
    Kind.PLANT: frozenset({"plant"}),
    Kind.COLLECT_WASTE: frozenset({"collect_waste"}),
    Kind.FERTILIZE: frozenset({"fertilize"}),
    Kind.MILK: frozenset({"milk"}),
    Kind.EAT: frozenset({"eat", "eat_waste"}),
    Kind.EAT_CARCASS: frozenset({"eat_carcass"}),
    Kind.DEPOSIT: frozenset({"deposit"}),
    Kind.ENTER_REFUGE: frozenset({"enter_refuge"}),
    Kind.ATTACK: frozenset({"nomad_attack", "attack"}),
    Kind.CONTRIBUTE_FESTIVAL: frozenset({"festival_contribution"}),
    Kind.START_FARMER_FESTIVAL: frozenset({"festival_started"}),
    Kind.START_NOMAD_FESTIVAL: frozenset({"festival_started"}),
    Kind.START_COMBINED_FESTIVAL: frozenset({"festival_started"}),
}


def capture_actors(world) -> dict[int, ActorStamp]:
    """Read positions and species before a tick, including actors that may die."""
    return {aid: ActorStamp(a.x, a.y, a.species) for aid, a in world.actors.items()}


def classify_event(event: Event, before: Mapping[int, ActorStamp],
                   after: Mapping[int, ActorStamp]) -> DisplayEvent | None:
    """Turn one confirmed physical event into a viewer-only event."""
    style = STYLES.get(event.kind)
    if style is None:
        return None
    actor = after.get(event.actor_id) or before.get(event.actor_id)
    x = event.x if event.x is not None else actor.x if actor is not None else None
    y = event.y if event.y is not None else actor.y if actor is not None else None
    label = style.label
    effect = style.effect
    color = style.color
    major = style.major
    if event.kind == "charge_resolved":
        if event.detail == "hit":
            label, color, major = "Charge connected", RED, True
        elif event.detail == "miss":
            label, color, major = "Charge missed", ORANGE, True
        elif event.detail == "obstacle":
            label, color, major = "Charge hit obstacle", RED, True
        else:
            label = "Charge blocked"
    elif event.kind == "festival_failed":
        label = "Festival failed" + (f" ({event.detail.split(':')[-1]})" if event.detail else "")
    elif event.kind in ("charge_hit", "attack", "nomad_attack"):
        # Damage events are emitted at the attacker's location, so follow the
        # victim for impact effects. A death may remove the victim that tick.
        if event.detail.startswith("target:"):
            try:
                victim_id = int(event.detail.removeprefix("target:"))
            except ValueError:
                victim_id = -1
            victim = after.get(victim_id) or before.get(victim_id)
            if victim is not None:
                x, y = victim.x, victim.y
    elif event.kind == "birth":
        label = "New " + (actor.species.value if actor is not None else "actor")
    if event.amount > 0 and event.kind in ("harvest", "fertilize", "graze", "milk"):
        label += f" +{event.amount:.1f}"
    start = None
    if event.kind in ("drag_carcass", "charge_resolved") and event.actor_id in before:
        a = before[event.actor_id]
        start = (a.x, a.y)
    return DisplayEvent(event.tick, event.kind, label, x, y, event.actor_id,
                        actor.species if actor is not None else None,
                        event.amount, color, effect, style.lifetime, major, start)


class EventObserver:
    """Bounded, read-only observer owned by the viewer, not the world."""

    def __init__(self, *, history_limit: int = 160, effect_limit: int = 120):
        if history_limit < 1 or effect_limit < 1:
            raise ValueError("Event limits must be positive")
        self.history: deque[DisplayEvent] = deque(maxlen=history_limit)
        self.important_history: deque[DisplayEvent] = deque(maxlen=history_limit)
        self.effects: deque[DisplayEvent] = deque(maxlen=effect_limit)
        self.feedback: dict[int, ActionFeedback] = {}
        self.confirmed_actions: dict[int, Action] = {}
        self.tick = 0

    def accept(self, world, actions: Mapping[int, Action],
               before: Mapping[int, ActorStamp]) -> None:
        """Consume the just-completed tick. Never touch RNG or world objects."""
        self.tick = world.tick
        after = capture_actors(world)
        emitted = world.last_events
        confirmed_by_actor: dict[int, set[str]] = {}
        for event in emitted:
            if event.actor_id is not None:
                confirmed_by_actor.setdefault(event.actor_id, set()).add(event.kind)
            cue = classify_event(event, before, after)
            if cue is not None:
                self.history.append(cue)
                if cue.major:
                    self.important_history.append(cue)
                self.effects.append(cue)
        self.effects = deque((cue for cue in self.effects
                              if self.tick - cue.tick < cue.lifetime),
                             maxlen=self.effects.maxlen)
        self.feedback = {}
        self.confirmed_actions = {}
        for actor_id, action in actions.items():
            expected = CONFIRMING_EVENTS.get(action.kind)
            if expected is None:
                result = "state"  # E.g. WAIT or REST has no explicit confirmation.
                self.confirmed_actions[actor_id] = action
            elif expected.intersection(confirmed_by_actor.get(actor_id, ())):
                result = "confirmed"
                self.confirmed_actions[actor_id] = action
            else:
                result = "unconfirmed"
                # Do not draw an action-specific success pose for an action
                # that did not actually happen. Persistent states still draw.
                self.confirmed_actions[actor_id] = Action()
            self.feedback[actor_id] = ActionFeedback(action.kind, result, self.tick)

    def recent(self, count: int = 6, *, important_only: bool = True,
               species: Species | None = None,
               actor_id: int | None = None) -> list[DisplayEvent]:
        """Read a filtered view without mutating the underlying event history."""
        if count < 0:
            raise ValueError("Event count cannot be negative")
        if count == 0:
            return []
        result = []
        source = self.important_history if important_only else self.history
        for item in reversed(source):
            if species is not None and item.species != species:
                continue
            if actor_id is not None and item.actor_id != actor_id:
                continue
            result.append(item)
            if len(result) >= count:
                break
        return result

    def export_json(self, path) -> None:
        """Export the bounded, confirmed visual event stream (not all world.events)."""
        from dataclasses import asdict
        import json
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema_version": 1,
            "note": "Bounded confirmed events from the viewer observation window",
            "events": [asdict(event) for event in self.history],
        }, indent=2), encoding="utf-8")

    def active(self) -> tuple[DisplayEvent, ...]:
        return tuple(cue for cue in self.effects
                     if self.tick - cue.tick < cue.lifetime)
