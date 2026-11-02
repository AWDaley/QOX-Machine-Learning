"""Immutable, local-policy decision inspection at the pre-action world snapshot.

This describes what a policy *received* and *selected*, not why its hidden
neurons reached that choice. Scripted actors receive observations but no
fabricated neural probabilities. Reading never advances the policy RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .actions import Action
from .entities import Species
from .perception import action_catalog, features_for, observe


@dataclass(frozen=True, slots=True)
class Choice:
    label: str
    probability: float
    legal: bool
    chosen: bool = False


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    tick: int  # World tick *before* this action was requested.
    actor_id: int
    species: Species
    observations: tuple[tuple[str, float], ...]
    choices: tuple[Choice, ...]
    action: str = "Not yet acted"
    policy: str = "scripted or random"
    outcome: str = "pending"

    def top_actions(self, count: int = 5) -> tuple[Choice, ...]:
        return tuple(sorted((c for c in self.choices if c.legal),
                            key=lambda c: c.probability, reverse=True)[:count])


def _flags(controller, species: Species) -> dict[str, bool]:
    """Match exact input schema of the actor's actual loaded checkpoint."""
    if controller is None:
        return {"cow_calls": False, "nomad_v2": False, "festivals": False}
    if hasattr(controller, "models"):
        model = controller.models[species]
    elif hasattr(controller, "cow") and hasattr(controller, "nomad"):
        model = controller.cow.model if species == Species.COW else (
            controller.nomad.model if species == Species.NOMAD else None)
    elif hasattr(controller, "species") and controller.species != species:
        model = None
    elif hasattr(controller, "model") and (
            not hasattr(controller, "species") or controller.species == species):
        model = controller.model
    else:
        model = None
    return {"cow_calls": bool(getattr(model, "cow_calls", False)),
            "nomad_v2": bool(getattr(model, "nomad_v2", False)),
            "festivals": bool(getattr(model, "festival_actions", False) or
                              (getattr(controller, "species", None) == species and
                               hasattr(controller, "scenario")))}


def action_label(action: Action) -> str:
    if action.dx or action.dy:
        direction = {(0, -1): "N", (1, 0): "E", (0, 1): "S", (-1, 0): "W"}.get(
            (action.dx, action.dy), f"({action.dx},{action.dy})")
        return f"{action.kind.value} {direction}"
    return action.kind.value


def capture_decision(world, actor_id: int, controller=None, *,
                     policy_label: str | None = None) -> DecisionSnapshot | None:
    """Snapshot observations and optional neural probabilities before acting."""
    actor = world.actors.get(actor_id)
    if actor is None:
        return None
    opts = _flags(controller, actor.species)
    names = features_for(actor.species, **opts)
    data = observe(world, actor, **opts)
    numbers = data["observation"].tolist()
    mask = data["action_mask"].tolist()
    actions = action_catalog(actor.species, festivals=opts["festivals"])
    if len(numbers) != len(names) or len(mask) != len(actions):
        raise ValueError("Observation or action contract mismatch during inspection")
    policy = policy_label or "scripted or random"
    probs = None
    if controller is not None:
        try:
            probs = controller.probabilities(world, actor_id)
            policy = "neural policy (frozen inference)"
        except (ValueError, KeyError, AttributeError):
            # Other species may use scripted behavior in single-policy mode.
            pass
    if probs is not None and len(probs) != len(actions):
        raise ValueError("Neural policy output does not match its action catalog")
    choices = tuple(Choice(action_label(a), float(probs[i]) if probs else 0.0,
                           bool(mask[i])) for i, a in enumerate(actions))
    return DecisionSnapshot(world.tick, actor_id, actor.species,
                            tuple((key, float(num)) for key, num in zip(names, numbers, strict=True)),
                            choices, policy=policy)


def with_outcome(decision: DecisionSnapshot, action: Action | None,
                 result: str = "unknown") -> DecisionSnapshot:
    """Bind actual requested action to prior snapshot, never recompute inputs."""
    label = action_label(action) if action is not None else "No action"
    return replace(decision, action=label, outcome=result,
                   choices=tuple(replace(c, chosen=(c.label == label))
                                 for c in decision.choices))
