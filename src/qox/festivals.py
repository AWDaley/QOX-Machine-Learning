"""Festival rules: explicit contribution escrow, local attendance, finite duration.

This module owns *rules*, not decision-making. A policy must choose whether to
commit real food, propose a celebration, attend, defend, or retreat.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .entities import Actor, Species

if TYPE_CHECKING:
    from .world import World


class FestivalKind(StrEnum):
    FARMER = "farmer"
    NOMAD = "nomad"
    COMBINED = "combined"


@dataclass(slots=True)
class Camp:
    id: int
    x: int
    y: int
    festival_milk: float = 0.0
    cooldown_until: int = 0


@dataclass(slots=True)
class Festival:
    id: int
    kind: FestivalKind
    site_kind: str
    site_id: int
    x: int
    y: int
    start_tick: int
    end_tick: int
    proposer_id: int
    grain_spent: float
    milk_spent: float


@dataclass(frozen=True, slots=True)
class FestivalResult:
    id: int
    kind: FestivalKind
    tick: int
    succeeded: bool
    participants: tuple[int, ...]
    reason: str


def distance(actor: Actor, x: int, y: int) -> int:
    return abs(actor.x - x) + abs(actor.y - y)


def camp_for(world: World, actor: Actor) -> Camp | None:
    if actor.species != Species.NOMAD:
        return None
    return next((c for c in world.camps if c.id == actor.band_id), None)


def nearby_camp(world: World, actor: Actor) -> Camp | None:
    camp = camp_for(world, actor)
    return (camp if camp is not None and
            distance(actor, camp.x, camp.y) <= world.config.festival_site_radius
            else None)


def contribution_target(world: World, actor: Actor):
    """Prefer settlement contribution for nomads at an actual meeting site.

    Farmer settlement contributions are separate from survival stores; nomads
    can contribute milk to a camp or to a settlement they physically visit.
    """
    if not world.config.festival_enabled or actor.sheltered:
        return None
    if actor.species == Species.FARMER:
        return world._nearby_settlement(actor)
    if actor.species == Species.NOMAD:
        settlement = next((s for s in world.settlements
                           if distance(actor, s.x, s.y) <= world.config.festival_site_radius),
                          None)
        # Nomads meeting farmers can invest milk in a combined settlement
        # festival; otherwise their milk belongs to their own camp.
        if settlement is not None and any(
                a.species == Species.FARMER and a.home_id == settlement.id
                for a in world.actors.values()):
            return settlement
        return nearby_camp(world, actor)
    return None


def site_for(world: World, actor: Actor, kind: FestivalKind):
    if not world.config.festival_enabled or actor.sheltered:
        return None
    if kind == FestivalKind.FARMER:
        return (world._nearby_settlement(actor)
                if actor.species == Species.FARMER else None)
    if kind == FestivalKind.NOMAD:
        return nearby_camp(world, actor)
    if actor.species not in (Species.FARMER, Species.NOMAD):
        return None
    return next((s for s in world.settlements
                 if distance(actor, s.x, s.y) <= world.config.festival_site_radius
                 and (actor.species != Species.FARMER or actor.home_id == s.id)), None)


def participants(world: World, kind: FestivalKind, x: int, y: int) -> tuple[int, ...]:
    radius = world.config.festival_attendance_radius
    return tuple(sorted(a.id for a in world.actors.values()
                        if not a.sheltered and a.health >= 25
                        and a.species in (Species.FARMER, Species.NOMAD)
                        and distance(a, x, y) <= radius))


def sufficient_attendance(world: World, kind: FestivalKind, x: int, y: int,
                          site_id: int) -> bool:
    attending = [world.actors[i] for i in participants(world, kind, x, y)]
    farmers = sum(a.species == Species.FARMER and a.home_id == site_id
                  for a in attending)
    nomads = sum(a.species == Species.NOMAD and
                 (kind == FestivalKind.COMBINED or a.band_id == site_id)
                 for a in attending)
    cows = sum(a.species == Species.COW and a.health >= 25 and
               distance(a, x, y) <= world.config.festival_cattle_radius
               for a in world.actors.values())
    if kind == FestivalKind.FARMER:
        return farmers >= world.config.festival_farmers_required
    if kind == FestivalKind.NOMAD:
        return (nomads >= world.config.festival_nomads_required and
                cows >= world.config.festival_cows_required)
    return (farmers >= world.config.festival_farmers_required and
            nomads >= world.config.festival_nomads_required and
            cows >= world.config.festival_cows_required)


def requirements(world: World, kind: FestivalKind) -> tuple[float, float]:
    c = world.config
    if kind == FestivalKind.FARMER:
        return c.festival_farmer_grain, 0.0
    if kind == FestivalKind.NOMAD:
        return 0.0, c.festival_nomad_milk
    return c.festival_combined_grain, c.festival_combined_milk


def can_start(world: World, actor: Actor, kind: FestivalKind) -> bool:
    site = site_for(world, actor, kind)
    if site is None or world.tick < site.cooldown_until:
        return False
    site_kind = "camp" if kind == FestivalKind.NOMAD else "settlement"
    if any(f.site_kind == site_kind and f.site_id == site.id
           for f in world.active_festivals.values()):
        return False
    grain, milk = requirements(world, kind)
    return (getattr(site, "festival_food", 0) + 1e-8 >= grain
            and site.festival_milk + 1e-8 >= milk
            and sufficient_attendance(world, kind, site.x, site.y, site.id))


def start(world: World, actor: Actor, kind: FestivalKind) -> bool:
    """Spend escrow exactly once, then begin a multi-tick public event."""
    if not can_start(world, actor, kind):
        return False
    site = site_for(world, actor, kind)
    assert site is not None
    grain, milk = requirements(world, kind)
    if grain:
        site.festival_food = max(0.0, site.festival_food - grain)
    if milk:
        site.festival_milk = max(0.0, site.festival_milk - milk)
    festival = Festival(id=world.next_festival_id, kind=kind,
                        site_kind="camp" if kind == FestivalKind.NOMAD else "settlement",
                        site_id=site.id, x=site.x, y=site.y,
                        start_tick=world.tick,
                        end_tick=world.tick + world.config.festival_duration,
                        proposer_id=actor.id, grain_spent=grain, milk_spent=milk)
    world.next_festival_id += 1
    world.active_festivals[festival.id] = festival
    site.cooldown_until = festival.end_tick + world.config.festival_cooldown
    world.emit_at("festival_started", festival.x, festival.y, actor.id,
                  detail=f"{festival.id}:{festival.kind.value}")
    return True


def festival_cue(world: World, observer: Actor) -> tuple[float, float, float]:
    """Bounded, directional smell/noise from existing celebrations only.

    Not an exact global coordinate; the signal disappears beyond its radius.
    """
    strongest = (0.0, 0.0, 0.0)
    radius = world.config.festival_signal_radius
    for event in world.active_festivals.values():
        d = distance(observer, event.x, event.y)
        if d > radius:
            continue
        scale = 1.0 if event.kind != FestivalKind.COMBINED else 1.45
        strength = scale * (1.0 - d / (radius + 1.0))
        if strength > strongest[2]:
            strongest = (max(-1, min(1, (event.x - observer.x) / radius)),
                         max(-1, min(1, (event.y - observer.y) / radius)),
                         min(1.0, strength))
    return strongest


def advance(world: World) -> None:
    """Called AFTER action resolution and metabolism, once per world tick."""
    for festival in list(sorted(world.active_festivals.values(), key=lambda x: x.id)):
        attendees = participants(world, festival.kind, festival.x, festival.y)
        quorum = sufficient_attendance(world, festival.kind, festival.x,
                                        festival.y, festival.site_id)
        at_gate = any(a.species == Species.RAVAGER and
                      distance(a, festival.x, festival.y) <= 1
                      for a in world.actors.values())
        reason = "predator_at_site" if at_gate else "attendance_lost" if not quorum else ""
        if not reason and world.tick >= festival.end_tick:
            reason = "completed"
        # Signal each active tick at the actual festival site; do not conjure predators.
        world.emit_at("festival_noise", festival.x, festival.y,
                      amount=1.45 if festival.kind == FestivalKind.COMBINED else 1.0,
                      detail=f"{festival.id}:{festival.kind.value}")
        if reason:
            success = reason == "completed"
            world.festival_results.append(FestivalResult(
                festival.id, festival.kind, world.tick, success,
                tuple(sorted(set(attendees) | {festival.proposer_id})), reason))
            world.emit_at("festival_success" if success else "festival_failed",
                          festival.x, festival.y,
                          amount=festival.grain_spent + festival.milk_spent,
                          detail=f"{festival.id}:{festival.kind.value}:{reason}")
            del world.active_festivals[festival.id]
