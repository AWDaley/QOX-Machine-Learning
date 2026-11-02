"""M12: outcome-driven VFX, action honesty, bounded history, replay integrity."""

from qox.actions import Action, Kind
from qox.agents import actions_for
from qox.config import SimulationConfig
from qox.entities import Event, Species
from qox.viewer import _focus_event
from qox.viewer_events import (ActorStamp, DisplayEvent, EventObserver,
                               capture_actors, classify_event)
from qox.viewer_layout import Camera, layout_for
from qox.world import World


def _actor(world, species):
    return next(a for a in world.actors.values() if a.species == species)


def test_confirmed_grazing_animates_and_reports_honest_outcome():
    world = World(SimulationConfig(seed=12))
    cow = _actor(world, Species.COW)
    tile = world.grid.get(*cow.position)
    tile.planted = False
    tile.grass = 5.0
    before = capture_actors(world)
    observer = EventObserver()
    order = {cow.id: Action(Kind.GRAZE)}
    world.step(order)
    observer.accept(world, order, before)
    assert observer.feedback[cow.id].result == "confirmed"
    assert observer.confirmed_actions[cow.id] == order[cow.id]
    assert any(c.kind == "graze" and c.actor_id == cow.id for c in observer.active())
    assert any(c.kind == "graze" for c in observer.recent(6, important_only=False))
    assert not any(c.kind == "graze" for c in observer.recent(6, important_only=True))


def test_failed_grazing_does_not_show_success_sprite_or_event():
    world = World(SimulationConfig(seed=13))
    cow = _actor(world, Species.COW)
    tile = world.grid.get(*cow.position)
    tile.grass = tile.crop = 0.0
    tile.planted = False
    before = capture_actors(world)
    observer = EventObserver()
    order = {cow.id: Action(Kind.GRAZE)}
    world.step(order)
    observer.accept(world, order, before)
    assert observer.feedback[cow.id].result == "unconfirmed"
    assert observer.confirmed_actions[cow.id] == Action(Kind.WAIT)
    assert not any(c.kind == "graze" and c.actor_id == cow.id for c in observer.active())


def test_failed_charge_is_not_mislabeled_as_a_hit():
    world = World(SimulationConfig(seed=14))
    ravager = _actor(world, Species.RAVAGER)
    ravager.charge_cooldown = 10
    before = capture_actors(world)
    observer = EventObserver()
    order = {ravager.id: Action(Kind.CHARGE, 1, 0)}
    world.step(order)
    observer.accept(world, order, before)
    assert observer.feedback[ravager.id].result == "unconfirmed"
    assert not any(c.kind.startswith("charge") and c.actor_id == ravager.id
                   for c in observer.active())


def test_impact_effect_is_at_victim_not_attacker():
    before = {1: ActorStamp(2, 4, Species.RAVAGER),
              2: ActorStamp(3, 4, Species.COW)}
    event = Event(tick=2, kind="charge_hit", actor_id=1, amount=55,
                  detail="target:2", x=2, y=4)
    cue = classify_event(event, before, before)
    assert cue is not None
    assert (cue.x, cue.y) == (3, 4)
    assert cue.effect == "impact" and cue.major


def test_death_display_keeps_removed_actor_species_and_position():
    before = {8: ActorStamp(7, 9, Species.FARMER)}
    cue = classify_event(Event(7, "death", 8, 0, "farmer", 7, 9), before, {})
    assert cue is not None
    assert (cue.x, cue.y) == (7, 9)
    assert cue.species == Species.FARMER
    assert cue.major


def test_charge_miss_vs_hit_and_festival_failure_are_not_confused():
    before = {8: ActorStamp(4, 4, Species.RAVAGER)}
    miss = classify_event(Event(9, "charge_resolved", 8, detail="miss", x=6, y=4),
                          before, {8: ActorStamp(6, 4, Species.RAVAGER)})
    assert miss.label == "Charge missed"
    assert miss.start == (4, 4)
    assert miss.effect == "trail"
    festival = classify_event(Event(10, "festival_failed", detail="3:combined:predator_at_site",
                                    x=7, y=5), before, {})
    assert festival.label.startswith("Festival failed") and festival.major
    assert festival.color != miss.color


def test_event_history_and_effect_lifetimes_are_bounded():
    class FakeWorld:
        tick = 0
        actors = {}
        last_events = []

    world = FakeWorld()
    observer = EventObserver(history_limit=12, effect_limit=7)
    for tick in range(1, 900):
        world.tick = tick
        world.last_events = [Event(tick, "festival_started", None, x=3, y=2)] * 4
        observer.accept(world, {}, {})
        assert len(observer.history) <= 12
        assert len(observer.important_history) <= 12
        assert len(observer.effects) <= 7
    assert all(observer.tick - event.tick < event.lifetime for event in observer.active())
    assert len(observer.recent(5)) == 5
    assert observer.recent(0) == []


def test_no_tick_time_consumed_by_rendering_or_history_filters():
    cfg = SimulationConfig(seed=101, population_dynamics_enabled=True,
                           reproduction_enabled=True, festival_enabled=True)
    observed = World(cfg)
    control = World(cfg)
    viewer_events = EventObserver()
    for _ in range(45):
        before = capture_actors(observed)
        actions_a = actions_for(observed, "scripted")
        actions_b = actions_for(control, "scripted")
        assert actions_a == actions_b
        observed.step(actions_a)
        control.step(actions_b)
        viewer_events.accept(observed, actions_a, before)
        viewer_events.recent(5, important_only=True)
        viewer_events.recent(5, important_only=False)
        viewer_events.active()
        assert observed.state_hash() == control.state_hash()


def test_clicking_an_event_only_moves_camera():
    layout = layout_for(1250, 850)
    camera = Camera(40, 40, layout.map)
    world_position = (34, 31)
    event = DisplayEvent(20, "death", "Died", 34, 31, 9,
                         Species.COW, 0.0, (255, 0, 0), "burst", 7, True)
    _focus_event(camera, layout.map, event)
    ox, oy = camera.origin(layout.map)
    cx = ox + world_position[0] * camera.tile + camera.tile // 2
    cy = oy + world_position[1] * camera.tile + camera.tile // 2
    assert layout.map.contains((cx, cy))


def test_unknown_and_background_events_are_not_fake_visual_actions():
    before = {1: ActorStamp(2, 2, Species.FARMER)}
    assert classify_event(Event(2, "carcass_decay", amount=0.04), before, {}) is None
    assert classify_event(Event(2, "charge_resolved", 1, detail="hit", x=2, y=2),
                          before, before).label == "Charge connected"


def test_high_priority_feed_survives_a_flood_of_grazing_effects():
    class FakeWorld:
        tick = 0
        actors = {}
        last_events = []

    world = FakeWorld()
    observer = EventObserver(history_limit=10)
    world.tick = 1
    world.last_events = [Event(1, "festival_success", None, x=5, y=5)]
    observer.accept(world, {}, {})
    for tick in range(2, 82):
        world.tick = tick
        world.last_events = [Event(tick, "graze", 99, 1.0, "grass", 5, 5)]
        observer.accept(world, {}, {})
    assert not any(item.kind == "festival_success" for item in observer.history)
    assert observer.recent(2, important_only=True)[0].kind == "festival_success"
