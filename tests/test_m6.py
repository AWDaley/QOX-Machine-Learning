"""M6 ravager mechanics: reproducible physics, not preprogrammed hunting AI."""
import pytest

from qox.actions import Action, Kind
from qox.config import SimulationConfig
from qox.entities import Carcass, Species
from qox.envs.core import QoxCoreEnv
from qox.perception import ACTIONS, _nearest_actor, legal_action_mask, observation_size, observe
from qox.trace import Trace, replay
from qox.world import World


def small_world(*, cow: bool = True, nomad: bool = False) -> tuple[World, object, object | None]:
    config = SimulationConfig(width=20, height=14, farmers=0, cows=int(cow),
                              nomads=int(nomad), scavengers=0, ravagers=1,
                              ecology_interval=9)
    world = World(config)
    pred = next(a for a in world.actors.values() if a.species == Species.RAVAGER)
    pred.x, pred.y = 5, 5
    target = next((a for a in world.actors.values() if a.species == Species.COW), None)
    if target:
        target.x, target.y = 8, 5
    for tile in world.grid.tiles:
        tile.cover = 0.0
    world.check_invariants()
    return world, pred, target


def test_charge_windup_is_not_instant_and_impact_costs():
    world, pred, cow = small_world()
    energy, stamina = pred.energy, pred.stamina
    world.step({pred.id: Action(Kind.CHARGE, 1, 0)})
    assert cow.health == 100 and pred.position == (5, 5)
    assert pred.charge_windup and pred.stamina < stamina - 20 and pred.energy < energy
    world.step({pred.id: Action()})
    assert cow.health == 100 - world.config.ravager_charge_damage
    assert pred.position == (7, 5)
    assert any(e.kind == "charge_hit" for e in world.last_events)
    assert pred.charge_recovery > 0
    mask = legal_action_mask(world, pred)
    assert not mask[ACTIONS[Species.RAVAGER].index(Action(Kind.CHARGE, 1, 0))]


def test_charge_can_be_dodged_and_miss_does_not_deal_damage():
    world, pred, cow = small_world()
    world.step({pred.id: Action(Kind.CHARGE, 1, 0)})
    world.step({cow.id: Action(Kind.MOVE, 0, 1)})
    assert cow.health == 100
    assert any(e.kind == "charge_resolved" and e.detail == "miss"
               for e in world.last_events)
    assert pred.position == (8, 5)


def test_charge_stops_at_wall_and_causes_fatigue_and_injury():
    world, pred, _ = small_world(cow=False)
    world.grid.get(7, 5).passable = False
    world.step({pred.id: Action(Kind.CHARGE, 1, 0)})
    world.step({})
    assert pred.position == (6, 5)
    assert pred.health == 97
    assert any(e.kind == "charge_resolved" and e.detail == "obstacle"
               for e in world.last_events)


def test_carcass_grab_move_drop_prevents_duplication():
    world, pred, _ = small_world(cow=False)
    corpse = Carcass(1, 6, 5, 8, Species.COW)
    world.carcasses[corpse.id] = corpse
    world.step({pred.id: Action(Kind.GRAB_CARCASS)})
    assert pred.dragged_carcass_id == 1
    assert corpse.carrier_id == pred.id
    assert corpse.x == pred.x
    world.step({pred.id: Action(Kind.MOVE, 0, 1)})
    assert corpse.x == pred.x and corpse.y == pred.y
    assert pred.drag_cooldown == 1
    origin = pred.position
    world.step({pred.id: Action(Kind.MOVE, 0, 1)})
    assert pred.position == origin
    world.step({pred.id: Action(Kind.DROP_CARCASS)})
    assert pred.dragged_carcass_id is None and corpse.carrier_id is None
    assert len(world.carcasses) == 1 and corpse.meat <= 8
    world.check_invariants()


def test_carcass_consumption_and_decayed_carcass_release_carrier():
    world, pred, _ = small_world(cow=False)
    corpse = Carcass(1, pred.x, pred.y, 2.2, Species.COW)
    world.carcasses[1] = corpse
    world.step({pred.id: Action(Kind.GRAB_CARCASS)})
    world.step({pred.id: Action(Kind.EAT_CARCASS)})
    assert pred.satiety > 0 and 0 < corpse.meat < 2.2
    world.step({pred.id: Action(Kind.EAT_CARCASS)})
    assert 1 not in world.carcasses and pred.dragged_carcass_id is None


def test_hibernation_requires_satiation_and_saves_calories():
    active, ap, _ = small_world(cow=False)
    sleeping, sp, _ = small_world(cow=False)
    for predator in (ap, sp):
        predator.satiety = 0.7
        predator.energy = 90
    sleeping.step({sp.id: Action(Kind.HIBERNATE)})
    active.step({ap.id: Action()})
    assert sp.hibernating
    for _ in range(15):
        sleeping.step({})
        active.step({})
    assert sp.energy > ap.energy
    mask = legal_action_mask(sleeping, sp)
    assert mask.sum() == 2
    assert mask[ACTIONS[Species.RAVAGER].index(Action(Kind.WAKE))] == 1
    sleeping.step({sp.id: Action(Kind.WAKE)})
    assert not sp.hibernating
    hungry, hp, _ = small_world(cow=False)
    hungry.step({hp.id: Action(Kind.HIBERNATE)})
    assert not hp.hibernating


def test_sneak_is_slow_tiring_and_not_invisible_at_touching_distance():
    world, pred, cow = small_world()
    pred.x, pred.y = 5, 5
    cow.x, cow.y = 9, 5
    world.grid.get(6, 5).cover = 0.9
    world.step({pred.id: Action(Kind.SNEAK, 1, 0)})
    assert pred.position == (6, 5) and pred.sneaking
    assert _nearest_actor(world, cow, Species.RAVAGER) is None
    assert legal_action_mask(world, pred)[
        ACTIONS[Species.RAVAGER].index(Action(Kind.SNEAK, 1, 0))] == 0
    world.step({pred.id: Action(Kind.WAIT)})
    world.step({pred.id: Action(Kind.MOVE, 1, 0)})
    assert not pred.sneaking
    assert _nearest_actor(world, cow, Species.RAVAGER) is pred
    pred.x, pred.y = 8, 5
    pred.sneaking = True
    assert _nearest_actor(world, cow, Species.RAVAGER) is pred


def test_observation_contract_legacy_unmodified_and_ravager_local():
    world, pred, _ = small_world()
    assert observation_size(Species.COW) == 60
    assert observation_size(Species.NOMAD) == 72
    assert observation_size(Species.RAVAGER) == 82
    assert observe(world, pred)["observation"].shape == (82,)
    assert observe(world, pred)["action_mask"].shape == (20,)
    assert (observe(world, pred)["observation"] <= 1).all()
    assert (observe(world, pred)["observation"] >= -1).all()


@pytest.mark.parametrize("scenario", ["ravager_ambush", "ravager_carcass",
                                       "ravager_hibernate", "ravager_mixed"])
def test_ravager_curricula_and_reward_are_finite(scenario):
    env = QoxCoreEnv(scenario=scenario, scripted_others=True, max_steps=12)
    obs, _ = env.reset(seed=303)
    pred = next(name for name in obs if name.startswith("ravager_"))
    for _ in range(12):
        mask = obs[pred]["action_mask"]
        index = int(mask.nonzero()[0][0])
        obs, rewards, term, trunc, info = env.step({pred: index})
        assert float(rewards[pred]) == pytest.approx(float(rewards[pred]))
        if term[pred] or trunc[pred]:
            break
    assert env.world.tick > 0


def test_ravager_state_is_serialized_and_replayable():
    # Traces store seed plus actions, not manual test fixture mutations.
    w = World(SimulationConfig(width=20, height=14, farmers=0, cows=1,
                               nomads=0, scavengers=0, ravagers=1))
    pred = next(a for a in w.actors.values() if a.species == Species.RAVAGER)
    cow = next(a for a in w.actors.values() if a.species == Species.COW)
    trace = Trace(w.config.as_dict())
    for order in ({pred.id: Action(Kind.SNEAK, 1, 0)},
                  {pred.id: Action(Kind.WAIT)},
                  {pred.id: Action(Kind.CHARGE, 1, 0)},
                  {cow.id: Action(Kind.MOVE, 0, 1)}):
        w.step(order)
        trace.append(w, order)
    assert w.state_hash() == replay(trace).state_hash()


def test_charge_config_validates():
    with pytest.raises(ValueError):
        SimulationConfig(ravager_charge_stamina=0)
    with pytest.raises(ValueError):
        SimulationConfig(ravager_charge_distance=0)
