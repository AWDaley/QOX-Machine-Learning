from dataclasses import asdict

import pytest

from qox.actions import Action, Kind
from qox.config import SimulationConfig
from qox.entities import Species
from qox.trace import Trace, replay, run
from qox.world import World


def small(**overrides):
    return SimulationConfig(**{**SimulationConfig(width=12, height=12, farmers=2, cows=3,
                                                nomads=1, scavengers=1, ravagers=1).as_dict(),
                               **overrides})


def test_seed_and_initial_snapshot_reproducible():
    a, b = World(small(seed=91)), World(small(seed=91))
    assert a.state_hash() == b.state_hash()
    assert a.snapshot() == b.snapshot()
    assert World(small(seed=92)).state_hash() != a.state_hash()


@pytest.mark.parametrize("mode", ["scripted", "random", "idle"])
def test_multi_tick_reproducibility_and_trace(tmp_path, mode):
    config = small(seed=4)
    world, trace = run(config, ticks=110, mode=mode, record=True)
    assert trace is not None
    assert len(trace.frames) == 110
    assert trace.frames[-1]["state_hash"] == world.state_hash()
    target = tmp_path / "trace.json"
    trace.save(target)
    assert replay(Trace.load(target)).state_hash() == world.state_hash()
    duplicate, _ = run(config, ticks=110, mode=mode)
    assert world.state_hash() == duplicate.state_hash()
    assert [asdict(x) for x in world.events] == [asdict(x) for x in duplicate.events]


def test_trace_detects_tampering():
    _, trace = run(small(), 5, record=True)
    trace.frames[2]["state_hash"] = "invalid"
    with pytest.raises(ValueError, match="Replay divergence at tick 3"):
        replay(trace)


def test_conflicting_movements_and_swaps():
    world = World(small(farmers=0, cows=2, nomads=0, scavengers=0, ravagers=0))
    a, b = list(world.actors.values())
    a.x, a.y, b.x, b.y = 4, 4, 6, 4
    world.step({a.id: Action(Kind.MOVE, 1, 0), b.id: Action(Kind.MOVE, -1, 0)})
    assert {a.position, b.position} == {(5, 4), (4, 4)} or {a.position, b.position} == {(5, 4), (6, 4)}
    assert len({a.position, b.position}) == 2
    a.x, a.y, b.x, b.y = 3, 3, 4, 3
    world.step({a.id: Action(Kind.MOVE, 1, 0), b.id: Action(Kind.MOVE, -1, 0)})
    assert a.position == (4, 3) and b.position == (3, 3)


def test_blocking_chain_is_resolved_without_overlap():
    world = World(small(farmers=0, cows=3, nomads=0, scavengers=0, ravagers=0))
    a, b, c = list(world.actors.values())
    a.x, a.y, b.x, b.y, c.x, c.y = 2, 2, 3, 2, 4, 2
    world.step({a.id: Action(Kind.MOVE, 1, 0), b.id: Action(Kind.MOVE, 1, 0)})
    assert (a.position, b.position, c.position) == ((2, 2), (3, 2), (4, 2))


def test_grazing_uses_finite_biomass_and_produces_dung():
    world = World(small(farmers=0, cows=1, nomads=0, scavengers=0, ravagers=0))
    cow = next(iter(world.actors.values()))
    tile = world.grid.get(cow.x, cow.y)
    tile.grass, tile.planted, tile.crop = 1.5, False, 0
    world.step({cow.id: Action(Kind.GRAZE)})
    assert tile.grass == 0
    assert cow.dung_pending == pytest.approx(0.45)
    assert cow.milk == pytest.approx(0.24)
    nutrition = sum(e.amount for e in world.events if e.kind == "graze")
    world.step({cow.id: Action(Kind.GRAZE)})
    assert sum(e.amount for e in world.events if e.kind == "graze") == nutrition
    for _ in range(world.config.ecology_interval - 2):
        world.step({cow.id: Action(Kind.WAIT)})
    assert tile.waste > 0
    assert cow.dung_pending == pytest.approx(0)


def test_crop_growth_harvesting_manure():
    world = World(small(farmers=1, cows=0, nomads=0, scavengers=0, ravagers=0))
    farmer = next(iter(world.actors.values()))
    tile = world.grid.get(farmer.x, farmer.y)
    tile.crop, tile.planted, tile.fertility, tile.waste = 0, False, 0.4, 2
    world.step({farmer.id: Action(Kind.COLLECT_WASTE)})
    assert farmer.carried_waste == pytest.approx(1.5)
    assert tile.waste == pytest.approx(0.5)
    world.step({farmer.id: Action(Kind.FERTILIZE)})
    assert tile.fertility == pytest.approx(0.58)
    assert farmer.carried_waste == pytest.approx(0.5)
    before = farmer.carried_food
    world.step({farmer.id: Action(Kind.PLANT)})
    assert tile.planted and tile.crop > 0
    assert farmer.carried_food == pytest.approx(before - 0.3)
    for _ in range(20):
        world.step({farmer.id: Action(Kind.WAIT)})
    crop_before = tile.crop
    world.step({farmer.id: Action(Kind.HARVEST)})
    assert tile.crop < crop_before
    assert any(e.kind == "harvest" for e in world.events)


def test_carcass_food_is_finite():
    world = World(small(farmers=0, cows=1, nomads=0, scavengers=1, ravagers=0))
    cow = next(a for a in world.actors.values() if a.species == Species.COW)
    scav = next(a for a in world.actors.values() if a.species == Species.SCAVENGER)
    scav.x, scav.y = cow.x + 1, cow.y
    cow.health, cow.energy = 0.3, 0
    world.step({})
    assert cow.id not in world.actors and len(world.carcasses) == 1
    carcass = next(iter(world.carcasses.values()))
    scav.x, scav.y = carcass.x, carcass.y
    previous = carcass.meat
    world.step({scav.id: Action(Kind.EAT_CARCASS)})
    assert carcass.meat == previous - 2
    for _ in range(6):
        world.step({scav.id: Action(Kind.EAT_CARCASS)})
    assert not world.carcasses
    assert sum(e.amount for e in world.events if e.kind == "eat_carcass") <= 12


def test_invalid_config_and_action():
    with pytest.raises(ValueError):
        SimulationConfig(width=4)
    with pytest.raises(ValueError):
        SimulationConfig(farmers=-1)
    with pytest.raises(ValueError):
        Action(Kind.MOVE, 3, 0)
    with pytest.raises(ValueError):
        World(small()).step({999: Action(Kind.REST)})


def test_nonnegative_resources_in_long_run():
    world, _ = run(small(seed=20), ticks=250, mode="random")
    world.check_invariants()
    summary = world.summary()
    assert summary["tick"] == 250
    assert summary["grass"] >= 0 and summary["stored_food"] >= 0
