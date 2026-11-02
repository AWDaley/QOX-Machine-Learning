"""M15 exact world continuation, integrity, and policy identity tests."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from qox.actions import Kind
from qox.agents import actions_for
from qox.config import SimulationConfig
from qox.entities import CallSignal, Carcass, Species
from qox.festivals import Festival, FestivalKind
from qox.world import World
from qox.world_save import (WorldSaveInfo, load_world, policy_fingerprints,
                            save_world)


def test_save_and_restore_all_state_with_active_events(tmp_path):
    w = World(SimulationConfig(seed=235, reproduction_enabled=True,
                               population_dynamics_enabled=True, festival_enabled=True,
                               reproduction_interval=5))
    for _ in range(17):
        w.step(actions_for(w, "scripted"))
    # Realistic partial transient states, including an uncompleted charge, a
    # carcass attached to its carrier, an active festival and sound waves.
    ravager = next(a for a in w.actors.values() if a.species == Species.RAVAGER)
    ravager.charge_windup = 1
    ravager.charge_dx = 1
    ravager.charge_dy = 0
    ravager.charge_cooldown = 3
    carcass = Carcass(w.next_carcass_id, ravager.x, ravager.y, 5.0, Species.COW,
                      carrier_id=ravager.id)
    w.carcasses[carcass.id] = carcass
    w.next_carcass_id += 1
    ravager.dragged_carcass_id = carcass.id
    w.calls.append(CallSignal(1, Kind.FRIENDLY_CALL.value, 4, 5, w.tick, 6))
    settlement = w.settlements[0]
    festival = Festival(w.next_festival_id, FestivalKind.FARMER, "settlement",
                        settlement.id, settlement.x, settlement.y,
                        w.tick, w.tick + 8, 1, 8.0, 0.0)
    w.active_festivals[festival.id] = festival
    w.next_festival_id += 1
    w.check_invariants()
    saved_hash = w.state_hash()
    rng_before = w.rng.getstate()
    policy_rng_before = w.policy_rng.getstate()
    path = save_world(w, tmp_path / "full-test", mode="scripted")
    assert path.suffix == ".qoxworld" and path.is_file()
    assert w.state_hash() == saved_hash
    assert w.rng.getstate() == rng_before
    assert w.policy_rng.getstate() == policy_rng_before

    restored, info = load_world(path)
    assert isinstance(info, WorldSaveInfo)
    assert restored.state_hash() == w.state_hash()
    assert restored.rng.getstate() == w.rng.getstate()
    assert restored.policy_rng.getstate() == w.policy_rng.getstate()
    assert [asdict(e) for e in restored.events] == [asdict(e) for e in w.events]
    assert restored.last_events == w.last_events
    assert restored.next_carcass_id == w.next_carcass_id
    assert restored.next_id == w.next_id
    assert restored.birth_totals == w.birth_totals
    assert restored.active_festivals == w.active_festivals
    assert restored.camps == w.camps
    for _ in range(40):
        # Real, independently generated decisions also stress policy sampling.
        original_actions = actions_for(w, "scripted")
        restored_actions = actions_for(restored, "scripted")
        assert original_actions == restored_actions
        w.step(original_actions)
        restored.step(restored_actions)
        assert restored.state_hash() == w.state_hash()
        assert restored.events == w.events


@pytest.mark.parametrize("mode", ["scripted", "random", "idle"])
@pytest.mark.parametrize("reproduction,festivals", [(False, False), (True, True)])
def test_interrupted_equals_uninterrupted(mode, reproduction, festivals, tmp_path):
    config = SimulationConfig(seed=743, reproduction_enabled=reproduction,
                              population_dynamics_enabled=reproduction,
                              festival_enabled=festivals, reproduction_interval=10)
    direct = World(config)
    for _ in range(30):
        direct.step(actions_for(direct, mode))
    save = save_world(direct, tmp_path / "world.qoxworld", mode=mode)
    resumed, _ = load_world(save)
    for _ in range(40):
        action1 = actions_for(direct, mode)
        action2 = actions_for(resumed, mode)
        assert action1 == action2
        direct.step(action1)
        resumed.step(action2)
        assert direct.state_hash() == resumed.state_hash()
        assert direct.rng.getstate() == resumed.rng.getstate()
        assert direct.policy_rng.getstate() == resumed.policy_rng.getstate()


def _policy_files(folder: Path):
    folder.mkdir(exist_ok=True)
    policies = {}
    for role in Species:
        name = folder / f"{role.value}.pt"
        name.write_bytes(b"pretend-weight:" + role.value.encode())
        policies[role.value] = name
    return policies


def test_policy_binding_and_options(tmp_path):
    files = _policy_files(tmp_path / "models")
    world = World(SimulationConfig(seed=47, festival_enabled=True))
    controls = {"sample_actions": False, "festival_assist": True}
    target = save_world(world, tmp_path / "trained.qoxworld", mode="multi-learned",
                        policy_files=files, options=controls)
    restored, meta = load_world(target, policy_files=files)
    assert restored.state_hash() == world.state_hash()
    assert meta.policy_hashes == policy_fingerprints(files)
    assert meta.options == controls
    assert meta.mode == "multi-learned"
    with pytest.raises(ValueError, match="original five neural policies"):
        load_world(target)
    files["cow"].write_bytes(b"a different checkpoint")
    with pytest.raises(ValueError, match="matching .qox model set"):
        load_world(target, policy_files=files)


def test_save_refuses_incomplete_policy_list_and_invalid_options(tmp_path):
    world = World()
    with pytest.raises(ValueError, match="all five"):
        save_world(world, tmp_path / "bad.qoxworld", mode="multi-learned",
                   policy_files={"cow": tmp_path / "cow.pt"})
    with pytest.raises(ValueError, match="controller options"):
        save_world(world, tmp_path / "bad.qoxworld", options={"sample_actions": True})
    with pytest.raises(ValueError, match="Unsupported mode"):
        save_world(world, tmp_path / "bad.qoxworld", mode="made-up")


def _rewrite(path: Path, dest: Path, fn):
    with ZipFile(path) as src, ZipFile(dest, "w", ZIP_DEFLATED) as dst:
        for name in src.namelist():
            dst.writestr(name, fn(name, src.read(name)))
    return dest


def test_rejects_corrupt_payload_and_preserves_existing_save(tmp_path):
    world = World()
    saved = save_world(world, tmp_path / "good.qoxworld")
    original = saved.read_bytes()
    bad = _rewrite(saved, tmp_path / "bad.qoxworld",
                   lambda name, blob: blob + b"garbage" if name == "world.json" else blob)
    with pytest.raises(ValueError, match="integrity"):
        load_world(bad)
    with pytest.raises(ValueError):
        save_world(world, saved, mode="multi-learned")
    assert saved.read_bytes() == original


def test_refuses_unexpected_zip_members_duplicate_and_bad_version(tmp_path):
    saved = save_world(World(), tmp_path / "good.qoxworld")
    with ZipFile(saved) as src, ZipFile(tmp_path / "extra.qoxworld", "w") as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("../../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="archive"):
        load_world(tmp_path / "extra.qoxworld")
    changed = _rewrite(saved, tmp_path / "version.qoxworld",
                       lambda name, data: json.dumps({**json.loads(data), "version": 99}).encode()
                       if name == "manifest.json" else data)
    with pytest.raises(ValueError, match="Unsupported"):
        load_world(changed)


def test_detects_forged_world_state_even_when_checksum_matches(tmp_path):
    saved = save_world(World(), tmp_path / "good.qoxworld")
    with ZipFile(saved) as src:
        world = json.loads(src.read("world.json"))
        manifest = json.loads(src.read("manifest.json"))
    world["next_id"] = 1
    data = json.dumps(world).encode()
    manifest["world_sha256"] = hashlib.sha256(data).hexdigest()
    with ZipFile(tmp_path / "forged.qoxworld", "w", ZIP_DEFLATED) as dst:
        dst.writestr("world.json", data)
        dst.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="entity identity"):
        load_world(tmp_path / "forged.qoxworld")


def test_save_when_population_disabled_preserves_off_snapshot_data(tmp_path):
    world = World(SimulationConfig(seed=31))
    a = next(iter(world.actors.values()))
    a.parent_id = None
    a.last_birth_tick = -15
    world.policy_rng.random()
    world.next_id += 100  # excluded from legacy M8 state hash
    saved = save_world(world, tmp_path / "old-repro-mode.qoxworld")
    loaded, _ = load_world(saved)
    assert loaded.next_id == world.next_id
    assert loaded.policy_rng.getstate() == world.policy_rng.getstate()
    assert loaded.actors[a.id].last_birth_tick == a.last_birth_tick
