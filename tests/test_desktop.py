"""Desktop checks don't require a window or an actual Windows machine."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json

import pytest

pytest.importorskip("torch")

from qox.desktop import _parse_positive_integer, smoke_test
from qox.desktop_models import (ModelSession, MultiPolicyController, ROLES,
                                checkpoint_information, train_role)
from qox.config import SimulationConfig
from qox.entities import Species
from qox.perception import ACTIONS
from qox.world import World


@pytest.fixture
def fresh_models(tmp_path: Path):
    return ModelSession.fresh(tmp_path / "original", seed=123)


def test_fresh_bundle_is_real_and_roundtrips(fresh_models, tmp_path):
    session = fresh_models
    original = {role: checkpoint_information(role, session.path(role)) for role in ROLES}
    assert all(v["steps"] == 0 for v in original.values())
    assert original[Species.COW]["observation_version"] == 2
    assert original[Species.NOMAD]["observation_version"] == 2
    output = session.save(tmp_path / "my-lab")
    assert output.suffix == ".qox" and session.dirty is False
    another = ModelSession.load(output, tmp_path / "another")
    assert not another.dirty
    for species in ROLES:
        assert another.path(species).read_bytes() == session.path(species).read_bytes()
        assert another.status[species].steps == 0
    assert original == {role: checkpoint_information(role, another.path(role))
                        for role in ROLES}


def test_new_seed_changes_weights(tmp_path):
    first = ModelSession.fresh(tmp_path / "a", seed=17)
    same = ModelSession.fresh(tmp_path / "b", seed=17)
    different = ModelSession.fresh(tmp_path / "c", seed=19)
    import torch
    for species in ROLES:
        a = torch.load(first.path(species), map_location="cpu", weights_only=True)
        b = torch.load(same.path(species), map_location="cpu", weights_only=True)
        c = torch.load(different.path(species), map_location="cpu", weights_only=True)
        for weight in a["policy_state"]:
            assert torch.equal(a["policy_state"][weight], b["policy_state"][weight])
        assert any(not torch.equal(a["policy_state"][weight],
                                       c["policy_state"][weight])
                   for weight in a["policy_state"])


def test_multi_species_actions_are_legal(fresh_models):
    controller = MultiPolicyController(fresh_models, sample_actions=False,
                                       festival_assist=True)
    world = World(SimulationConfig(seed=21, festival_enabled=True,
                                   reproduction_enabled=True,
                                   population_dynamics_enabled=True))
    for _ in range(5):
        actions = controller.actions(world)
        assert set(actions) == set(world.actors)
        for actor_id, action in actions.items():
            species = world.actors[actor_id].species
            assert action in ACTIONS[species] or action.kind.value.startswith(
                ("contribute_festival", "start_"))
        world.step(actions)
    assert world.tick == 5


def test_bundle_rejects_tampering_without_replacing_current(fresh_models, tmp_path):
    output = fresh_models.save(tmp_path / "clean.qox")
    before = {s: fresh_models.path(s).read_bytes() for s in ROLES}
    modified = tmp_path / "tampered.qox"
    with ZipFile(output) as source, ZipFile(modified, "w", ZIP_DEFLATED) as destination:
        for name in source.namelist():
            if name == "policies/cow.pt":
                destination.writestr(name, source.read("policies/farmer.pt"))
            else:
                destination.writestr(name, source.read(name))
    with pytest.raises(ValueError, match="(schema|mismatch)"):
        ModelSession.load(modified, tmp_path / "invalid")
    assert all(fresh_models.path(s).read_bytes() == before[s] for s in ROLES)


def test_bundle_rejects_extraneous_files_and_versions(fresh_models, tmp_path):
    output = fresh_models.save(tmp_path / "clean.qox")
    modified = tmp_path / "bad.qox"
    with ZipFile(output) as source, ZipFile(modified, "w", ZIP_DEFLATED) as target:
        for filename in source.namelist():
            target.writestr(filename, source.read(filename))
        target.writestr("../evil.txt", "not accepted")
    with pytest.raises(ValueError, match="unexpected"):
        ModelSession.load(modified, tmp_path / "invalid")
    changed_version = tmp_path / "version.qox"
    with ZipFile(output) as source, ZipFile(changed_version, "w", ZIP_DEFLATED) as target:
        for filename in source.namelist():
            content = source.read(filename)
            if filename == "manifest.json":
                data = json.loads(content)
                data["version"] = 999
                content = json.dumps(data).encode()
            target.writestr(filename, content)
    with pytest.raises(ValueError, match="supported"):
        ModelSession.load(changed_version, tmp_path / "invalid")


@pytest.mark.parametrize("species", list(ROLES))
def test_training_continues_from_current_weights(species, fresh_models, tmp_path):
    output = tmp_path / f"trained-{species.value}.pt"
    result = train_role(species, fresh_models.path(species), output, steps=128, seed=7)
    assert result is not None
    first = checkpoint_information(species, output)
    assert first["steps"] >= 128
    again = tmp_path / f"trained-again-{species.value}.pt"
    train_role(species, output, again, steps=128, seed=8)
    second = checkpoint_information(species, again)
    assert second["steps"] >= first["steps"] + 128
    fresh_models.replace(species, again, origin="trained")
    assert fresh_models.status[species].steps == second["steps"]


def test_validation_and_smoke():
    with pytest.raises(ValueError):
        _parse_positive_integer("hello", name="Seed")
    with pytest.raises(ValueError):
        _parse_positive_integer("-2", name="Seed", minimum=0)
    assert _parse_positive_integer("0", name="Seed", minimum=0) == 0
    assert smoke_test() == 0
