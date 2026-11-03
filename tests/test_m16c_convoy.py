"""M16c: generalizable convoy protection and non-promoting comparisons."""

from __future__ import annotations

import json

import pytest

from qox.entities import Species
from qox.envs.core import QoxCoreEnv, seed_variant
from qox.perception import features_for


def test_convoy_variations_are_physical_and_locally_observable():
    directions, approaches, heights, predators = set(), set(), set(), set()
    for seed in range(72):
        env = QoxCoreEnv(scenario="nomad_convoy", max_steps=30,
                         scripted_others=True, nomad_v2=True, cow_calls=True)
        obs, _ = env.reset(seed=seed)
        world = env.world
        assert len([k for k in obs if k.startswith("cow_")]) == 3
        cows = [a for a in world.actors.values() if a.species == Species.COW]
        nomad = next(a for a in world.actors.values() if a.species == Species.NOMAD)
        ravager = next(a for a in world.actors.values() if a.species == Species.RAVAGER)
        assert len({a.position for a in world.actors.values()}) == 5
        assert world.herd_goal is not None
        assert nomad.health == 100
        assert ravager.health in (35., 50., 65.)
        assert len(features_for(Species.NOMAD, nomad_v2=True)) == 75
        assert "herd_goal_dx" not in features_for(Species.COW, cow_calls=True)
        directions.add(world.herd_goal[0])
        approaches.add(ravager.x > cows[0].x)
        heights.add(cows[1].y)
        predators.add(seed_variant(seed))
        for _ in range(5):
            env.step({})
            world.check_invariants()
    assert directions == {2, 15}
    assert approaches == {True, False}
    assert len(heights) == 3
    assert predators == {0, 1, 2}


def test_convoy_seed_and_scripted_actions_reproduce_identical_world_hashes():
    for seed in (17, 29, 46):
        first = QoxCoreEnv(scenario="nomad_convoy", scripted_others=True,
                           nomad_v2=True, cow_calls=True, max_steps=32)
        second = QoxCoreEnv(scenario="nomad_convoy", scripted_others=True,
                            nomad_v2=True, cow_calls=True, max_steps=32)
        first.reset(seed=seed)
        second.reset(seed=seed)
        for _ in range(20):
            assert first.world.state_hash() == second.world.state_hash()
            first.step({})
            second.step({})
        assert first.world.state_hash() == second.world.state_hash()


def test_mixed_convoy_alternates_a_real_predator_and_safe_migration():
    seen = {"safe": 0, "threat": 0}
    for seed in range(32):
        env = QoxCoreEnv(scenario="nomad_convoy_mixed", max_steps=16,
                         scripted_others=True, nomad_v2=True, cow_calls=True)
        env.reset(seed=seed)
        n = sum(actor.species == Species.RAVAGER for actor in env.world.actors.values())
        assert n in (0, 1)
        assert env.world.herd_goal is not None
        seen["threat" if n else "safe"] += 1
        for _ in range(2):
            env.step({})
            env.world.check_invariants()
    assert seen == {"safe": 16, "threat": 16}


def test_comparison_same_checkpoints_has_zero_paired_difference(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.joint_ppo import JointTrainOptions, _save_model
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import NomadPolicy
    from qox.learning.protector_eval import compare_protectors, SCENARIOS

    options = JointTrainOptions(worlds=2, rollout=96, cow_steps=96, nomad_steps=96)
    cow = tmp_path / "cow.pt"
    nomad = tmp_path / "nomad.pt"
    c = CowPolicy(hidden=16, cow_calls=True)
    n = NomadPolicy(hidden=16, nomad_v2=True)
    _save_model(c, torch.optim.Adam(c.parameters()), Species.COW, cow, options, 0, 0)
    _save_model(n, torch.optim.Adam(n.parameters()), Species.NOMAD, nomad, options, 0, 0)
    original = cow.read_bytes(), nomad.read_bytes()
    r1 = compare_protectors(cow, nomad, cow, nomad, episodes=2, seed=1170000,
                            max_steps=12)
    r2 = compare_protectors(cow, nomad, cow, nomad, episodes=2, seed=1170000,
                            max_steps=12)
    assert r1 == r2
    assert not r1["eligible_for_manual_review"]
    assert set(r1["cases"]) == set(SCENARIOS)
    for entry in r1["cases"].values():
        assert all(metric["mean"] == 0 for metric in entry["difference"].values())
    assert (cow.read_bytes(), nomad.read_bytes()) == original


def test_m16c_cli_exports_comparison_without_promoting(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.joint_ppo import JointTrainOptions, _save_model
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import NomadPolicy
    opts = JointTrainOptions(worlds=2, rollout=96, cow_steps=96, nomad_steps=96)
    cfile, nfile, report = tmp_path / "c.pt", tmp_path / "n.pt", tmp_path / "result.json"
    c, n = CowPolicy(16, cow_calls=True), NomadPolicy(16, nomad_v2=True)
    _save_model(c, torch.optim.Adam(c.parameters()), Species.COW, cfile, opts, 0, 0)
    _save_model(n, torch.optim.Adam(n.parameters()), Species.NOMAD, nfile, opts, 0, 0)
    original = cfile.read_bytes(), nfile.read_bytes()
    assert main(["compare-protectors", "--baseline-cow", str(cfile),
                 "--baseline-nomad", str(nfile), "--candidate-cow", str(cfile),
                 "--candidate-nomad", str(nfile), "--episodes", "2",
                 "--max-steps", "12", "--report", str(report)]) == 0
    payload = json.loads(report.read_text())
    assert "nomad_convoy" in payload["cases"]
    assert payload["promotion_checks"]["enough_heldout_worlds"] is False
    assert (cfile.read_bytes(), nfile.read_bytes()) == original


def test_joint_and_nomad_training_accept_convoy_curricula(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.joint_ppo import JointTrainOptions
    from qox.learning.nomad_ppo import NomadTrainOptions
    assert JointTrainOptions(scenario="nomad_convoy_mixed",
                             worlds=2, rollout=96).scenario == "nomad_convoy_mixed"
    assert NomadTrainOptions(scenario="nomad_convoy",
                             nomad_v2=True).scenario == "nomad_convoy"
