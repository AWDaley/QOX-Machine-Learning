"""M13: authoritative telemetry, neural inspection, and deterministic seek tests."""
from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError

import pytest

from qox.actions import Action, Kind
from qox.agents import scripted_actions
from qox.config import SimulationConfig
from qox.decision_inspector import action_label, capture_decision, with_outcome
from qox.entities import Species
from qox.perception import features_for, observe
from qox.replay_session import ReplayCursor
from qox.telemetry import Telemetry
from qox.trace import Trace, replay
from qox.world import World


def _world(*, seed=555, births=False) -> World:
    return World(SimulationConfig(seed=seed, population_dynamics_enabled=births,
                                  reproduction_enabled=births, festival_enabled=True))


def test_telemetry_does_not_mutate_world_and_bounds_its_history():
    world = _world()
    stats = Telemetry(sample_every=2, limit=4)
    original_hash = world.state_hash()
    first = stats.observe(world)
    assert first.tick == 0 and first.populations["cow"] == world.config.cows
    assert world.state_hash() == original_hash
    assert stats.observe(world) is None  # duplicated presentation frames
    for _ in range(16):
        orders = scripted_actions(world)
        world.step(orders)
        before = world.state_hash()
        stats.observe(world)
        assert world.state_hash() == before
    assert [s.tick for s in stats.samples] == [10, 12, 14, 16]
    assert stats.event_totals == {k: sum(e.kind == k for e in world.events)
                                  for k in stats.event_totals}
    assert len(stats.series("pop_cow")) == 4
    assert all(value >= 0 for tick, value in stats.series("pop_cow"))
    assert stats.samples[-1].fertility >= 0
    with pytest.raises(ValueError):
        stats.observe(_world())


def test_telemetry_exports_json_csv_with_metadata(tmp_path):
    world = _world(seed=899)
    telemetry = Telemetry(sample_every=1)
    telemetry.observe(world)
    for _ in range(4):
        world.step(scripted_actions(world))
        telemetry.observe(world)
    json_path = telemetry.export_json(tmp_path / "stats" / "data.json", policy="scripted")
    csv_path = telemetry.export_csv(tmp_path / "stats" / "data.csv")
    data = json.loads(json_path.read_text())
    assert data["schema_version"] == 1 and data["seed"] == 899
    assert data["policy"] == "scripted" and len(data["samples"]) == 5
    with csv_path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 5
    assert int(records[-1]["pop_cow"]) == telemetry.samples[-1].populations["cow"]
    assert "fertility" in records[-1] and "births" in records[-1]


def test_telemetry_rejects_duplicate_or_skipped_ticks():
    world = _world()
    t = Telemetry(sample_every=1)
    t.observe(world)
    for _ in range(2):
        world.step(scripted_actions(world))
    with pytest.raises(ValueError, match="each world tick"):
        t.observe(world)
    for kwargs in ({"sample_every": 0}, {"limit": 1}):
        with pytest.raises(ValueError):
            Telemetry(**kwargs)


def test_decision_inspection_uses_real_local_features_mask_and_action_names():
    world = _world()
    actor = next(a for a in world.actors.values() if a.species == Species.NOMAD)
    original_hash = world.state_hash()
    decision = capture_decision(world, actor.id)
    observed = observe(world, actor)
    assert len(decision.observations) == len(features_for(Species.NOMAD))
    assert tuple(name for name, val in decision.observations) == features_for(Species.NOMAD)
    assert [float(value) for name, value in decision.observations] == pytest.approx(
        observed["observation"].tolist())
    assert [int(choice.legal) for choice in decision.choices] == list(
        observed["action_mask"].astype(int))
    assert action_label(Action(Kind.MOVE, 0, -1)) == "move N"
    assert decision.policy == "scripted or random"
    assert world.state_hash() == original_hash
    assert capture_decision(world, -1) is None
    with pytest.raises(FrozenInstanceError):
        decision.action = "do something else"


class _KnownPolicy:
    class Model:
        cow_calls = True
        nomad_v2 = False
    def __init__(self):
        self.models = {Species.COW: self.Model()}
    def probabilities(self, world, actor_id):
        from qox.perception import ACTIONS
        return [1.0] + [0.0] * (len(ACTIONS[Species.COW]) - 1)


def test_decision_inspector_uses_controller_schema_and_preserves_pre_step_values():
    world = _world()
    actor = next(a for a in world.actors.values() if a.species == Species.COW)
    policy = _KnownPolicy()
    pre = capture_decision(world, actor.id, policy)
    assert len(pre.observations) == 66
    assert pre.policy.startswith("neural")
    assert pre.top_actions()[0].probability == 1.0
    assert all(c.legal for c in pre.top_actions())
    orders = scripted_actions(world)
    world.step(orders)
    post = with_outcome(pre, orders[actor.id], "confirmed")
    assert post.tick == 0 and post.outcome == "confirmed"
    assert post.action == action_label(orders[actor.id])
    assert post.observations == pre.observations
    assert post.actor_id == actor.id


def test_replay_cursor_seek_and_hashes_with_population_events():
    world = _world(seed=270, births=True)
    trace = Trace(world.config.as_dict())
    all_hashes = [world.state_hash()]
    for _ in range(85):
        actions = scripted_actions(world)
        world.step(actions)
        trace.append(world, actions)
        all_hashes.append(world.state_hash())
    final_hash = replay(trace).state_hash()
    assert final_hash == all_hashes[-1]
    cursor = ReplayCursor(trace, checkpoint_every=7, max_checkpoints=5)
    for tick in (1, 41, 12, 50, 85, 3, 85, 0, 85, 20):
        assert cursor.seek(tick).state_hash() == all_hashes[tick]
        assert cursor.position == tick
        assert len(cursor._cache) <= 5
        assert 0 in cursor._cache
        if tick:
            assert cursor.last_actions
    before = cursor.world.state_hash()
    events = cursor.major_events()
    assert all(0 < e.tick <= 85 for e in events)
    assert cursor.world.state_hash() == before
    assert cursor.next_event_tick(forward=True) is None or cursor.next_event_tick(forward=True) > 20
    with pytest.raises(IndexError):
        cursor.seek(86)
    with pytest.raises(IndexError):
        cursor.seek(-1)


def test_replay_cursor_rejects_tampered_trace():
    world = _world()
    trace = Trace(world.config.as_dict())
    for _ in range(8):
        orders = scripted_actions(world)
        world.step(orders)
        trace.append(world, orders)
    trace.frames[2]["state_hash"] = "wrong"
    cursor = ReplayCursor(trace)
    with pytest.raises(ValueError, match="Replay divergence"):
        cursor.seek(6)


def test_trace_cli_parsing_and_legacy_replay_remain_available():
    from qox.cli import parser
    args = parser().parse_args(["replay-viewer", "--trace", "my_trace.json"])
    assert args.command == "replay-viewer"
    args = parser().parse_args(["viewer", "--record-trace", "trace.json"])
    assert args.record_trace.name == "trace.json"
    args = parser().parse_args(["replay", "--trace", "trace.json"])
    assert args.command == "replay"


def test_event_history_filter_and_bounded_json_export(tmp_path):
    from qox.entities import Event
    from qox.viewer_events import EventObserver

    class FakeWorld:
        tick = 2
        actors = {}
        last_events = [Event(2, "graze", 1, x=3, y=3),
                       Event(2, "charge_hit", 2, detail="target:1", x=4, y=3),
                       Event(2, "festival_success", None, x=5, y=5)]

    from qox.viewer_events import ActorStamp
    observer = EventObserver(history_limit=3)
    before = {1: ActorStamp(3, 3, Species.COW),
              2: ActorStamp(4, 3, Species.RAVAGER)}
    observer.accept(FakeWorld(), {}, before)
    assert len(observer.recent(5, important_only=False, species=Species.COW)) == 1
    assert len(observer.recent(5, important_only=False, actor_id=2)) == 1
    assert len(observer.recent(5, important_only=True, species=Species.COW)) == 0
    path = tmp_path / "events.json"
    observer.export_json(path)
    obj = json.loads(path.read_text())
    assert obj["schema_version"] == 1
    assert len(obj["events"]) == 3
    assert obj["events"][-1]["kind"] == "festival_success"
