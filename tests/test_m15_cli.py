"""Headless M15 user entry points for save and resume."""

from qox.cli import main
from qox.config import SimulationConfig
from qox.agents import actions_for
from qox.world import World
from qox.world_save import load_world


def test_cli_save_and_resume_matches_direct_world(tmp_path, capsys):
    first = tmp_path / "first.qoxworld"
    second = tmp_path / "second.qoxworld"
    assert main(["run", "--ticks", "20", "--mode", "scripted",
                 "--save-world", str(first)]) == 0
    assert first.exists()
    assert main(["resume-world", "--input", str(first), "--ticks", "30",
                 "--output", str(second)]) == 0
    restored, _ = load_world(second)
    direct = World(SimulationConfig())
    for _ in range(50):
        direct.step(actions_for(direct, "scripted"))
    assert restored.state_hash() == direct.state_hash()
    assert restored.tick == 50
    capsys.readouterr()
