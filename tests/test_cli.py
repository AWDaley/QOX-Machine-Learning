import json

from qox.cli import main
from qox.config import SimulationConfig


def test_config_toml(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[world]\nwidth=12\nheight=13\nseed=123\n", encoding="utf-8")
    loaded = SimulationConfig.from_toml(config)
    assert (loaded.width, loaded.height, loaded.seed) == (12, 13, 123)
    config.write_text("[world]\nunknown=1\n", encoding="utf-8")
    try:
        SimulationConfig.from_toml(config)
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("Unknown setting accepted")


def test_cli_run_report_and_replay(tmp_path, capsys):
    report = tmp_path / "out" / "report.json"
    trace = tmp_path / "out" / "trace.json"
    assert main(["run", "--ticks", "30", "--seed", "5", "--mode", "scripted",
                 "--report", str(report), "--trace", str(trace)]) == 0
    assert json.loads(report.read_text())["tick"] == 30
    assert json.loads(trace.read_text())["frames"][-1]["tick"] == 30
    assert main(["replay", "--trace", str(trace)]) == 0
    assert "Replay OK" in capsys.readouterr().out
