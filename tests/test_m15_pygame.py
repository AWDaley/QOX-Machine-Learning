"""Windows CI exercises keyboard world-saving through a real Pygame renderer."""

import pytest

pygame = pytest.importorskip("pygame")

from qox.config import SimulationConfig  # noqa: E402
from qox.viewer import display  # noqa: E402
from qox.world import World  # noqa: E402
from qox.world_save import load_world  # noqa: E402


def test_viewer_f5_writes_resumable_world(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    frames = iter([
        [pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_F5, "mod": 0})],
        [pygame.event.Event(pygame.QUIT)],
    ])
    monkeypatch.setattr(pygame.event, "get", lambda: next(frames, []))
    display(SimulationConfig(seed=118), mode="scripted")
    files = list((tmp_path / "Documents" / "QOX" / "Worlds").glob("*.qoxworld"))
    assert len(files) == 1
    world, info = load_world(files[0])
    assert info.mode == "scripted"
    assert world.state_hash() == World(SimulationConfig(seed=118)).state_hash()
