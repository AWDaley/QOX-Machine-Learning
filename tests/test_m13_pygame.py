"""Optional offscreen Pygame smoke coverage for new M13 panels and timeline."""

import pytest

pygame = pytest.importorskip("pygame")

from qox.config import SimulationConfig  # noqa: E402
from qox.decision_inspector import capture_decision  # noqa: E402
from qox.entities import Species  # noqa: E402
from qox.telemetry import Telemetry  # noqa: E402
from qox.viewer_analytics import draw_decision, draw_graphs  # noqa: E402
from qox.viewer_layout import layout_for  # noqa: E402
from qox.world import World  # noqa: E402


@pytest.mark.parametrize("window", [(1440, 930), (820, 580), (740, 500)])
def test_offscreen_graph_and_neural_panels(window):
    pygame.font.init()
    try:
        world = World(SimulationConfig(seed=203))
        t = Telemetry(sample_every=1)
        t.observe(world)
        for _ in range(3):
            world.step({})
            t.observe(world)
        surface = pygame.Surface(window)
        side = layout_for(*window).sidebar
        title = pygame.font.Font(None, 32)
        normal = pygame.font.Font(None, 23)
        small = pygame.font.Font(None, 19)
        surface.fill((0, 0, 0))
        draw_graphs(surface, side, t, title, normal, small)
        assert surface.get_at((side.x + 6, side.y + 6))[:3] != (0, 0, 0)
        actor = next(a for a in world.actors.values() if a.species == Species.COW)
        decision = capture_decision(world, actor.id)
        surface.fill((0, 0, 0))
        max_scroll = draw_decision(surface, side, decision, title, normal, small,
                                   scroll=9999)
        assert max_scroll >= 0
        assert surface.get_at((side.x + 6, side.y + 6))[:3] != (0, 0, 0)
    finally:
        pygame.font.quit()
