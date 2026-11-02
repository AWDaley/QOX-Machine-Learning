"""Pygame smoke coverage: event effects and clickable recent-event rows."""

import pytest

pygame = pytest.importorskip("pygame")

from qox.entities import Event  # noqa: E402
from qox.viewer import _event_effects, _event_panel, _focus_event  # noqa: E402
from qox.viewer_events import EventObserver  # noqa: E402
from qox.viewer_layout import Camera, layout_for  # noqa: E402


def test_draw_event_overlays_on_an_offscreen_surface():
    pygame.font.init()
    try:
        surface = pygame.Surface((1180, 760))
        viewport = layout_for(*surface.get_size()).map
        camera = Camera(40, 40, viewport)
        font = pygame.font.Font(None, 20)

        class FakeWorld:
            tick = 1
            last_events = [
                Event(1, "festival_success", None, x=5, y=5),
                Event(1, "scary_call", 12, detail="radius:6", x=8, y=8),
                Event(1, "charge_resolved", 15, detail="miss", x=11, y=11),
            ]
            actors = {}

        observer = EventObserver()
        observer.accept(FakeWorld(), {}, {})
        surface.fill((9, 15, 25))
        _event_effects(surface, camera, viewport, observer, font)
        rows = _event_panel(surface, camera, viewport, observer, font)
        assert len(rows) == 3
        for rectangle, event in rows:
            assert rectangle.width > 0 and rectangle.height > 0
            assert rectangle.collidepoint(rectangle.center)
            _focus_event(camera, viewport, event)
        assert surface.get_at((viewport.x + 13, viewport.bottom - 25)) != (9, 15, 25)
    finally:
        pygame.font.quit()
