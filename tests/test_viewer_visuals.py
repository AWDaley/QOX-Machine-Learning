"""No-display checks for sprite variants and the resizable camera."""
from qox.actions import Action, Kind
from qox.entities import Actor, Species
from qox.sprites import BEHAVIORS, actor_behavior, sprite_pixels, SIZE
from qox.viewer_layout import Camera, layout_for


def test_every_species_has_a_visible_sprite_and_action_variants():
    for species in Species:
        idle = sprite_pixels(species, BEHAVIORS[Kind.WAIT].pose)
        assert len(idle) == SIZE
        assert all(len(line) == SIZE for line in idle)
        assert sum(pixel is not None for row in idle for pixel in row) >= 25
        for kind in (Kind.MOVE, Kind.ATTACK, Kind.EAT, Kind.REST):
            assert sprite_pixels(species, BEHAVIORS[kind].pose) != idle


def test_selected_behavior_and_persistent_states():
    actor = Actor(id=1, species=Species.RAVAGER, x=2, y=2)
    assert actor_behavior(actor, Action(Kind.CHARGE, dx=1)).label == 'Charging'
    actor.hibernating = True
    assert actor_behavior(actor, Action(Kind.MOVE, dx=1)).label == 'Hibernating'
    actor.hibernating = False
    actor.sneaking = True
    assert actor_behavior(actor, Action(Kind.MOVE, dx=1)).label == 'Sneaking'


def test_layout_and_camera_for_large_and_resized_window():
    layout = layout_for(1440, 930)
    assert layout.map.right < layout.sidebar.x
    assert layout.sidebar.right <= 1440
    camera = Camera(40, 40, layout.map)
    origin = camera.origin(layout.map)
    point = (origin[0] + camera.tile * 5 + 1, origin[1] + camera.tile * 7 + 1)
    assert camera.tile_at(point, layout.map) == (5, 7)
    camera.zoom(1, layout.map, point)
    assert abs(camera.tile_at(point, layout.map)[0] - 5) <= 1
    resized = layout_for(1024, 768)
    camera.clamp(resized.map)
    assert camera.tile > 0
