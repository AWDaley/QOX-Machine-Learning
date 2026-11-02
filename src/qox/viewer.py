"""Resizable 2D QOX observer, handcrafted pixel-art sprites, and action badges.

The viewer is intentionally read-only: it records the requested actions for
presentation, but only World.step mutates physical state. Panning, zoom, and
sprite animation do not affect policies, RNG, or deterministic replay.
"""

from __future__ import annotations

from .actions import Action, Kind
from .agents import actions_for
from .config import SimulationConfig
from .entities import Species
from .population import is_juvenile
from .sprites import (BEHAVIORS, SpriteAtlas, actor_behavior, draw_badge,
                      sprite_pixels)
from .viewer_layout import Camera, Rect, ViewerLayout, layout_for
from .world import World

SPECIES_COLORS = {
    Species.FARMER: (70, 140, 245),
    Species.COW: (244, 242, 226),
    Species.NOMAD: (250, 170, 65),
    Species.SCAVENGER: (170, 100, 210),
    Species.RAVAGER: (230, 75, 85),
}
PANEL = (22, 32, 45)
PANEL_DARK = (16, 25, 36)
LIGHT = (237, 244, 253)
MUTED = (158, 176, 194)
ACCENT = (245, 200, 90)


# Do not import pygame here; core QOX users should not need it installed.
def _pygame():
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Viewer requires: python -m pip install -e '.[viewer]'") from exc
    return pygame


def _text(screen, font, text: str, x: int, y: int, color=LIGHT, max_width=None) -> None:
    """Avoid label spillover into neighboring panels on compact displays."""
    text = str(text)
    if max_width is not None and font.size(text)[0] > max_width:
        while text and font.size(text + "...")[0] > max_width:
            text = text[:-1]
        text += "..."
    if text:
        screen.blit(font.render(text, True, color), (x, y))


def _rounded(screen, rectangle, fill, border=None, radius=10) -> None:
    pg = _pygame()
    pg.draw.rect(screen, fill, rectangle, border_radius=radius)
    if border:
        pg.draw.rect(screen, border, rectangle, 1, border_radius=radius)


def _meter(screen, x: int, y: int, width: int, label: str,
           amount: float, total: float, color, font) -> None:
    pg = _pygame()
    _text(screen, font, label, x, y - 2, MUTED)
    _text(screen, font, f"{amount:.0f}", x + width - 30, y - 2, LIGHT)
    bar = pg.Rect(x, y + 16, width, 7)
    pg.draw.rect(screen, (45, 59, 73), bar, border_radius=3)
    fraction = min(1.0, max(0.0, amount / max(total, 0.000001)))
    if fraction:
        pg.draw.rect(screen, color, (x, y + 16, max(1, int(width * fraction)), 7),
                     border_radius=3)


def _terrain_color(tile, config):
    brightness = int(38 + 78 * tile.fertility + 60 * (tile.grass / config.max_grass))
    green = (22, min(220, brightness), 29)
    if tile.planted:
        return (min(230, 105 + int(tile.crop * 14)), 167, 42)
    if tile.cover >= 0.5:
        return (max(10, green[0] - 8), max(15, green[1] - 48), 55)
    return green


def draw_world(screen, world: World, camera: Camera, viewport: Rect,
               sprites: SpriteAtlas, actions: dict[int, Action],
               selected_id: int | None = None, *, badges: bool = True,
               faces: dict[int, bool] | None = None) -> None:
    """Draw only visible map tiles and actors within a clipped viewport."""
    pg = _pygame()
    clip = screen.get_clip()
    region = pg.Rect(viewport.x, viewport.y, viewport.width, viewport.height)
    pg.draw.rect(screen, (10, 28, 31), region)
    screen.set_clip(region)
    tile = camera.tile
    ox, oy = camera.origin(viewport)
    start_x = max(0, (viewport.x - ox) // tile)
    start_y = max(0, (viewport.y - oy) // tile)
    end_x = min(world.config.width, (viewport.right - ox) // tile + 2)
    end_y = min(world.config.height, (viewport.bottom - oy) // tile + 2)
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            terrain = world.grid.get(x, y)
            sx, sy = ox + x * tile, oy + y * tile
            pg.draw.rect(screen, _terrain_color(terrain, world.config),
                         (sx, sy, tile, tile))
            if tile >= 26:
                pg.draw.rect(screen, (24, 59, 43), (sx, sy, tile, tile), 1)
            if terrain.waste >= 0.4:
                pg.draw.circle(screen, (95, 65, 40),
                               (sx + tile // 2, sy + tile // 2), max(2, tile // 7))
    for sound in world.calls:
        fraction = max(0.2, 1 - (world.tick - sound.tick) /
                       (world.config.call_duration + 1))
        radius = int(sound.radius * tile * fraction)
        color = (90, 195, 255) if sound.kind == "friendly_call" else (255, 125, 90)
        pg.draw.circle(screen, color, (ox + sound.x * tile + tile // 2,
                                       oy + sound.y * tile + tile // 2),
                       max(2, radius), 1)
    for settlement in world.settlements:
        sx, sy = ox + settlement.x * tile, oy + settlement.y * tile
        pg.draw.rect(screen, (64, 60, 123), (sx, sy, tile, tile), border_radius=3)
        pg.draw.rect(screen, (180, 170, 250), (sx + tile // 4, sy + tile // 4,
                                              tile // 2, tile // 2), 1)
        people = world.refuge_occupants(settlement.id)
        if people:
            pg.draw.circle(screen, (249, 227, 173), (sx + tile - 3, sy + 3),
                           max(3, tile // 6))
    for camp in world.camps:
        pg.draw.circle(screen, (240, 185, 95),
                       (ox + camp.x * tile + tile // 2, oy + camp.y * tile + tile // 2),
                       max(4, tile // 2), 2)
    for festival in world.active_festivals.values():
        progress = (world.tick - festival.start_tick) / max(
            1, festival.end_tick - festival.start_tick)
        pg.draw.circle(screen, (255, 222, 87),
                       (ox + festival.x * tile + tile // 2,
                        oy + festival.y * tile + tile // 2),
                       int((2.5 + progress * 2.0) * tile), 2)
    for carcass in world.carcasses.values():
        pg.draw.circle(screen, (123, 43, 33),
                       (ox + carcass.x * tile + tile // 2,
                        oy + carcass.y * tile + tile // 2), max(2, tile // 3))
    for actor in world.actors.values():
        if actor.sheltered:
            continue
        px, py = ox + actor.x * tile, oy + actor.y * tile
        if (px + tile < viewport.x or py + tile < viewport.y or
                px >= viewport.right or py >= viewport.bottom):
            continue
        juvenile = is_juvenile(world, actor)
        behavior = actor_behavior(actor, actions.get(actor.id))
        sprite = sprites.get(actor.species, behavior, max(12, tile - 2),
                             frame=world.tick // 3, juvenile=juvenile,
                             injured=actor.health < 35,
                             face_left=(faces or {}).get(actor.id, False))
        screen.blit(sprite, (px + (tile - sprite.get_width()) // 2,
                             py + (tile - sprite.get_height()) // 2))
        if actor.id == selected_id:
            pg.draw.rect(screen, (255, 225, 100), (px - 2, py - 2, tile + 4, tile + 4),
                         2, border_radius=4)
        if badges:
            badge_size = max(11, min(15, tile // 2 + 2))
            draw_badge(screen, px + tile - badge_size // 2 - 1,
                       py - badge_size // 3, behavior, badge_size)
    screen.set_clip(clip)
    pg.draw.rect(screen, (55, 79, 83), region, 1, border_radius=4)


def _sidebar(screen, world: World, layout: ViewerLayout, sprites: SpriteAtlas,
             actions: dict[int, Action], selected_id: int | None, font,
             title_font, small_font, mode: str, controller=None) -> None:
    pg = _pygame()
    side = layout.sidebar
    _rounded(screen, (side.x, side.y, side.width, side.height), PANEL,
             (48, 62, 77), radius=11)
    x = side.x + 16
    width = side.width - 32
    yy = side.y + 15
    _text(screen, title_font, "POPULATIONS", x, yy)
    yy += 34
    pop = world.summary()["population"]
    for species in Species:
        sprite = sprites.get(species, BEHAVIORS[Kind.WAIT], 31)
        screen.blit(sprite, (x + 2, yy - 3))
        _text(screen, font, species.value.capitalize(), x + 45, yy + 2)
        count = pop.get(species.value, 0)
        _text(screen, font, str(count), x + width - 36, yy + 2,
              SPECIES_COLORS[species])
        yy += 35
    pg.draw.line(screen, (61, 75, 92), (x, yy + 3), (x + width, yy + 3))
    yy += 18
    selected = world.actors.get(selected_id)
    if selected is None:
        _text(screen, title_font, "SELECT AN ACTOR", x, yy, ACCENT)
        yy += 34
        _text(screen, small_font, "Click any sprite to inspect it.", x, yy, MUTED)
        yy += 25
        _text(screen, small_font, "Action badges show its latest", x, yy, MUTED)
        yy += 19
        _text(screen, small_font, "attempted action or current state.", x, yy, MUTED)
        yy += 38
    else:
        beh = actor_behavior(selected, actions.get(selected.id))
        portrait = sprites.get(selected.species, beh, 47,
                               juvenile=is_juvenile(world, selected),
                               injured=selected.health < 35)
        screen.blit(portrait, (x, yy - 2))
        _text(screen, title_font, f"{selected.species.value.title()} #{selected.id}",
              x + 56, yy, LIGHT, width - 56)
        _text(screen, small_font, "Juvenile" if is_juvenile(world, selected) else "Adult",
              x + 56, yy + 26, MUTED)
        yy += 52
        draw_badge(screen, x, yy, beh, 17)
        _text(screen, font, beh.label, x + 25, yy,
              beh.color, width - 25)
        yy += 30
        _meter(screen, x, yy, width, "Health", selected.health, 100, (238, 112, 112),
               small_font)
        yy += 34
        _meter(screen, x, yy, width, "Energy", selected.energy, 100, (104, 209, 133),
               small_font)
        yy += 34
        _meter(screen, x, yy, width, "Stamina", selected.stamina, 100, (100, 183, 247),
               small_font)
        yy += 37
        _text(screen, small_font, f"Age: {selected.age}   Tile: {selected.x}, {selected.y}",
              x, yy, MUTED, width)
        yy += 20
        if selected.sheltered:
            _text(screen, small_font, "Safe inside settlement", x, yy, ACCENT)
            yy += 20
        elif selected.species == Species.RAVAGER:
            _text(screen, small_font, f"Satiety: {selected.satiety:.2f}", x, yy, MUTED)
            yy += 20
        if controller is not None and not selected.sheltered:
            try:
                probabilities = controller.probabilities(world, selected.id)
                _text(screen, small_font, "Neural action top-2:", x, yy, MUTED)
                yy += 17
                from .perception import action_catalog
                catalog = action_catalog(selected.species, festivals=(mode == "learned-festival"))
                for idx in sorted(range(len(probabilities)),
                                  key=lambda i: probabilities[i], reverse=True)[:2]:
                    _text(screen, small_font,
                          f"{catalog[idx].kind.value}  {probabilities[idx]:.0%}",
                          x + 6, yy, LIGHT, width - 6)
                    yy += 17
            except (ValueError, IndexError, KeyError, AttributeError):
                pass  # Never allow unsupported inspection to interrupt the viewer.
        yy += 11
    if yy < side.bottom - 105:
        pg.draw.line(screen, (61, 75, 92), (x, yy), (x + width, yy))
        yy += 12
        _text(screen, title_font, "ACTION KEY", x, yy)
        yy += 30
        examples = [Kind.MOVE, Kind.GRAZE, Kind.HARVEST, Kind.ATTACK,
                    Kind.FRIENDLY_CALL, Kind.CHARGE, Kind.REST, Kind.START_FARMER_FESTIVAL]
        for i, kind in enumerate(examples):
            r, c = divmod(i, 2)
            rowx, rowy = x + c * (width // 2), yy + r * 26
            if rowy + 25 > side.bottom - 10:
                break
            behavior = BEHAVIORS[kind]
            draw_badge(screen, rowx, rowy, behavior, 14)
            short = {Kind.MOVE: "Move", Kind.GRAZE: "Feed", Kind.HARVEST: "Work",
                     Kind.ATTACK: "Fight", Kind.FRIENDLY_CALL: "Call",
                     Kind.CHARGE: "Charge", Kind.REST: "Rest", 
                     Kind.START_FARMER_FESTIVAL: "Festival"}[kind]
            _text(screen, small_font, short, rowx + 20, rowy, MUTED,
                  width // 2 - 20)


def _header_and_footer(screen, layout: ViewerLayout, world: World,
                       fonts: tuple, fps: int, paused: bool, badges: bool) -> None:
    pg = _pygame()
    title, regular, small = fonts
    w = screen.get_width()
    pg.draw.rect(screen, (15, 24, 35), (0, 0, w, screen.get_height()))
    _text(screen, title, "QOX", 20, 10, ACCENT)
    _text(screen, regular, "ARTIFICIAL LIFE LAB", 90, 20, LIGHT)
    state = "PAUSED" if paused else "LIVE"
    summary = world.summary()
    count = sum(summary["population"].values())
    status = (f"{state}  |  TICK {world.tick:,}  |  {fps} TPS  |  "
              f"{count} actors  |  {summary['festival_successes']} festivals")
    _text(screen, regular, status, max(320, w - 570), 20, MUTED)
    foot = layout.footer
    _rounded(screen, (foot.x, foot.y, foot.width, foot.height - 7), PANEL_DARK,
             (44, 60, 76), radius=8)
    _text(screen, small, "SPACE pause  ·  N step  ·  +/- speed  ·  Wheel zoom  ·  ",
          foot.x + 14, foot.y + 12, LIGHT)
    _text(screen, small, "Arrows / WASD pan  ·  F fit  ·  H badges  ·  F11 fullscreen  ·  ESC exit",
          foot.x + 405, foot.y + 12, MUTED, foot.width - 420)
    if not badges:
        _text(screen, small, "Badges hidden", foot.right - 130, foot.y + 27, ACCENT)


def _load_controller(mode: str, checkpoint, cow_checkpoint, scenario: str):
    if mode not in ("learned-cow", "learned-farmer", "learned-nomad", "learned-herd",
                    "learned-ravager", "learned-scavenger", "learned-festival"):
        return None
    if checkpoint is None:
        raise ValueError("--checkpoint is required for learned actor mode")
    from .learning.inference import (CowController, FarmerController, NomadController,
                                     HerdController, RavagerController, ScavengerController,
                                     FestivalController)
    if mode == "learned-herd" and cow_checkpoint is None:
        raise ValueError("--cow-checkpoint required for learned-herd")
    controller = (CowController(checkpoint) if mode == "learned-cow"
                  else FarmerController(checkpoint) if mode == "learned-farmer"
                  else NomadController(checkpoint) if mode == "learned-nomad"
                  else RavagerController(checkpoint) if mode == "learned-ravager"
                  else ScavengerController(checkpoint) if mode == "learned-scavenger"
                  else FestivalController(checkpoint) if mode == "learned-festival"
                  else HerdController(checkpoint, cow_checkpoint))
    if (mode == "learned-festival" and scenario != "natural" and
            controller.scenario != scenario):
        raise ValueError("Festival checkpoint must match the selected viewer scenario")
    return controller


def display(config: SimulationConfig, mode: str = "scripted", checkpoint=None,
            cow_checkpoint=None, scenario: str = "natural", active_controller=None) -> None:
    pg = _pygame()
    if scenario == "natural":
        world = World(config)
    else:
        from .envs.core import QoxCoreEnv
        event_world = QoxCoreEnv(scenario=scenario)
        event_world.reset(seed=config.seed)
        world = event_world.world
        config = world.config
    controller = active_controller or _load_controller(mode, checkpoint, cow_checkpoint, scenario)
    pg.init()
    info = pg.display.Info()
    desktop_width = info.current_w if info.current_w >= 740 else 1500
    desktop_height = info.current_h if info.current_h >= 520 else 1000
    initial_size = (min(1440, max(740, desktop_width - 80)),
                    min(930, max(500, desktop_height - 100)))
    screen = pg.display.set_mode(initial_size, pg.RESIZABLE)
    pg.display.set_caption("QOX — Artificial Life Lab | Ecosystem Observer")
    title = pg.font.Font(None, 32)
    regular = pg.font.Font(None, 23)
    small = pg.font.Font(None, 19)
    font_set = title, regular, small
    clock = pg.time.Clock()
    layout = layout_for(*screen.get_size())
    camera = Camera(config.width, config.height, layout.map)
    sprites = SpriteAtlas()
    actions: dict[int, Action] = {}
    faces: dict[int, bool] = {}
    selected_id: int | None = None
    paused = False
    advance_one = False
    badges = True
    fps = 12
    running = True
    windowed_size = initial_size
    fullscreen = False
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.VIDEORESIZE and not fullscreen:
                new_size = (max(740, event.w), max(500, event.h))
                screen = pg.display.set_mode(new_size, pg.RESIZABLE)
                layout = layout_for(*new_size)
                camera.clamp(layout.map)
                windowed_size = new_size
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                position = camera.tile_at(event.pos, layout.map)
                if position is not None:
                    selected_id = next((a.id for a in world.actors.values()
                                        if a.position == position and not a.sheltered), None)
            elif event.type == pg.MOUSEWHEEL:
                camera.zoom(event.y, layout.map, pg.mouse.get_pos())
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    paused = not paused
                elif event.key == pg.K_n:
                    advance_one = True
                elif event.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    fps = min(120, fps + 5)
                elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    fps = max(1, fps - 5)
                elif event.key in (pg.K_LEFT, pg.K_a):
                    camera.pan(-4, 0, layout.map)
                elif event.key in (pg.K_RIGHT, pg.K_d):
                    camera.pan(4, 0, layout.map)
                elif event.key in (pg.K_UP, pg.K_w):
                    camera.pan(0, -4, layout.map)
                elif event.key in (pg.K_DOWN, pg.K_s):
                    camera.pan(0, 4, layout.map)
                elif event.key == pg.K_f:
                    camera.fit(layout.map)
                elif event.key == pg.K_h:
                    badges = not badges
                elif event.key == pg.K_F11:
                    if not fullscreen:
                        windowed_size = screen.get_size()
                        screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
                    else:
                        screen = pg.display.set_mode(windowed_size, pg.RESIZABLE)
                    fullscreen = not fullscreen
                    layout = layout_for(*screen.get_size())
                    camera.clamp(layout.map)
        if not paused or advance_one:
            actions = controller.actions(world) if controller else actions_for(world, mode)
            for aid, action in actions.items():
                if action.kind in (Kind.MOVE, Kind.DASH, Kind.SNEAK, Kind.CHARGE):
                    if action.dx:
                        faces[aid] = action.dx < 0
            world.step(actions)
            advance_one = False
            # Visual metadata about dead actors can be released without changing world state.
            if len(faces) > len(world.actors) * 2 + 10:
                faces = {aid: facing for aid, facing in faces.items() if aid in world.actors}
        screen.fill((15, 24, 35))
        _header_and_footer(screen, layout, world, font_set, fps, paused, badges)
        draw_world(screen, world, camera, layout.map, sprites, actions, selected_id,
                   badges=badges, faces=faces)
        _sidebar(screen, world, layout, sprites, actions, selected_id, regular,
                 title, small, mode, controller)
        pg.display.flip()
        clock.tick(fps)
    pg.quit()
