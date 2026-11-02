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
from .sprites import BEHAVIORS, SpriteAtlas, actor_behavior, draw_badge
from .viewer_layout import Camera, Rect, ViewerLayout, layout_for
from .viewer_events import DisplayEvent, EventObserver, capture_actors
from .decision_inspector import capture_decision, with_outcome
from .telemetry import Telemetry
from .replay_session import ReplayCursor
from .viewer_analytics import draw_decision, draw_graphs
from .trace import Trace
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



def _event_effects(screen, camera: Camera, viewport: Rect,
                   observer: EventObserver, small_font) -> None:
    """Paint only world-confirmed, short-lived events. Never mutate physics."""
    pg = _pygame()
    old_clip = screen.get_clip()
    region = pg.Rect(viewport.x, viewport.y, viewport.width, viewport.height)
    screen.set_clip(region)
    tile = camera.tile
    ox, oy = camera.origin(viewport)
    for cue in observer.active():
        if cue.x is None or cue.y is None:
            continue
        cx = ox + cue.x * tile + tile // 2
        cy = oy + cue.y * tile + tile // 2
        if not region.inflate(tile * 3, tile * 3).collidepoint(cx, cy):
            continue
        progress = max(0.0, min(1.0, (observer.tick - cue.tick) / cue.lifetime))
        radius = max(5, int(tile * (0.35 + 0.8 * progress)))
        if cue.effect == "trail" and cue.start is not None:
            ax = ox + cue.start[0] * tile + tile // 2
            ay = oy + cue.start[1] * tile + tile // 2
            if (ax, ay) != (cx, cy):
                pg.draw.line(screen, cue.color, (ax, ay), (cx, cy),
                             max(2, tile // 9))
                pg.draw.circle(screen, cue.color, (cx, cy), max(3, tile // 6), 1)
        elif cue.effect == "ring":
            pg.draw.circle(screen, cue.color, (cx, cy), radius + 3, 2)
            pg.draw.circle(screen, cue.color, (cx, cy), max(3, radius // 2), 1)
        elif cue.effect == "impact":
            pg.draw.line(screen, cue.color, (cx - radius, cy - radius),
                         (cx + radius, cy + radius), 3)
            pg.draw.line(screen, cue.color, (cx - radius, cy + radius),
                         (cx + radius, cy - radius), 3)
        elif cue.effect == "burst":
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1),
                           (1, 1), (-1, 1), (1, -1)):
                pg.draw.line(screen, cue.color,
                             (cx + dx * radius // 2, cy + dy * radius // 2),
                             (cx + dx * radius, cy + dy * radius), 2)
        elif cue.effect == "leaf":
            pg.draw.ellipse(screen, cue.color,
                            (cx - radius // 3, cy - radius, max(4, radius // 2),
                             max(5, radius)))
        else:
            pg.draw.circle(screen, cue.color, (cx, cy), max(3, radius // 3), 2)
            pg.draw.circle(screen, cue.color, (cx - radius // 2, cy - radius), 2)
        if cue.major and tile >= 20 and progress < 0.6:
            label = cue.label if len(cue.label) < 25 else cue.label[:22] + "..."
            # A compact label makes decisive events easy to catch without
            # flooding the world with text for every ordinary move or graze.
            _text(screen, small_font, label, cx + 7, cy - radius - 13,
                  cue.color, max(70, min(180, viewport.right - cx - 9)))
    screen.set_clip(old_clip)


def _event_panel(screen, camera: Camera, viewport: Rect,
                 observer: EventObserver, font, *, important_only: bool = True,
                 species_filter: Species | None = None,
                 actor_filter: int | None = None):
    """Return clickable event-row locations, keeping the world unobstructed."""
    pg = _pygame()
    width = min(440, viewport.width - 24)
    if width < 140:
        return []
    rows = observer.recent(5, important_only=important_only,
                           species=species_filter, actor_id=actor_filter)
    height = 35 + 26 * max(2, len(rows)) + 9
    x, y = viewport.x + 12, viewport.bottom - height - 12
    overlay = pg.Surface((width, height), pg.SRCALPHA)
    overlay.fill((11, 22, 35, 230))
    screen.blit(overlay, (x, y))
    pg.draw.rect(screen, (64, 82, 99), (x, y, width, height), 1, border_radius=5)
    mode = "major" if important_only else "all"
    subgroup = species_filter.value if species_filter is not None else "all species"
    if actor_filter is not None:
        subgroup += f" #{actor_filter}"
    _text(screen, font, f"EVENTS [{mode}, {subgroup}]  B species · Y actor", x + 9, y + 8,
          (245, 211, 115), width - 16)
    hitboxes = []
    for index, event in enumerate(rows):
        top = y + 34 + index * 26
        hitbox = pg.Rect(x + 6, top - 1, width - 12, 25)
        if index % 2 == 0:
            pg.draw.rect(screen, (26, 42, 55), hitbox, border_radius=3)
        aid = f"#{event.actor_id}" if event.actor_id is not None else "World"
        label = f"{event.tick:>5}  {aid:<7} {event.label}"
        _text(screen, font, label, x + 12, top + 2, event.color, width - 22)
        hitboxes.append((hitbox, event))
    if not rows:
        _text(screen, font, "No confirmed events yet", x + 12, y + 39,
              MUTED, width - 20)
    return hitboxes


def _focus_event(camera: Camera, viewport: Rect, event: DisplayEvent) -> None:
    """Camera-only action for clicked feed rows; no world coordinates change."""
    if event.x is None or event.y is None:
        return
    camera.x = (event.x * camera.tile + camera.tile // 2 - viewport.width // 2)
    camera.y = (event.y * camera.tile + camera.tile // 2 - viewport.height // 2)
    camera.clamp(viewport)

def _sidebar(screen, world: World, layout: ViewerLayout, sprites: SpriteAtlas,
             actions: dict[int, Action], selected_id: int | None, font,
             title_font, small_font, mode: str, controller=None,
             observer: EventObserver | None = None, *, tab: str = "actors",
             telemetry: Telemetry | None = None, decision=None,
             observation_scroll: int = 0) -> None:
    pg = _pygame()
    side = layout.sidebar
    if tab == "graphs" and telemetry is not None:
        draw_graphs(screen, side, telemetry, title_font, font, small_font)
        return
    if tab == "neural":
        draw_decision(screen, side, decision, title_font, font, small_font,
                      scroll=observation_scroll)
        return
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
        _text(screen, small_font, "Action badges show confirmed results", x, yy, MUTED)
        yy += 19
        _text(screen, small_font, "or persistent states.", x, yy, MUTED)
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
        yy += 27
        if observer is not None and selected.id in observer.feedback:
            last = observer.feedback[selected.id]
            if last.result != "state":
                note = "Confirmed" if last.result == "confirmed" else "No confirmed outcome"
                tone = (105, 220, 151) if last.result == "confirmed" else (249, 166, 113)
                _text(screen, small_font, f"{last.kind.value}: {note}", x, yy, tone, width)
                yy += 22
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
                       fonts: tuple, fps: int, paused: bool, badges: bool, *,
                       replay_status: str = "", note: str = "") -> None:
    pg = _pygame()
    title, regular, small = fonts
    w = screen.get_width()
    pg.draw.rect(screen, (15, 24, 35), (0, 0, w, screen.get_height()))
    _text(screen, title, "QOX", 20, 10, ACCENT)
    _text(screen, regular, "ARTIFICIAL LIFE LAB", 90, 20, LIGHT)
    state = "REPLAY PAUSED" if replay_status and paused else (
        "REPLAY" if replay_status else "PAUSED" if paused else "LIVE")
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
    _text(screen, small, "1/2/3 tabs · G graphs · I neural · J/K inputs · E events · X export",
          foot.x + 400, foot.y + 12, MUTED, foot.width - 410)
    if replay_status:
        _text(screen, small, "REPLAY " + replay_status +
              "  |  ,/. step  PgUp/Dn jump  [/] events  Home/End",
              foot.x + 14, foot.y + 29, ACCENT, foot.width - 30)
    elif note:
        _text(screen, small, note, foot.x + 14, foot.y + 29, ACCENT,
              foot.width - 30)
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
            cow_checkpoint=None, scenario: str = "natural", active_controller=None,
            replay_trace: Trace | None = None,
            record_trace: str | None = None) -> None:
    pg = _pygame()
    if replay_trace is not None and record_trace is not None:
        raise ValueError("Cannot record and replay simultaneously")
    cursor = ReplayCursor(replay_trace) if replay_trace is not None else None
    if cursor is not None:
        world = cursor.world
        config = world.config
    elif scenario == "natural":
        world = World(config)
    else:
        from .envs.core import QoxCoreEnv
        event_world = QoxCoreEnv(scenario=scenario)
        event_world.reset(seed=config.seed)
        world = event_world.world
        config = world.config
    controller = (None if cursor is not None else
                  active_controller or _load_controller(mode, checkpoint, cow_checkpoint, scenario))
    recording = Trace(config.as_dict()) if record_trace else None
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
    selected_decision = None
    panel_tab = "actors"
    observation_scroll = 0
    telemetry = Telemetry(sample_every=8)
    telemetry.observe(world, force=True)
    feedback_note = ""
    paused = cursor is not None
    advance_one = False
    badges = True
    event_effects = True
    event_feed = True
    important_events_only = True
    species_filter_index = 0
    selected_actor_events = False
    observer = EventObserver()
    event_rows = []
    fps = 12
    running = True
    windowed_size = initial_size
    fullscreen = False
    def jump_replay(target: int) -> None:
        nonlocal world, actions, observer, telemetry, selected_decision, advance_one
        if cursor is None or not 0 <= target <= cursor.length:
            return
        if target == cursor.position + 1:
            before = capture_actors(world)
            selected_decision = (capture_decision(
                world, selected_id, policy_label="recorded actions (no weights)")
                if selected_id else None)
            world = cursor.seek(target)
            actions = cursor.last_actions
            observer.accept(world, actions, before)
            if selected_decision is not None:
                result = observer.feedback.get(selected_decision.actor_id)
                selected_decision = with_outcome(
                    selected_decision, actions.get(selected_decision.actor_id),
                    result.result if result else "missing")
            telemetry.observe(world)
        else:
            world = cursor.seek(target)
            actions = cursor.last_actions
            observer = EventObserver()
            telemetry = Telemetry(sample_every=8)
            telemetry.observe(world, force=True)
            selected_decision = (capture_decision(
                world, selected_id, policy_label="recorded actions (no weights)")
                if selected_id else None)
        advance_one = False
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
                if cursor is not None and layout.footer.contains(event.pos) and event.pos[1] >= layout.footer.y + 32:
                    fraction = (event.pos[0] - layout.footer.x) / layout.footer.width
                    jump_replay(round(max(0, min(1, fraction)) * cursor.length))
                    paused = True
                    continue
                hit = next((cue for box, cue in event_rows if box.collidepoint(event.pos)), None)
                if hit is not None:
                    _focus_event(camera, layout.map, hit)
                    selected_id = hit.actor_id if hit.actor_id in world.actors else None
                    selected_decision = (capture_decision(
                        world, selected_id, controller,
                        policy_label=("recorded actions (no weights)" if cursor else None))
                        if selected_id is not None else None)
                    observation_scroll = 0
                else:
                    position = camera.tile_at(event.pos, layout.map)
                    if position is not None:
                        selected_id = next((a.id for a in world.actors.values()
                                            if a.position == position and not a.sheltered), None)
                        selected_decision = (capture_decision(
                            world, selected_id, controller,
                            policy_label=("recorded actions (no weights)" if cursor else None))
                            if selected_id is not None else None)
                        observation_scroll = 0
            elif event.type == pg.MOUSEWHEEL:
                camera.zoom(event.y, layout.map, pg.mouse.get_pos())
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    paused = not paused
                elif event.key == pg.K_n:
                    advance_one = True
                elif event.key in (pg.K_1, pg.K_2, pg.K_3):
                    panel_tab = {pg.K_1: "actors", pg.K_2: "neural",
                                 pg.K_3: "graphs"}[event.key]
                elif event.key == pg.K_g:
                    panel_tab = "actors" if panel_tab == "graphs" else "graphs"
                elif event.key == pg.K_i:
                    panel_tab = "actors" if panel_tab == "neural" else "neural"
                elif event.key == pg.K_j:
                    observation_scroll = max(0, observation_scroll - 5)
                elif event.key == pg.K_k:
                    observation_scroll += 5
                elif event.key == pg.K_x:
                    try:
                        from pathlib import Path
                        folder = Path.home() / "Documents" / "QOX"
                        stem = f"qox_seed{config.seed}_tick{world.tick}"
                        telemetry.export_json(folder / (stem + ".json"), policy=mode)
                        telemetry.export_csv(folder / (stem + ".csv"))
                        observer.export_json(folder / (stem + "_events.json"))
                        feedback_note = f"Exported telemetry to {folder}"
                    except OSError as exc:
                        feedback_note = f"Export failed: {exc}"
                elif cursor is not None and event.key in (pg.K_COMMA, pg.K_PERIOD,
                                                           pg.K_PAGEUP, pg.K_PAGEDOWN,
                                                           pg.K_HOME, pg.K_END,
                                                           pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    if event.key == pg.K_COMMA:
                        target = cursor.position - 1
                    elif event.key == pg.K_PERIOD:
                        target = cursor.position + 1
                    elif event.key == pg.K_PAGEUP:
                        target = cursor.position - 25
                    elif event.key == pg.K_PAGEDOWN:
                        target = cursor.position + 25
                    elif event.key == pg.K_HOME:
                        target = 0
                    elif event.key == pg.K_END:
                        target = cursor.length
                    else:
                        target = cursor.next_event_tick(
                            forward=event.key == pg.K_RIGHTBRACKET)
                    if target is not None:
                        jump_replay(max(0, min(cursor.length, target)))
                    paused = True
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
                elif event.key == pg.K_e:
                    event_feed = not event_feed
                elif event.key == pg.K_t:
                    important_events_only = not important_events_only
                elif event.key == pg.K_y:
                    selected_actor_events = not selected_actor_events
                elif event.key == pg.K_b and event_feed:
                    species_filter_index = (species_filter_index + 1) % (len(Species) + 1)
                elif event.key == pg.K_v:
                    event_effects = not event_effects
                elif event.key == pg.K_F11:
                    if not fullscreen:
                        windowed_size = screen.get_size()
                        screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
                    else:
                        screen = pg.display.set_mode(windowed_size, pg.RESIZABLE)
                    fullscreen = not fullscreen
                    layout = layout_for(*screen.get_size())
                    camera.clamp(layout.map)
        if cursor is not None:
            if (not paused or advance_one) and cursor.position < cursor.length:
                jump_replay(cursor.position + 1)
            if cursor.position == cursor.length:
                paused = True
            advance_one = False
        elif not paused or advance_one:
            before = capture_actors(world)
            if selected_id is not None:
                selected_decision = capture_decision(world, selected_id, controller)
            actions = controller.actions(world) if controller else actions_for(world, mode)
            for aid, action in actions.items():
                if action.kind in (Kind.MOVE, Kind.DASH, Kind.SNEAK, Kind.CHARGE):
                    if action.dx:
                        faces[aid] = action.dx < 0
            world.step(actions)
            observer.accept(world, actions, before)
            if recording is not None:
                recording.append(world, actions)
            telemetry.observe(world)
            if selected_decision is not None:
                result = observer.feedback.get(selected_decision.actor_id)
                selected_decision = with_outcome(
                    selected_decision, actions.get(selected_decision.actor_id),
                    result.result if result else "missing")
            advance_one = False
            # Visual metadata about dead actors can be released without changing world state.
            if len(faces) > len(world.actors) * 2 + 10:
                faces = {aid: facing for aid, facing in faces.items() if aid in world.actors}
        screen.fill((15, 24, 35))
        _header_and_footer(screen, layout, world, font_set, fps, paused, badges,
                           replay_status=(f"{cursor.position}/{cursor.length}"
                                          if cursor is not None else ""),
                           note=feedback_note)
        if cursor is not None:
            track = pg.Rect(layout.footer.x + 3, layout.footer.y + 39,
                            layout.footer.width - 6, 5)
            pg.draw.rect(screen, (53, 71, 88), track, border_radius=2)
            if cursor.length:
                pg.draw.rect(screen, ACCENT, (track.x, track.y,
                    max(1, int(track.width * cursor.position / cursor.length)), 5),
                    border_radius=2)
        draw_world(screen, world, camera, layout.map, sprites,
                   observer.confirmed_actions, selected_id, badges=badges, faces=faces)
        if event_effects:
            _event_effects(screen, camera, layout.map, observer, small)
        species_filter = (None if species_filter_index == 0 else
                          tuple(Species)[species_filter_index - 1])
        event_rows = (_event_panel(screen, camera, layout.map, observer, small,
                                   important_only=important_events_only,
                                   species_filter=species_filter,
                                   actor_filter=(selected_id if selected_actor_events
                                                 else None))
                      if event_feed else [])
        _sidebar(screen, world, layout, sprites, observer.confirmed_actions,
                 selected_id, regular, title, small, mode, controller, observer,
                 tab=panel_tab, telemetry=telemetry, decision=selected_decision,
                 observation_scroll=observation_scroll)
        pg.display.flip()
        clock.tick(fps)
    pg.quit()
    if recording is not None:
        recording.save(record_trace)
