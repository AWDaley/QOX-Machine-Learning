"""Pixel-art QOX species and action badges.

The sprites are authored as tiny, transparent RGBA pixel canvases, then scaled
with nearest-neighbour interpolation. They are generated in code so the Windows
portable build needs no external art assets or image-loading dependencies.

This module intentionally imports pygame only when drawing: headless simulations
and the learning/training CLI must not require a display library.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action, Kind
from .entities import Actor, Species

RGB = tuple[int, int, int]
SIZE = 20
INK: RGB = (17, 25, 36)
WHITE: RGB = (246, 242, 224)
GOLD: RGB = (250, 198, 92)


@dataclass(frozen=True)
class Behavior:
    label: str
    pose: str
    icon: str
    color: RGB


BEHAVIORS: dict[Kind, Behavior] = {
    Kind.WAIT: Behavior("Idle", "idle", "dot", (164, 177, 190)),
    Kind.MOVE: Behavior("Moving", "moving", "arrow", (181, 220, 244)),
    Kind.DASH: Behavior("Dashing", "moving", "arrow", (253, 192, 97)),
    Kind.SNEAK: Behavior("Sneaking", "sneaking", "eye", (144, 199, 170)),
    Kind.CHARGE: Behavior("Charging", "charging", "bolt", (255, 142, 76)),
    Kind.GRAB_CARCASS: Behavior("Grabbing carcass", "carrying", "bag", GOLD),
    Kind.DROP_CARCASS: Behavior("Dropping carcass", "carrying", "bag", GOLD),
    Kind.HIBERNATE: Behavior("Hibernating", "sleeping", "sleep", (145, 187, 255)),
    Kind.WAKE: Behavior("Waking up", "idle", "sun", (251, 218, 120)),
    Kind.FRIENDLY_CALL: Behavior("Friendly cattle call", "calling", "sound", (97, 211, 255)),
    Kind.SCARY_CALL: Behavior("Scary cattle call", "calling", "sound", (255, 142, 112)),
    Kind.GRAZE: Behavior("Grazing", "feeding", "leaf", (148, 223, 106)),
    Kind.HARVEST: Behavior("Harvesting crops", "working", "tool", GOLD),
    Kind.PLANT: Behavior("Planting", "working", "leaf", (148, 223, 106)),
    Kind.COLLECT_WASTE: Behavior("Collecting manure", "working", "bag", GOLD),
    Kind.FERTILIZE: Behavior("Fertilizing", "working", "leaf", (148, 223, 106)),
    Kind.MILK: Behavior("Milking cattle", "working", "drop", (225, 240, 255)),
    Kind.EAT: Behavior("Eating", "feeding", "leaf", (148, 223, 106)),
    Kind.EAT_CARCASS: Behavior("Eating meat", "feeding", "bone", (255, 170, 139)),
    Kind.DEPOSIT: Behavior("Depositing supplies", "carrying", "bag", GOLD),
    Kind.ENTER_REFUGE: Behavior("Entering refuge", "idle", "home", (151, 175, 255)),
    Kind.REST: Behavior("Resting", "sleeping", "sleep", (145, 187, 255)),
    Kind.ATTACK: Behavior("Fighting", "fighting", "swords", (255, 117, 109)),
    Kind.CONTRIBUTE_FESTIVAL: Behavior("Festival contribution", "festival", "star", GOLD),
    Kind.START_FARMER_FESTIVAL: Behavior("Starting farmer festival", "festival", "star", GOLD),
    Kind.START_NOMAD_FESTIVAL: Behavior("Starting nomad festival", "festival", "star", GOLD),
    Kind.START_COMBINED_FESTIVAL: Behavior("Starting combined festival", "festival", "star", GOLD),
}

SHELTERED = Behavior("Sheltered", "sleeping", "home", (151, 175, 255))
HIBERNATING = BEHAVIORS[Kind.HIBERNATE]
WINDING_UP = Behavior("Charge wind-up", "charging", "bolt", (255, 142, 76))
RECOVERING = Behavior("Charge recovery", "sleeping", "sleep", (145, 187, 255))
DRAGGING = Behavior("Dragging carcass", "carrying", "bag", GOLD)
SNEAKING = BEHAVIORS[Kind.SNEAK]


def actor_behavior(actor: Actor, action: Action | None = None) -> Behavior:
    """Action requested this tick, overridden by authoritative persistent states.

    A badge reflects the actor's *current selected action*, even if an attempted
    action is blocked by world rules. Last-action text in the viewer says so.
    """
    if actor.sheltered:
        return SHELTERED
    if actor.hibernating:
        return HIBERNATING
    if actor.charge_windup > 0:
        return WINDING_UP
    if actor.charge_recovery > 0:
        return RECOVERING
    if actor.dragged_carcass_id is not None:
        return DRAGGING
    if actor.sneaking:
        return SNEAKING
    return BEHAVIORS.get(action.kind, BEHAVIORS[Kind.WAIT]) if action else BEHAVIORS[Kind.WAIT]


class PixelCanvas:
    """Portable, easily tested 20x20 sprite painter; transparent by default."""

    def __init__(self):
        self.pixels: list[list[RGB | None]] = [[None] * SIZE for _ in range(SIZE)]

    def rect(self, x0: int, y0: int, x1: int, y1: int, c: RGB) -> None:
        for y in range(max(y0, 0), min(y1, SIZE)):
            for x in range(max(x0, 0), min(x1, SIZE)):
                self.pixels[y][x] = c

    def dot(self, x: int, y: int, c: RGB) -> None:
        if 0 <= x < SIZE and 0 <= y < SIZE:
            self.pixels[y][x] = c

    def line(self, x0: int, y0: int, x1: int, y1: int, c: RGB) -> None:
        # Bresenham: only crisp, single-pixel lines.
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.dot(x0, y0, c)
            if (x0, y0) == (x1, y1):
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def export(self) -> tuple[tuple[RGB | None, ...], ...]:
        return tuple(tuple(row) for row in self.pixels)


def _farmer(p: PixelCanvas) -> None:
    blue, dark_blue = (64, 145, 245), (39, 84, 167)
    # Straw hat, face, blue overalls, boots, and a small belt.
    p.rect(5, 1, 15, 4, INK)
    p.rect(6, 1, 14, 3, GOLD)
    p.rect(3, 4, 17, 6, INK)
    p.rect(4, 4, 16, 5, (232, 171, 61))
    p.rect(6, 6, 14, 10, INK)
    p.rect(7, 6, 13, 9, (240, 183, 122))
    p.dot(9, 7, INK)
    p.dot(12, 7, INK)
    p.rect(6, 10, 14, 16, INK)
    p.rect(7, 10, 13, 15, blue)
    p.rect(8, 12, 12, 13, (247, 199, 108))
    p.rect(3, 10, 6, 15, INK)
    p.rect(4, 11, 6, 14, blue)
    p.rect(14, 10, 17, 15, INK)
    p.rect(14, 11, 16, 14, blue)
    p.rect(5, 14, 15, 16, dark_blue)
    p.rect(6, 16, 9, 19, INK)
    p.rect(7, 16, 9, 18, dark_blue)
    p.rect(12, 16, 15, 19, INK)
    p.rect(12, 16, 14, 18, dark_blue)


def _cow(p: PixelCanvas) -> None:
    # Chunky side-on cattle with a face, two horns, four hooves and spots.
    p.rect(3, 6, 16, 15, INK)
    p.rect(4, 7, 15, 14, (241, 242, 233))
    p.rect(2, 9, 4, 12, INK)  # tail
    p.rect(1, 8, 3, 10, (225, 228, 222))
    p.rect(6, 8, 9, 11, (49, 58, 67))
    p.rect(10, 10, 13, 13, (49, 58, 67))
    p.rect(4, 14, 7, 18, INK)
    p.rect(5, 14, 7, 17, (225, 227, 222))
    p.rect(12, 14, 15, 18, INK)
    p.rect(12, 14, 14, 17, (225, 227, 222))
    p.rect(13, 4, 18, 12, INK)
    p.rect(14, 5, 18, 10, (244, 243, 233))
    p.rect(13, 2, 15, 5, (234, 227, 189))  # horns
    p.rect(17, 2, 19, 5, (234, 227, 189))
    p.rect(15, 6, 16, 8, (53, 64, 74))
    p.rect(16, 9, 20, 12, INK)
    p.rect(17, 9, 20, 11, (236, 160, 171))
    p.dot(18, 10, (149, 97, 107))


def _nomad(p: PixelCanvas) -> None:
    # Amber hood, scarf, travel cloak and a staff/spear.
    p.rect(5, 1, 14, 8, INK)
    p.rect(6, 2, 13, 7, (208, 116, 47))
    p.rect(7, 4, 13, 8, (241, 181, 121))
    p.rect(6, 7, 14, 9, (247, 200, 94))
    p.rect(9, 5, 12, 6, INK)
    p.rect(5, 9, 15, 17, INK)
    p.rect(6, 9, 14, 16, (227, 140, 63))
    p.rect(7, 9, 10, 15, (246, 172, 79))
    p.rect(9, 13, 13, 14, (102, 78, 79))
    p.rect(6, 16, 9, 19, INK)
    p.rect(11, 16, 14, 19, INK)
    p.rect(2, 3, 4, 17, INK)
    p.rect(3, 3, 4, 16, (198, 202, 214))
    p.dot(3, 1, GOLD)


def _scavenger(p: PixelCanvas) -> None:
    # A violet carrion bird, deliberately unlike the human silhouettes.
    p.rect(4, 8, 17, 15, INK)
    p.rect(5, 8, 16, 14, (161, 103, 214))
    p.rect(6, 9, 12, 13, (106, 68, 160))  # folded wing
    p.rect(2, 12, 7, 14, INK)  # tail
    p.rect(2, 12, 6, 13, (192, 134, 232))
    p.rect(12, 3, 18, 11, INK)  # head
    p.rect(13, 4, 17, 10, (181, 120, 226))
    p.rect(14, 1, 16, 4, (166, 96, 200))  # crest
    p.rect(16, 6, 18, 8, INK)
    p.dot(17, 7, (252, 227, 126))
    p.rect(17, 9, 20, 11, INK)
    p.rect(18, 9, 20, 10, (244, 177, 91))  # hooked beak
    p.rect(7, 14, 9, 18, INK)
    p.rect(13, 14, 15, 18, INK)
    p.dot(6, 17, (241, 181, 98))
    p.dot(12, 17, (241, 181, 98))


def _ravager(p: PixelCanvas) -> None:
    # Broad red predator with raised hackles, large head, fangs and claws.
    p.rect(3, 8, 17, 15, INK)
    p.rect(4, 8, 16, 14, (211, 69, 83))
    p.rect(6, 9, 13, 12, (163, 42, 66))
    for x, y in ((4, 6), (7, 5), (10, 6), (13, 5)):
        p.rect(x, y, x + 2, 9, INK)
        p.dot(x + 1, y + 1, (230, 93, 91))
    p.rect(1, 10, 4, 12, INK)  # tail
    p.rect(2, 9, 4, 11, (219, 76, 89))
    p.rect(4, 14, 7, 19, INK)
    p.rect(12, 14, 15, 19, INK)
    p.rect(5, 15, 7, 17, (148, 45, 66))
    p.rect(13, 15, 15, 17, (148, 45, 66))
    p.rect(13, 5, 19, 13, INK)
    p.rect(14, 6, 19, 11, (226, 77, 83))
    p.rect(16, 8, 18, 9, (255, 208, 87))  # alert eye
    p.rect(16, 11, 20, 13, INK)  # jaw
    p.rect(17, 11, 19, 12, (255, 243, 218))


_DRAW_BASE = {
    Species.FARMER: _farmer,
    Species.COW: _cow,
    Species.NOMAD: _nomad,
    Species.SCAVENGER: _scavenger,
    Species.RAVAGER: _ravager,
}


def sprite_pixels(species: Species, pose: str = "idle", frame: int = 0,
                  injured: bool = False) -> tuple[tuple[RGB | None, ...], ...]:
    """Generate an actual different image for each behavioral pose, without pygame."""
    p = PixelCanvas()
    _DRAW_BASE[species](p)

    if pose in ("sleeping", "sneaking"):
        # Compress the silhouette toward the ground: sleeping vs stealth differ
        # in darkening and the additional visible state badge.
        scaled: list[list[RGB | None]] = [[None] * SIZE for _ in range(SIZE)]
        new_height = 11 if pose == "sleeping" else 13
        for y in range(new_height):
            src_y = min(SIZE - 1, int(y * SIZE / new_height))
            for x, pixel in enumerate(p.pixels[src_y]):
                if pixel is not None:
                    if pose == "sneaking":
                        pixel = tuple(max(0, int(c * 0.72)) for c in pixel)
                    scaled[SIZE - new_height + y][x] = pixel
        p.pixels = scaled
        if pose == "sleeping":
            p.line(14, 3, 17, 3, (166, 200, 255))
            p.line(17, 3, 14, 6, (166, 200, 255))
            p.line(14, 6, 17, 6, (166, 200, 255))
    elif pose == "moving":
        # The leading foot / hoof visibly changes with each motion frame.
        x = 6 if frame % 2 else 13
        p.rect(x, 17, x + 2, 20, INK)
        p.rect(x, 18, x + 2, 19, (226, 204, 134))
        p.dot(1, 17, (225, 210, 155))
    elif pose == "feeding":
        p.rect(16, 15, 19, 17, (113, 202, 81))
        p.rect(17, 13, 18, 16, (164, 221, 104))
        p.dot(15, 15, (244, 217, 113))
    elif pose == "working":
        p.line(15, 5, 17, 15, (249, 216, 133))
        p.rect(14, 5, 19, 7, INK)
        p.rect(15, 5, 18, 6, (208, 218, 221))
        p.dot(18, 14, (176, 124, 70))
    elif pose == "fighting":
        p.line(15, 4, 19, 10, (255, 226, 152))
        p.line(18, 4, 14, 10, (255, 226, 152))
        p.dot(12, 6, (255, 106, 95))
    elif pose == "charging":
        for y, length in ((7, 3), (10, 5), (13, 3)):
            p.rect(0, y, length, y + 1, (255, 175, 78))
        p.rect(16, 8, 18, 9, (255, 243, 142))
    elif pose == "calling":
        p.line(13, 1, 16, 3, (176, 223, 252))
        p.line(16, 3, 18, 6, (176, 223, 252))
        p.dot(19, 5, (242, 244, 217))
    elif pose == "carrying":
        p.rect(14, 12, 20, 18, INK)
        p.rect(15, 13, 19, 17, (167, 114, 75))
        p.rect(16, 11, 18, 13, (225, 183, 114))
    elif pose == "festival":
        for x, y, col in ((2, 3, GOLD), (17, 2, (252, 130, 173)),
                          (1, 12, (138, 221, 234)), (18, 16, GOLD)):
            p.dot(x, y, col)

    if injured:
        p.line(6, 7, 8, 9, (249, 82, 82))
        p.line(8, 7, 6, 9, (249, 82, 82))
    return p.export()


class SpriteAtlas:
    """Cache scaled pygame surfaces per pose/size; all UI-only state."""

    def __init__(self):
        self.cache: dict[tuple, object] = {}

    def get(self, species: Species, behavior: Behavior, size: int, *, frame: int = 0,
            juvenile: bool = False, injured: bool = False, face_left: bool = False):
        import pygame

        size = max(8, int(size * 0.73)) if juvenile else max(8, size)
        key = (species, behavior.pose, frame % 2, size, injured, face_left)
        if key not in self.cache:
            surface = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
            for y, row in enumerate(sprite_pixels(species, behavior.pose, frame, injured)):
                for x, color in enumerate(row):
                    if color is not None:
                        surface.set_at((x, y), color)
            if face_left:
                surface = pygame.transform.flip(surface, True, False)
            self.cache[key] = pygame.transform.scale(surface, (size, size))
        return self.cache[key]


def draw_badge(screen, x: int, y: int, behavior: Behavior, size: int = 13) -> None:
    """Draw a legible tiny pictogram at the *top-right* of an actor's cell."""
    import pygame

    n = max(11, size)
    rect = pygame.Rect(x, y, n, n)
    pygame.draw.rect(screen, (15, 27, 38), rect, border_radius=3)
    pygame.draw.rect(screen, behavior.color, rect, width=1, border_radius=3)
    # Draw in 10x10 logical units so symbols remain consistent across zoom.
    cx, cy = x + n // 2, y + n // 2
    c = behavior.color
    icon = behavior.icon
    if icon == "dot":
        pygame.draw.circle(screen, c, (cx, cy), 2)
    elif icon == "arrow":
        pygame.draw.polygon(screen, c, [(cx - 3, cy - 2), (cx + 3, cy), (cx - 3, cy + 2)])
    elif icon == "leaf":
        pygame.draw.ellipse(screen, c, (cx - 3, cy - 4, 6, 7))
        pygame.draw.line(screen, (19, 34, 29), (cx - 1, cy + 3), (cx + 2, cy - 2), 1)
    elif icon == "swords":
        pygame.draw.line(screen, c, (cx - 3, cy - 3), (cx + 3, cy + 3), 2)
        pygame.draw.line(screen, c, (cx + 3, cy - 3), (cx - 3, cy + 3), 2)
    elif icon == "sleep":
        # A crisp zigzag 'Z', readable without relying on any system font.
        pygame.draw.line(screen, c, (cx - 3, cy - 3), (cx + 3, cy - 3), 1)
        pygame.draw.line(screen, c, (cx + 3, cy - 3), (cx - 3, cy + 3), 1)
        pygame.draw.line(screen, c, (cx - 3, cy + 3), (cx + 3, cy + 3), 1)
    elif icon == "bolt":
        pygame.draw.polygon(screen, c, [(cx, cy - 5), (cx - 3, cy + 1),
                                         (cx, cy + 1), (cx - 1, cy + 5),
                                         (cx + 4, cy - 2), (cx + 1, cy - 2)])
    elif icon == "eye":
        pygame.draw.ellipse(screen, c, (cx - 4, cy - 2, 8, 5), 1)
        pygame.draw.circle(screen, c, (cx, cy), 2)
    elif icon == "sound":
        pygame.draw.circle(screen, c, (cx - 2, cy), 1)
        pygame.draw.arc(screen, c, (cx - 3, cy - 4, 8, 8), -1.0, 1.0, 1)
        pygame.draw.arc(screen, c, (cx - 2, cy - 5, 10, 10), -0.9, 0.9, 1)
    elif icon == "home":
        pygame.draw.polygon(screen, c, [(cx - 4, cy), (cx, cy - 4), (cx + 4, cy)])
        pygame.draw.rect(screen, c, (cx - 3, cy, 6, 4))
    elif icon == "bag":
        pygame.draw.rect(screen, c, (cx - 3, cy - 1, 6, 5))
        pygame.draw.line(screen, c, (cx - 2, cy - 2), (cx + 2, cy - 2), 1)
    elif icon == "drop":
        pygame.draw.polygon(screen, c, [(cx, cy - 4), (cx - 3, cy + 1),
                                         (cx - 2, cy + 4), (cx + 2, cy + 4), (cx + 3, cy + 1)])
    elif icon == "tool":
        pygame.draw.line(screen, c, (cx - 2, cy + 4), (cx + 1, cy - 4), 2)
        pygame.draw.line(screen, c, (cx - 3, cy - 4), (cx + 4, cy - 2), 2)
    elif icon == "bone":
        pygame.draw.line(screen, c, (cx - 3, cy + 3), (cx + 3, cy - 3), 2)
        for u, v in ((-4, 2), (-2, 4), (2, -4), (4, -2)):
            pygame.draw.circle(screen, c, (cx + u, cy + v), 2)
    elif icon in ("star", "sun"):
        pygame.draw.circle(screen, c, (cx, cy), 2)
        for dx, dy in ((-5, 0), (5, 0), (0, -5), (0, 5)):
            pygame.draw.line(screen, c, (cx + dx // 2, cy + dy // 2),
                             (cx + dx, cy + dy), 1)
