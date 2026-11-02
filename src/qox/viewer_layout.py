"""Pygame-independent resizable viewer geometry and a bounded zoom/pan camera."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, point: tuple[int, int]) -> bool:
        return self.x <= point[0] < self.right and self.y <= point[1] < self.bottom


@dataclass(frozen=True)
class ViewerLayout:
    map: Rect
    sidebar: Rect
    footer: Rect


def layout_for(width: int, height: int) -> ViewerLayout:
    """A large world view plus a fixed-width inspector, with no fixed resolution."""
    if width < 740 or height < 500:
        raise ValueError("Viewer window must be at least 740x500")
    margin, gap = 14, 14
    header, footer_h = 61, 51
    sidebar_w = max(250, min(330, width // 4))
    map_width = width - margin * 2 - gap - sidebar_w
    map_height = height - header - footer_h - 2 * margin
    world = Rect(margin, header, map_width, map_height)
    sidebar = Rect(world.right + gap, header, sidebar_w, map_height)
    footer = Rect(margin, world.bottom + 9, width - margin * 2, footer_h)
    return ViewerLayout(world, sidebar, footer)


class Camera:
    """World pixel offset: zoom and pan do not alter simulation coordinates."""

    def __init__(self, world_width: int, world_height: int, viewport: Rect):
        self.world_width = world_width
        self.world_height = world_height
        self.tile = 22
        self.x = 0
        self.y = 0
        self.fit(viewport)

    def fit(self, viewport: Rect) -> None:
        self.tile = max(12, min(42, min(viewport.width // self.world_width,
                                         viewport.height // self.world_height)))
        self.x = self.y = 0

    def clamp(self, viewport: Rect) -> None:
        self.x = max(0, min(self.x, self.world_width * self.tile - viewport.width))
        self.y = max(0, min(self.y, self.world_height * self.tile - viewport.height))

    def origin(self, viewport: Rect) -> tuple[int, int]:
        self.clamp(viewport)
        leftover_x = max(0, (viewport.width - self.world_width * self.tile) // 2)
        leftover_y = max(0, (viewport.height - self.world_height * self.tile) // 2)
        return viewport.x + leftover_x - self.x, viewport.y + leftover_y - self.y

    def tile_at(self, point: tuple[int, int], viewport: Rect) -> tuple[int, int] | None:
        if not viewport.contains(point):
            return None
        sx, sy = self.origin(viewport)
        tx, ty = (point[0] - sx) // self.tile, (point[1] - sy) // self.tile
        if 0 <= tx < self.world_width and 0 <= ty < self.world_height:
            return tx, ty
        return None

    def pan(self, dx: int, dy: int, viewport: Rect) -> None:
        self.x += dx * self.tile
        self.y += dy * self.tile
        self.clamp(viewport)

    def zoom(self, delta: int, viewport: Rect,
             anchor: tuple[int, int] | None = None) -> None:
        """Zoom towards the cursor so the same world point remains under it."""
        old_tile = self.tile
        new_tile = max(12, min(48, old_tile + (2 if delta > 0 else -2)))
        if new_tile == old_tile:
            return
        if anchor is None or not viewport.contains(anchor):
            anchor = (viewport.x + viewport.width // 2,
                      viewport.y + viewport.height // 2)
        ox, oy = self.origin(viewport)
        world_x = (anchor[0] - ox) / old_tile
        world_y = (anchor[1] - oy) / old_tile
        self.tile = new_tile
        new_leftover_x = max(0, (viewport.width - self.world_width * new_tile) // 2)
        new_leftover_y = max(0, (viewport.height - self.world_height * new_tile) // 2)
        self.x = int(viewport.x + new_leftover_x + world_x * new_tile - anchor[0])
        self.y = int(viewport.y + new_leftover_y + world_y * new_tile - anchor[1])
        self.clamp(viewport)
