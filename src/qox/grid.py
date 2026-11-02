"""Indexed tile grid with bounded resource fields."""

from __future__ import annotations

from .entities import Tile


class Grid:
    def __init__(self, width: int, height: int, tiles: list[Tile]):
        if len(tiles) != width * height:
            raise ValueError("Tile count does not match world dimensions")
        self.width = width
        self.height = height
        self.tiles = tiles

    def contains(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> Tile:
        if not self.contains(x, y):
            raise IndexError(f"Tile outside map: {(x, y)}")
        return self.tiles[y * self.width + x]

    def neighbors(self, x: int, y: int):
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.contains(nx, ny):
                yield nx, ny
