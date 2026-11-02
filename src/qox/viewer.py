"""Optional simple 2D observer. It never owns or changes the world rules."""

from __future__ import annotations

from .agents import actions_for
from .config import SimulationConfig
from .entities import Species
from .world import World


COLORS = {
    Species.FARMER: (70, 140, 245),
    Species.COW: (250, 250, 250),
    Species.NOMAD: (250, 170, 65),
    Species.SCAVENGER: (170, 100, 210),
    Species.RAVAGER: (230, 75, 85),
}


def display(config: SimulationConfig, mode: str = "scripted") -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Viewer requires: python -m pip install -e '.[viewer]'") from exc

    pygame.init()
    tile_px = max(10, min(22, 780 // max(config.width, config.height)))
    width = config.width * tile_px
    height = config.height * tile_px
    screen = pygame.display.set_mode((width, height + 54))
    pygame.display.set_caption("QOX M1 — Deterministic Ecosystem")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    world = World(config)
    paused = False
    single_step = False
    fps = 12
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    single_step = True
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(120, fps + 5)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(1, fps - 5)
        if not paused or single_step:
            world.step(actions_for(world, mode))
            single_step = False
        for y in range(config.height):
            for x in range(config.width):
                t = world.grid.get(x, y)
                brightness = int(38 + 78 * t.fertility + 60 * (t.grass / config.max_grass))
                color = (22, min(220, brightness), 29)
                if t.planted:
                    color = (min(230, 105 + int(t.crop * 14)), 167, 42)
                pygame.draw.rect(screen, color, (x * tile_px, y * tile_px, tile_px, tile_px))
                if t.waste >= 0.4:
                    pygame.draw.circle(screen, (95, 65, 40),
                                       (x * tile_px + tile_px // 2,
                                        y * tile_px + tile_px // 2), max(2, tile_px // 6))
        for settlement in world.settlements:
            pygame.draw.rect(screen, (72, 64, 142),
                             (settlement.x * tile_px, settlement.y * tile_px,
                              tile_px, tile_px))
        for carcass in world.carcasses.values():
            pygame.draw.circle(screen, (123, 43, 33),
                               (carcass.x * tile_px + tile_px // 2,
                                carcass.y * tile_px + tile_px // 2), tile_px // 3)
        for actor in world.actors.values():
            cx, cy = actor.x * tile_px + tile_px // 2, actor.y * tile_px + tile_px // 2
            pygame.draw.circle(screen, COLORS[actor.species], (cx, cy), max(3, tile_px // 3))
            if actor.species == Species.COW:
                pygame.draw.circle(screen, (30, 30, 30), (cx, cy), max(1, tile_px // 9))
        pygame.draw.rect(screen, (18, 23, 29), (0, height, width, 54))
        populations = world.summary()["population"]
        line1 = f"Tick {world.tick}  {'PAUSED' if paused else 'RUNNING'}  {fps} TPS"
        line2 = " ".join(f"{kind[0].upper()}:{n}" for kind, n in populations.items())
        line2 += "   SPACE pause | N step | +/- speed | ESC quit"
        screen.blit(font.render(line1, True, (230, 230, 230)), (9, height + 3))
        screen.blit(font.render(line2, True, (230, 230, 230)), (9, height + 28))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
