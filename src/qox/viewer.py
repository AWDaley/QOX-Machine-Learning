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


def display(config: SimulationConfig, mode: str = "scripted",
            checkpoint=None) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Viewer requires: python -m pip install -e '.[viewer]'") from exc

    pygame.init()
    tile_px = max(10, min(22, 780 // max(config.width, config.height)))
    width = config.width * tile_px
    height = config.height * tile_px
    screen = pygame.display.set_mode((width, height + 78))
    pygame.display.set_caption("QOX — Multi-Agent Ecosystem")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    world = World(config)
    controller = None
    if mode in ("learned-cow", "learned-farmer"):
        if checkpoint is None:
            raise ValueError("--checkpoint is required for learned actor mode")
        from .learning.inference import CowController, FarmerController
        controller = (CowController(checkpoint) if mode == "learned-cow"
                      else FarmerController(checkpoint))
    selected_id = None
    paused = False
    single_step = False
    fps = 12
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if my < height:
                    tx, ty = mx // tile_px, my // tile_px
                    selected_id = next((a.id for a in world.actors.values()
                                        if a.position == (tx, ty)), None)
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
            orders = controller.actions(world) if controller else actions_for(world, mode)
            world.step(orders)
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
            if actor.sheltered:
                continue  # occupants are represented by the settlement badge
            cx, cy = actor.x * tile_px + tile_px // 2, actor.y * tile_px + tile_px // 2
            pygame.draw.circle(screen, COLORS[actor.species], (cx, cy), max(3, tile_px // 3))
            if actor.species == Species.COW:
                pygame.draw.circle(screen, (30, 30, 30), (cx, cy), max(1, tile_px // 9))
        for settlement in world.settlements:
            count = world.refuge_occupants(settlement.id)
            if count:
                label = font.render(str(count), True, (255, 245, 210))
                screen.blit(label, (settlement.x * tile_px, settlement.y * tile_px))
        pygame.draw.rect(screen, (18, 23, 29), (0, height, width, 78))
        populations = world.summary()["population"]
        line1 = f"Tick {world.tick}  {'PAUSED' if paused else 'RUNNING'}  {fps} TPS"
        line2 = " ".join(f"{kind[0].upper()}:{n}" for kind, n in populations.items())
        line2 += "   SPACE pause | N step | +/- speed | ESC quit"
        screen.blit(font.render(line1, True, (230, 230, 230)), (9, height + 3))
        screen.blit(font.render(line2, True, (230, 230, 230)), (9, height + 26))
        selected = world.actors.get(selected_id)
        if selected is not None:
            info = (f"#{selected.id} {selected.species.value}  "
                    f"energy:{selected.energy:.1f} hp:{selected.health:.1f} "
                    f"stamina:{selected.stamina:.1f}"
                    f" {'SHELTERED' if selected.sheltered else 'outside'}")
            controlled = (Species.COW if mode == "learned-cow" else
                          Species.FARMER if mode == "learned-farmer" else None)
            if controller and selected.species == controlled:
                from .perception import ACTIONS
                probabilities = controller.probabilities(world, selected.id)
                best = sorted(range(len(probabilities)),
                              key=lambda i: probabilities[i], reverse=True)[:2]
                info += "  " + " ".join(f"{ACTIONS[controlled][i].kind.value}:"
                                       f"{probabilities[i]:.2f}" for i in best)
            screen.blit(font.render(info, True, (230, 230, 230)), (9, height + 52))
        else:
            screen.blit(font.render("Click an actor to inspect its state.", True,
                                    (180, 190, 200)), (9, height + 52))
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
