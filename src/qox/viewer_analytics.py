"""Read-only chart and neural-inspection panels for the QOX Pygame viewer."""

from __future__ import annotations

from .entities import Species

DARK = (22, 32, 45)
LIGHT = (237, 244, 253)
MUTED = (158, 176, 194)
GOLD = (245, 200, 90)
SPECIES_COLORS = {
    Species.FARMER: (70, 140, 245),
    Species.COW: (244, 242, 226),
    Species.NOMAD: (250, 170, 65),
    Species.SCAVENGER: (170, 100, 210),
    Species.RAVAGER: (230, 75, 85),
}


def _draw_text(screen, font, label, x: int, y: int, color=LIGHT, width: int | None = None):
    text = str(label)
    if width is not None:
        while text and font.size(text)[0] > width:
            text = text[:-1]
        if text != str(label) and len(text) > 3:
            text = text[:-3] + "..."
    if text:
        screen.blit(font.render(text, True, color), (x, y))


def _frame(screen, rect, title: str, title_font, small):
    import pygame

    pygame.draw.rect(screen, DARK,
                     (rect.x, rect.y, rect.width, rect.height), border_radius=10)
    pygame.draw.rect(screen, (48, 62, 77),
                     (rect.x, rect.y, rect.width, rect.height), 1, border_radius=10)
    _draw_text(screen, title_font, title, rect.x + 12, rect.y + 11, GOLD, rect.width - 24)
    _draw_text(screen, small, "1 ACTORS  |  2 NEURAL  |  3 GRAPHS", rect.x + 12,
               rect.y + 43, MUTED, rect.width - 24)


def _plot(screen, x: int, y: int, width: int, height: int,
          series: list[tuple[list[tuple[int, float]], tuple[int, int, int]]]):
    import pygame

    pygame.draw.rect(screen, (13, 23, 34), (x, y, width, height), border_radius=4)
    pygame.draw.line(screen, (62, 78, 89), (x + 6, y + height - 7),
                     (x + width - 6, y + height - 7), 1)
    all_points = [(tick, val) for values, _ in series for tick, val in values]
    if len(all_points) < 2:
        return
    left = min(t for t, _ in all_points)
    right = max(t for t, _ in all_points)
    floor = min(0.0, min(v for _, v in all_points))
    ceiling = max(floor + 1, max(v for _, v in all_points))
    for values, color in series:
        if len(values) < 2:
            continue
        points = [(x + 7 + int((tick - left) / max(1, right - left) * (width - 16)),
                   y + height - 8 - int((value - floor) / (ceiling - floor) * (height - 18)))
                  for tick, value in values]
        # Lines are presentation only; repeated integer points can overlap.
        pygame.draw.lines(screen, color, False, points, 2)


def draw_graphs(screen, rect, telemetry, title_font, font, small):
    """Population and resource trend charts from a finite sample buffer."""
    _frame(screen, rect, "ECOSYSTEM TRENDS", title_font, small)
    x, width = rect.x + 13, rect.width - 26
    y = rect.y + 79
    _draw_text(screen, font, "Populations", x, y)
    y += 27
    species = list(Species)
    colored = [(telemetry.series(f"pop_{s.value}"), SPECIES_COLORS[s]) for s in species]
    compact = rect.height < 535
    chart_h = max(35, min(145, (rect.height - (230 if compact else 210)) // 3))
    _plot(screen, x, y, width, chart_h, colored)
    y += chart_h + 9
    # Small, readable two-row legend with real population counts.
    for idx, kind in enumerate(species):
        col, row = idx % (3 if compact else 2), idx // (3 if compact else 2)
        latest = telemetry.samples[-1].populations[kind.value] if telemetry.samples else 0
        cell_width = width // (3 if compact else 2)
        _draw_text(screen, small, f"{kind.value[:3]}:{latest}",
                   x + col * cell_width, y + row * 18,
                   SPECIES_COLORS[kind], cell_width - 3)
    y += (2 if compact else 3) * 18 + 8
    _draw_text(screen, font, "Plants and stored food", x, y)
    y += 28
    resources = (("grass", (85, 210, 98)), ("crops", (241, 201, 89)),
                 ("stored_food", (107, 178, 250)))
    _plot(screen, x, y, width, chart_h,
          [(telemetry.series(key), color) for key, color in resources])
    y += chart_h + 8
    latest = telemetry.samples[-1] if telemetry.samples else None
    if latest:
        _draw_text(screen, small,
                   f"Grass {latest.grass:.0f}   Crops {latest.crops:.0f}", x, y,
                   MUTED, width)
        y += 21
        _draw_text(screen, small,
                   f"Stored {latest.stored_food:.1f}  Fertility {latest.fertility:.2f}",
                   x, y, MUTED, width)
        y += 22
        if compact and y + 75 > rect.bottom:
            return
        _draw_text(screen, font, "Life and celebrations", x, y, LIGHT, width)
        y += 28
        _draw_text(screen, small,
                   f"Births {latest.births}   Deaths {latest.deaths}", x, y, MUTED, width)
        y += 21
        _draw_text(screen, small,
                   f"Festivals {latest.festivals_succeeded} succeeded / "
                   f"{latest.festivals_failed} failed", x, y, MUTED, width)
    if telemetry.samples:
        last = telemetry.samples[-1].tick
        oldest = telemetry.samples[0].tick
        _draw_text(screen, small, f"Samples: ticks {oldest}–{last}",
                   x, rect.bottom - 33, MUTED, width)


def draw_decision(screen, rect, snapshot, title_font, font, small,
                  *, scroll: int = 0) -> int:
    """Draw the actual policy's last pre-action local observation and top actions.

    Returns maximum legal scroll offset for keyboard navigation.
    """
    _frame(screen, rect, "NEURAL INSPECTOR", title_font, small)
    x, width = rect.x + 13, rect.width - 26
    y = rect.y + 82
    if snapshot is None:
        _draw_text(screen, font, "Select an actor to inspect.", x, y, MUTED, width)
        return 0
    _draw_text(screen, font, f"{snapshot.species.value.title()} #{snapshot.actor_id}",
               x, y, LIGHT, width)
    y += 24
    _draw_text(screen, small, f"Pre-action tick: {snapshot.tick}", x, y, MUTED, width)
    y += 22
    _draw_text(screen, small, snapshot.policy, x, y, MUTED, width)
    y += 26
    _draw_text(screen, small, f"Requested: {snapshot.action}", x, y, LIGHT, width)
    y += 21
    _draw_text(screen, small, f"Outcome: {snapshot.outcome}", x, y,
               (112, 216, 139) if snapshot.outcome == "confirmed" else GOLD, width)
    y += 30
    _draw_text(screen, font, "Top 5 legal actions", x, y, GOLD, width)
    y += 25
    if snapshot.policy.startswith("neural"):
        for choice in snapshot.top_actions(3 if compact else 5):
            prefix = ">" if choice.chosen else " "
            _draw_text(screen, small,
                       f"{prefix} {choice.label}: {choice.probability:.1%}",
                       x, y, LIGHT if choice.chosen else MUTED, width)
            y += 20
    else:
        _draw_text(screen, small, "No neural policy controls this actor.",
                   x, y, MUTED, width)
        y += 23
    y += 10 if compact else 19
    _draw_text(screen, font, "Observed inputs (J/K scroll)", x, y, GOLD, width)
    y += 26
    available = max(1, (rect.bottom - 37 - y) // 19)
    max_scroll = max(0, len(snapshot.observations) - available)
    scroll = max(0, min(scroll, max_scroll))
    for name, value in snapshot.observations[scroll:scroll + available]:
        _draw_text(screen, small, f"{name}: {value:+.3f}", x, y, MUTED, width)
        y += 19
    _draw_text(screen, small,
               f"{scroll + 1}–{min(len(snapshot.observations), scroll + available)} / "
               f"{len(snapshot.observations)} inputs",
               x, rect.bottom - 28, MUTED, width)
    return max_scroll
