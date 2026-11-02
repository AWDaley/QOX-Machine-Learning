"""Bounded, read-only QOX ecosystem telemetry for graphs and research exports.

Never calls World.step(), samples policy RNG, or changes the environment.
"""

from __future__ import annotations

from collections import Counter, deque
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .entities import Species
from .population import is_juvenile


@dataclass(frozen=True, slots=True)
class Sample:
    tick: int
    populations: dict[str, int]
    juveniles: dict[str, int]
    grass: float
    crops: float
    waste: float
    stored_food: float
    fertility: float
    carcass_meat: float
    sheltered: int
    festivals_succeeded: int
    festivals_failed: int
    # Cumulative real events observed since sampler initialization.
    births: int
    deaths: int
    charge_hits: int
    graze_events: int
    manure_collected: float


class Telemetry:
    """Capture a fixed-size window of resource and population samples.

    Call observe() every simulation tick; it processes the last tick's events
    without scanning the unbounded authoritative `World.events` list. The
    sampling interval controls how often potentially costly tile totals run.
    """

    def __init__(self, *, sample_every: int = 10, limit: int = 240):
        if sample_every < 1 or limit < 2:
            raise ValueError("sample_every must be positive and limit >= 2")
        self.sample_every = sample_every
        self.samples: deque[Sample] = deque(maxlen=limit)
        self.event_totals: Counter[str] = Counter()
        self.event_amounts: Counter[str] = Counter()
        self.last_tick = -1
        self.seed: int | None = None
        self.config: dict | None = None

    def observe(self, world, *, force: bool = False) -> Sample | None:
        if world.tick < self.last_tick:
            raise ValueError("Telemetry requires monotonic world ticks; reset after replay seeking")
        if world.tick == self.last_tick:
            return None
        if self.last_tick >= 0 and world.tick != self.last_tick + 1:
            # Skipped events would make event totals misleading.
            raise ValueError("Telemetry must observe each world tick")
        if self.seed is None:
            self.seed = world.config.seed
            self.config = world.config.as_dict()
        if world.tick > 0:
            for event in world.last_events:
                self.event_totals[event.kind] += 1
                self.event_amounts[event.kind] += event.amount
        self.last_tick = world.tick
        if not (force or world.tick == 0 or world.tick % self.sample_every == 0):
            return None
        pop = Counter(a.species.value for a in world.actors.values())
        juveniles = Counter(a.species.value for a in world.actors.values()
                            if is_juvenile(world, a))
        tiles = world.grid.tiles
        sample = Sample(
            tick=world.tick,
            populations={s.value: pop[s.value] for s in Species},
            juveniles={s.value: juveniles[s.value] for s in Species},
            grass=round(sum(t.grass for t in tiles), 4),
            crops=round(sum(t.crop for t in tiles), 4),
            waste=round(sum(t.waste for t in tiles), 4),
            stored_food=round(sum(s.stored_food for s in world.settlements), 4),
            fertility=round(sum(t.fertility for t in tiles) / max(1, len(tiles)), 4),
            carcass_meat=round(sum(c.meat for c in world.carcasses.values()), 4),
            sheltered=sum(a.sheltered for a in world.actors.values()),
            festivals_succeeded=sum(f.succeeded for f in world.festival_results),
            festivals_failed=sum(not f.succeeded for f in world.festival_results),
            births=self.event_totals["birth"], deaths=self.event_totals["death"],
            charge_hits=self.event_totals["charge_hit"],
            graze_events=self.event_totals["graze"],
            manure_collected=round(self.event_amounts["collect_waste"], 4),
        )
        self.samples.append(sample)
        return sample

    def series(self, key: str) -> list[tuple[int, float]]:
        """A typed chart series; species population keys are prefixed 'pop_'."""
        result = []
        for sample in self.samples:
            if key.startswith("pop_"):
                value = sample.populations.get(key[4:], 0)
            elif key.startswith("juvenile_"):
                value = sample.juveniles.get(key[9:], 0)
            else:
                value = getattr(sample, key)  # documented Sample attribute
                if not isinstance(value, (int, float)):
                    raise ValueError("Chart series must be numerical")
            result.append((sample.tick, float(value)))
        return result

    def report(self, *, policy: str = "unspecified") -> dict:
        return {"schema_version": 1, "seed": self.seed, "config": self.config,
                "policy": policy, "sample_every": self.sample_every,
                "last_tick": self.last_tick, "event_counts": dict(self.event_totals),
                "event_amounts": {k: round(v, 5) for k, v in self.event_amounts.items()},
                "samples": [asdict(s) for s in self.samples],
                "note": "Retained samples are a bounded window; event totals begin at observation."}

    def export_json(self, path: str | Path, *, policy: str = "unspecified") -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.report(policy=policy), indent=2),
                               encoding="utf-8")
        return destination

    def export_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        columns = ["tick", *(f"pop_{s.value}" for s in Species),
                   *(f"juvenile_{s.value}" for s in Species),
                   "grass", "crops", "waste", "stored_food", "fertility",
                   "carcass_meat", "sheltered", "festivals_succeeded",
                   "festivals_failed", "births", "deaths", "charge_hits",
                   "graze_events", "manure_collected"]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for sample in self.samples:
                row = asdict(sample)
                pops = row.pop("populations")
                juveniles = row.pop("juveniles")
                row.update({f"pop_{s.value}": pops[s.value] for s in Species})
                row.update({f"juvenile_{s.value}": juveniles[s.value] for s in Species})
                writer.writerow(row)
        return destination
