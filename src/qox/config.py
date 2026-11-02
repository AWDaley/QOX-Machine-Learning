"""Small validated, serializable simulation configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    width: int = 40
    height: int = 40
    seed: int = 42
    ecology_interval: int = 5
    farmers: int = 12
    cows: int = 30
    nomads: int = 6
    scavengers: int = 12
    ravagers: int = 3
    max_grass: float = 10.0
    max_crop: float = 8.0
    max_waste: float = 10.0
    grass_growth: float = 0.45
    crop_growth: float = 0.55
    waste_decay: float = 0.12

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 8:
            raise ValueError("World dimensions must each be >= 8")
        if self.ecology_interval < 1:
            raise ValueError("ecology_interval must be positive")
        for name in ("farmers", "cows", "nomads", "scavengers", "ravagers"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if sum(getattr(self, n) for n in ("farmers", "cows", "nomads", "scavengers", "ravagers")) > self.width * self.height - 2:
            raise ValueError("Population exceeds available land")
        for name in ("max_grass", "max_crop", "max_waste"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("grass_growth", "crop_growth", "waste_decay"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path) -> SimulationConfig:
        with open(path, "rb") as file:
            cfg = tomllib.load(file)
        sections = ("world", "population", "resources")
        extras = set(cfg).difference(sections)
        if extras:
            raise ValueError(f"Unrecognized config sections: {sorted(extras)}")
        allowed = set(cls.__dataclass_fields__)
        fields: dict = {}
        for section in sections:
            for key, value in cfg.get(section, {}).items():
                if key not in allowed:
                    raise ValueError(f"Unrecognized config key: {section}.{key}")
                fields[key] = value
        return cls(**fields)
