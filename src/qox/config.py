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
    cow_metabolism: float = 0.045
    refuge_ration: float = 0.02
    call_radius: int = 6
    call_duration: int = 3
    call_cooldown: int = 5
    friendly_call_stamina: float = 2.5
    scary_call_stamina: float = 4.5
    nomad_dash_stamina: float = 3.0
    nomad_attack_stamina: float = 5.0
    nomad_attack_damage: float = 8.0
    ravager_sneak_stamina: float = 2.5
    ravager_sneak_energy: float = 0.2
    ravager_charge_stamina: float = 25.0
    ravager_charge_energy: float = 3.0
    ravager_charge_damage: float = 55.0
    ravager_charge_distance: int = 3
    ravager_charge_cooldown: int = 6
    ravager_charge_recovery: int = 2
    ravager_drag_stamina: float = 2.5
    ravager_hibernate_metabolism: float = 0.012

    def __post_init__(self) -> None:
        if self.width < 8 or self.height < 8:
            raise ValueError("World dimensions must each be >= 8")
        if min(self.call_radius, self.call_duration, self.call_cooldown) < 1:
            raise ValueError("Sound radii/durations/cooldowns must be positive")
        if min(self.friendly_call_stamina, self.scary_call_stamina,
               self.nomad_dash_stamina, self.nomad_attack_stamina,
               self.nomad_attack_damage) <= 0:
            raise ValueError("Nomad costs and damage must be positive")
        if min(self.ravager_sneak_stamina, self.ravager_sneak_energy,
               self.ravager_charge_stamina, self.ravager_charge_energy,
               self.ravager_charge_damage, self.ravager_drag_stamina,
               self.ravager_hibernate_metabolism) <= 0:
            raise ValueError("Ravager costs and damage must be positive")
        if min(self.ravager_charge_distance, self.ravager_charge_cooldown,
               self.ravager_charge_recovery) < 1:
            raise ValueError("Ravager charge limits must be positive")
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
        for name in ("grass_growth", "crop_growth", "waste_decay", "cow_metabolism", "refuge_ration"):
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
