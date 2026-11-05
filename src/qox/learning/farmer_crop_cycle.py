"""M16l: bounded, verified crop-cycle shaping for farmer PPO only.

The ledger observes positive physical events, never an attempted action, a
hidden world tile, or a scripted policy. The resulting training-only reward
cannot change QOX's physics or the reward measured at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..entities import Event


PLANT_CREDIT_CAP = 2
PLANT_CREDIT = 0.6
OWN_HARVEST_CREDIT = 0.4
OWN_DEPOSIT_CREDIT = 0.6


@dataclass(frozen=True)
class CropCycleDelta:
    """Physically verified quantities and the optional training bonus."""

    planted: int = 0
    harvested_from_own_plant: float = 0.0
    deposited_from_own_harvest: float = 0.0
    training_bonus: float = 0.0


@dataclass
class FarmerCropCycleLedger:
    """Track a farmer's own seed→crop→store chain within a single episode.

    All matched arms track identical successful events; the strength alone
    determines whether PPO receives the bonus. The maximum direct plant
    bonus is two plants per episode, discouraging plant-only reward farming.
    """

    planted_tiles: set[tuple[int, int]] = field(default_factory=set)
    own_harvest_on_hand: float = 0.0
    plant_credits_issued: int = 0

    def observe(self, events: Iterable[Event], farmer_id: int,
                strength: float) -> CropCycleDelta:
        if not 0 <= strength <= 2:
            raise ValueError("Crop-cycle credit strength must be between 0 and 2")
        planted = 0
        own_harvested = own_deposited = bonus = 0.0
        for event in events:
            if event.actor_id != farmer_id or event.amount <= 0:
                continue
            point = ((event.x, event.y) if event.x is not None and
                     event.y is not None else None)
            if event.kind == "plant":
                # Seed-grain use can consume previously harvested food; do
                # not pay for those same units again at a later deposit.
                self.own_harvest_on_hand = max(
                    0.0, self.own_harvest_on_hand - event.amount)
                if point is not None:
                    self.planted_tiles.add(point)
                    planted += 1
                    if self.plant_credits_issued < PLANT_CREDIT_CAP:
                        bonus += strength * PLANT_CREDIT
                        self.plant_credits_issued += 1
            elif event.kind == "harvest" and point in self.planted_tiles:
                # Only crop physically harvested at this farmer's previously
                # planted coordinate can receive chained harvest credit.
                own_harvested += event.amount
                self.own_harvest_on_hand += event.amount
                bonus += strength * OWN_HARVEST_CREDIT * event.amount
            elif event.kind == "deposit":
                # Crop ownership is fungible after harvesting; count no more
                # than the quantity this farmer harvested from own plots.
                delivered = min(event.amount, self.own_harvest_on_hand)
                own_deposited += delivered
                self.own_harvest_on_hand = max(0.0, self.own_harvest_on_hand - delivered)
                bonus += strength * OWN_DEPOSIT_CREDIT * delivered
            elif (event.kind == "eat" and event.detail == "carried" or
                  event.kind == "festival_contribution"):
                self.own_harvest_on_hand = max(
                    0.0, self.own_harvest_on_hand - event.amount)
        return CropCycleDelta(planted, own_harvested, own_deposited, bonus)
