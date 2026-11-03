"""Frozen M4 demonstrations for farmer skill-preserving PPO.

All samples come from local observations encountered by a *greedy* frozen
teacher. The sampled states and action masks are never inserted into a live
world or used for environment transitions during student training.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from ..actions import Kind
from ..entities import Species
from ..perception import ACTIONS
from .farmer_ppo import (_action_names, _first_farmer, _fresh_env,
                         load_farmer)

REHEARSAL_SCENARIOS = ("agriculture", "manure", "farmstead")
FARM_SKILLS = frozenset((Kind.PLANT, Kind.HARVEST, Kind.DEPOSIT,
                         Kind.COLLECT_WASTE, Kind.FERTILIZE))


@dataclass(frozen=True)
class FarmerRehearsalData:
    observations: torch.Tensor
    masks: torch.Tensor
    teacher_probs: torch.Tensor
    sampling_weights: torch.Tensor
    metadata: dict


def collect_farmer_rehearsal(checkpoint: str | Path, *, seed: int = 1_200_000,
                             episodes: int = 4, max_steps: int = 96) -> FarmerRehearsalData:
    """Gather deterministic, disjoint training demonstrations from M4.

    Each scenario uses an isolated 100,000-seed block. A fixed greedy policy
    avoids perturbing the global training RNG or selecting favorable rollouts.
    The KL loss later uses the full masked *distribution*, not hard labels.
    """
    if not 1 <= episodes < 100_000 or max_steps < 1 or not 0 <= seed < 2**32 - 200_000:
        raise ValueError("Invalid farmer rehearsal seed, episodes, or horizon")
    checkpoint = Path(checkpoint)
    model = load_farmer(checkpoint)
    torch.set_num_threads(1)
    seen_obs: list[np.ndarray] = []
    seen_masks: list[np.ndarray] = []
    seen_probs: list[torch.Tensor] = []
    seen_weights: list[float] = []
    details: dict = {}
    names = _action_names()
    for offset, scenario in enumerate(REHEARSAL_SCENARIOS):
        action_counts: dict[str, int] = {}
        visited = 0
        skill_states = 0
        for episode in range(episodes):
            env = _fresh_env(scenario, max_steps)
            states, _ = env.reset(seed=seed + offset * 100_000 + episode)
            farmer = _first_farmer(env)
            for _ in range(max_steps):
                state = states[farmer]
                observation = np.asarray(state["observation"], dtype=np.float32).copy()
                mask = np.asarray(state["action_mask"], dtype=np.float32).copy()
                with torch.no_grad():
                    distribution, _ = model(torch.from_numpy(observation).unsqueeze(0),
                                            torch.from_numpy(mask).unsqueeze(0))
                    choice = int(distribution.probs.argmax(dim=1).item())
                seen_obs.append(observation)
                seen_masks.append(mask)
                seen_probs.append(distribution.probs.squeeze(0).detach().clone())
                kind = ACTIONS[Species.FARMER][choice].kind
                if kind in FARM_SKILLS:
                    seen_weights.append(8.0)
                    skill_states += 1
                elif kind == Kind.MOVE:
                    seen_weights.append(2.0)
                else:
                    seen_weights.append(1.0)
                action_counts[names[choice]] = action_counts.get(names[choice], 0) + 1
                visited += 1
                states, _, terminated, truncated, _ = env.step({farmer: choice})
                if terminated[farmer] or truncated[farmer]:
                    break
        details[scenario] = {"episodes": episodes, "states": visited,
                             "farm_skill_states": skill_states,
                             "greedy_action_counts": action_counts}
    if not seen_obs:
        raise RuntimeError("Rehearsal collector found no living farmer states")
    observations = torch.from_numpy(np.stack(seen_obs))
    masks = torch.from_numpy(np.stack(seen_masks))
    teacher_probs = torch.stack(seen_probs)
    sampling_weights = torch.tensor(seen_weights)
    fingerprint = sha256()
    for array in (observations, masks, teacher_probs, sampling_weights):
        fingerprint.update(array.contiguous().numpy().tobytes())
    metadata = {"baseline_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "seed": seed, "episodes_per_scenario": episodes,
                "max_steps": max_steps, "scenarios": details,
                "total_states": len(seen_obs), "sample_sha256": fingerprint.hexdigest(),
                "teacher_action": "greedy masked distribution",
                "replay_sampling": "8x farming actions, 2x movement, 1x other"}
    return FarmerRehearsalData(observations, masks, teacher_probs,
                               sampling_weights, metadata)
