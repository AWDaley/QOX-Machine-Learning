"""PettingZoo ParallelEnv adapter over QOX's deterministic core.

Install ``pip install -e '.[marl]'`` before importing this module.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from ..config import SimulationConfig
from ..perception import ACTIONS, observation_size
from .core import QoxCoreEnv


class QoxParallelEnv(ParallelEnv):
    metadata = {"name": "qox_parallel_v0", "render_modes": ["ansi"],
                "is_parallelizable": True}

    def __init__(self, config: SimulationConfig | None = None,
                 max_steps: int = 128, scenario: str = "natural",
                 render_mode: str | None = None,
                 cow_calls: bool = False, allow_nomad_calls: bool = True,
                 nomad_v2: bool = False):
        if render_mode not in (None, "ansi"):
            raise ValueError("Only ANSI rendering is currently available")
        if scenario in ("farmer_mixed", "nomad_mixed", "ravager_mixed"):
            raise ValueError("Mixed curricula change agent populations across resets; "
                             "use a fixed scenario for the PettingZoo adapter")
        self.render_mode = render_mode
        self.core = QoxCoreEnv(config, max_steps, scenario,
                               cow_calls=cow_calls, allow_nomad_calls=allow_nomad_calls,
                               nomad_v2=nomad_v2)
        self.cow_calls = cow_calls
        self.nomad_v2 = nomad_v2
        self.possible_agents = list(self.core.possible_agents)
        self.agents = list(self.core.agents)
        self.observation_spaces = {}
        self.action_spaces = {}
        for name in self.possible_agents:
            actor = self.core.world.actors[self.core.name_to_id[name]]
            n = len(ACTIONS[actor.species])
            self.observation_spaces[name] = spaces.Dict({
                "observation": spaces.Box(low=-1, high=1,
                                           shape=(observation_size(actor.species, cow_calls=self.cow_calls,
                                                                   nomad_v2=self.nomad_v2),), dtype=np.float32),
                "action_mask": spaces.MultiBinary(n),
            })
            self.action_spaces[name] = spaces.Discrete(n)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        observations, infos = self.core.reset(seed=seed, options=options)
        self.agents = list(self.core.agents)
        return observations, infos

    def step(self, actions):
        result = self.core.step(actions)
        self.agents = list(self.core.agents)
        return result

    def render(self):
        summary = self.core.world.summary()
        text = f"QOX tick={summary['tick']} populations={summary['population']}"
        if self.render_mode == "ansi":
            return text
        return None

    def close(self):
        return self.core.close()


def parallel_env(**kwargs) -> QoxParallelEnv:
    """PettingZoo-standard factory function."""
    return QoxParallelEnv(**kwargs)
