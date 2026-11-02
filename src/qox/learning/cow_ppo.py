"""Small CPU-first, masked PPO trainer for a shared cow policy.

This is genuine on-policy actor-critic learning, not an imitation of the
scripted baseline. The trainer operates on the same core transitions as the
PettingZoo adapter; all other agents are fixed scripted opponents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..actions import Kind
from ..agents import scripted_actions
from ..entities import Species
from ..envs.core import QoxCoreEnv
from ..perception import (ACTIONS, OBSERVATION_VERSION,
                          COW_CALL_OBSERVATION_VERSION, features_for, observation_size)


class CowPolicy(nn.Module):
    """Shared 2-layer MLP policy and state-value estimator."""

    def __init__(self, hidden: int = 64, cow_calls: bool = False):
        super().__init__()
        self.hidden = hidden
        self.cow_calls = cow_calls
        self.encoder = nn.Sequential(nn.Linear(observation_size(Species.COW,
                                                                 cow_calls=cow_calls), hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(ACTIONS[Species.COW]))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        features = self.encoder(observations)
        logits = self.actor(features).masked_fill(masks <= 0, -1e9)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)


@dataclass(frozen=True)
class TrainOptions:
    steps: int = 8192
    seed: int = 42
    scenario: str = "forage"
    max_steps: int = 96
    worlds: int = 4
    rollout: int = 512
    epochs: int = 4
    minibatch: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    hidden: int = 64
    cow_calls: bool = False

    def __post_init__(self):
        if self.steps < 1 or self.worlds < 1 or self.max_steps < 1:
            raise ValueError("steps, worlds, and max_steps must be positive")
        if self.rollout < self.worlds or self.rollout % self.worlds:
            raise ValueError("rollout must be a multiple of worlds")
        if self.epochs < 1 or self.minibatch < 1 or self.hidden < 1:
            raise ValueError("epochs, minibatch and hidden must be positive")
        if self.scenario not in ("forage", "threat", "nomad_herd"):
            raise ValueError("Cow training supports forage, threat or nomad_herd")
        if self.scenario == "nomad_herd" and not self.cow_calls:
            raise ValueError("nomad_herd requires cow_calls=True / v2 observation")
        if not (0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.clip < 1):
            raise ValueError("Invalid PPO hyperparameters")


def _first_cow(env: QoxCoreEnv) -> str:
    return next(name for name in env.possible_agents if name.startswith("cow_"))


def _fresh_env(scenario: str, seed: int, max_steps: int, *,
               cow_calls: bool = False, allow_nomad_calls: bool = True) -> QoxCoreEnv:
    return QoxCoreEnv(scenario=scenario, max_steps=max_steps,
                      scripted_others=True, config=None, cow_calls=cow_calls,
                      allow_nomad_calls=allow_nomad_calls)


def _obs(env: QoxCoreEnv, cow: str) -> dict:
    return env._observations()[cow]


def train_cow(options: TrainOptions, output: str | Path) -> dict:
    """Train for at least ``steps`` cow decisions (rounded to rollout size)."""
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    model = CowPolicy(options.hidden, cow_calls=options.cow_calls)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)

    environments = [_fresh_env(options.scenario, options.seed + 1009 * j,
                               options.max_steps, cow_calls=options.cow_calls)
                    for j in range(options.worlds)]
    episode_index = [0] * options.worlds
    names = [_first_cow(env) for env in environments]
    observations = []
    for j, env in enumerate(environments):
        obs, _ = env.reset(seed=options.seed + j * 1009)
        observations.append(obs[names[j]])
    total_steps = 0
    completed = 0
    completed_returns = []
    active_returns = [0.0] * options.worlds
    history = []
    while total_steps < options.steps:
        # T x N rollout, with terminal markers for episode boundaries.
        batch_obs = []
        batch_mask = []
        batch_act = []
        batch_logp = []
        batch_value = []
        batch_reward = []
        batch_done = []
        for _ in range(options.rollout // options.worlds):
            o = torch.as_tensor(np.stack([s["observation"] for s in observations]),
                                dtype=torch.float32)
            m = torch.as_tensor(np.stack([s["action_mask"] for s in observations]),
                                dtype=torch.float32)
            with torch.no_grad():
                dist, values = model(o, m)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            reward_vec = []
            done_vec = []
            for j, env in enumerate(environments):
                next_obs, rewards, terminated, truncated, _ = env.step({
                    names[j]: int(actions[j].item())})
                rew = float(rewards[names[j]])
                done = bool(terminated[names[j]] or truncated[names[j]])
                active_returns[j] += rew
                reward_vec.append(rew)
                done_vec.append(float(done))
                if done:
                    completed += 1
                    completed_returns.append(active_returns[j])
                    active_returns[j] = 0.0
                    episode_index[j] += 1
                    # Strictly separate training and evaluation seed ranges.
                    new_seed = (options.seed + 1009 * j
                                + 100_003 * episode_index[j])
                    obs_next, _ = env.reset(seed=new_seed)
                    observations[j] = obs_next[names[j]]
                else:
                    observations[j] = next_obs[names[j]]
            batch_obs.append(o)
            batch_mask.append(m)
            batch_act.append(actions)
            batch_logp.append(log_probs)
            batch_value.append(values)
            batch_reward.append(torch.as_tensor(reward_vec, dtype=torch.float32))
            batch_done.append(torch.as_tensor(done_vec, dtype=torch.float32))
        with torch.no_grad():
            obs_after = torch.as_tensor(
                np.stack([s["observation"] for s in observations]), dtype=torch.float32)
            mask_after = torch.as_tensor(
                np.stack([s["action_mask"] for s in observations]), dtype=torch.float32)
            _, next_value = model(obs_after, mask_after)
        rewards_t = torch.stack(batch_reward)
        dones_t = torch.stack(batch_done)
        values_t = torch.stack(batch_value)
        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(options.worlds)
        for t in reversed(range(rewards_t.shape[0])):
            following = next_value if t == rewards_t.shape[0] - 1 else values_t[t + 1]
            nonterminal = 1 - dones_t[t]
            delta = rewards_t[t] + options.gamma * following * nonterminal - values_t[t]
            gae = (delta + options.gamma * options.gae_lambda * nonterminal * gae)
            advantages[t] = gae
        returns = (advantages + values_t).flatten().detach()
        obs_all = torch.stack(batch_obs).flatten(0, 1)
        mask_all = torch.stack(batch_mask).flatten(0, 1)
        act_all = torch.stack(batch_act).flatten().detach()
        old_logp = torch.stack(batch_logp).flatten().detach()
        adv_all = advantages.flatten().detach()
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std(unbiased=False) + 1e-8)
        n = obs_all.shape[0]
        last_loss = 0.0
        for _ in range(options.epochs):
            permutation = torch.randperm(n)
            for chunk in permutation.split(options.minibatch):
                distribution, values = model(obs_all[chunk], mask_all[chunk])
                ratio = (distribution.log_prob(act_all[chunk]) - old_logp[chunk]).exp()
                surrogate = ratio * adv_all[chunk]
                clipped = ratio.clamp(1 - options.clip, 1 + options.clip) * adv_all[chunk]
                loss = (-torch.minimum(surrogate, clipped).mean()
                        + 0.5 * (returns[chunk] - values).square().mean()
                        - options.entropy * distribution.entropy().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                last_loss = float(loss.detach())
        total_steps += n
        history.append({"cow_steps": total_steps, "completed_episodes": completed,
                        "loss": round(last_loss, 5),
                        "mean_completed_return": round(float(np.mean(completed_returns[-30:])), 4)
                        if completed_returns else None})

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "model_hidden": options.hidden,
                "obs_version": (COW_CALL_OBSERVATION_VERSION if options.cow_calls
                                else OBSERVATION_VERSION),
                "observation_features": features_for(Species.COW,
                                                     cow_calls=options.cow_calls),
                "actions": tuple(a.to_dict()["kind"] + (f"_{a.dx}_{a.dy}"
                                                    if a.kind == Kind.MOVE else "")
                                 for a in ACTIONS[Species.COW]),
                "options": asdict(options), "completed_steps": total_steps}, path)
    report = {"checkpoint": str(path), "cow_steps": total_steps,
              "completed_episodes": completed, "recent_returns": completed_returns[-30:],
              "history": history}
    path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_cow(path: str | Path) -> CowPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    version = data.get("obs_version")
    if version not in (OBSERVATION_VERSION, COW_CALL_OBSERVATION_VERSION):
        raise ValueError("Checkpoint observation schema mismatch")
    cow_calls = version == COW_CALL_OBSERVATION_VERSION
    if tuple(data.get("observation_features", ())) != features_for(Species.COW,
                                                                  cow_calls=cow_calls):
        raise ValueError("Checkpoint observation schema mismatch")
    current_actions = tuple(a.to_dict()["kind"] + (f"_{a.dx}_{a.dy}"
                                                  if a.kind == Kind.MOVE else "")
                            for a in ACTIONS[Species.COW])
    if tuple(data["actions"]) != current_actions:
        raise ValueError("Checkpoint action schema mismatch")
    model = CowPolicy(int(data["model_hidden"]), cow_calls=cow_calls)
    model.load_state_dict(data["policy_state"])
    model.eval()
    return model


def evaluate_cow(checkpoint: str | Path | None = None,
                 episodes: int = 20, seed: int = 100_000,
                 scenario: str = "forage", max_steps: int = 96,
                 mode: str = "policy") -> dict:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes/max_steps must be positive")
    if mode not in ("policy", "policy_nocalls", "random", "scripted", "scripted_nocalls"):
        raise ValueError("Unknown cow evaluation mode")
    if mode.startswith("policy") and checkpoint is None:
        raise ValueError("Policy evaluation needs a checkpoint")
    model = load_cow(checkpoint) if mode.startswith("policy") else None
    selector = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        env = _fresh_env(scenario, seed + episode, max_steps,
                         cow_calls=(model.cow_calls if model is not None
                                    else scenario == "nomad_herd"),
                         allow_nomad_calls=not mode.endswith("nocalls"))
        observations, _ = env.reset(seed=seed + episode)
        cow_name = _first_cow(env)
        ep_return = 0.0
        for _ in range(max_steps):
            obs = observations[cow_name]
            if mode == "random":
                valid = np.flatnonzero(obs["action_mask"])
                action_index = int(selector.choice(valid))
            elif mode.startswith("scripted"):
                action = scripted_actions(env.world)[env.name_to_id[cow_name]]
                try:
                    action_index = ACTIONS[Species.COW].index(action)
                except ValueError:
                    action_index = 0
                if obs["action_mask"][action_index] == 0:
                    action_index = 0
            else:
                with torch.no_grad():
                    x = torch.as_tensor(obs["observation"]).unsqueeze(0)
                    m = torch.as_tensor(obs["action_mask"]).unsqueeze(0)
                    dist, _ = model(x, m)
                    action_index = int(dist.probs.argmax(dim=1).item())
            observations, rewards, terminated, truncated, _ = env.step({cow_name: action_index})
            ep_return += rewards[cow_name]
            if terminated[cow_name] or truncated[cow_name]:
                break
        cow = env.world.actors.get(env.name_to_id[cow_name])
        graze = sum(ev.amount for ev in env.world.events
                    if ev.actor_id == env.name_to_id[cow_name] and ev.kind == "graze")
        results.append({"seed": seed + episode, "return": round(ep_return, 4),
                        "grazed": round(graze, 4), "alive": cow is not None,
                        "final_energy": round(cow.energy, 4) if cow else 0,
                        "in_pasture": int(env.name_to_id[cow_name] in env.reached_goal_ids),
                        "calls": sum(ev.kind in ("friendly_call", "scary_call")
                                     for ev in env.world.events),
                        "ticks": env.world.tick})
    return {"mode": mode, "scenario": scenario, "seed_start": seed,
            "episodes": episodes, "mean_return": round(float(np.mean([x["return"] for x in results])), 4),
            "mean_grazed": round(float(np.mean([x["grazed"] for x in results])), 4),
            "survival_rate": round(float(np.mean([x["alive"] for x in results])), 4),
            "pasture_arrival_rate": round(float(np.mean([x["in_pasture"] for x in results])), 4),
            "mean_calls": round(float(np.mean([x["calls"] for x in results])), 4),
            "mean_final_energy": round(float(np.mean([x["final_energy"] for x in results])), 4),
            "results": results}
