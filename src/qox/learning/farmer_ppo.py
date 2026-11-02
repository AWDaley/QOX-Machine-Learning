"""CPU-first farmer PPO and held-out agriculture/refuge/manure benchmarks.

Training uses only local observations and finite world outcomes. The trained
policy is species-shared, with no omniscient knowledge or scripted commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..actions import Kind
from ..agents import scripted_actions
from ..entities import Species
from ..envs.core import QoxCoreEnv
from ..perception import ACTIONS, FARMER_OBSERVATION_VERSION, features_for, observation_size


def _action_names() -> tuple[str, ...]:
    return tuple(a.to_dict()["kind"] + (f"_{a.dx}_{a.dy}" if a.kind == Kind.MOVE else "")
                 for a in ACTIONS[Species.FARMER])


class FarmerPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size(Species.FARMER), hidden),
                                     nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(ACTIONS[Species.FARMER]))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        features = self.encoder(observations)
        logits = self.actor(features).masked_fill(masks <= 0, -1e9)
        return Categorical(logits=logits), self.critic(features).squeeze(-1)


@dataclass(frozen=True)
class FarmerTrainOptions:
    steps: int = 16384
    seed: int = 42
    scenario: str = "farmer_mixed"
    max_steps: int = 96
    worlds: int = 4
    rollout: int = 512
    epochs: int = 4
    minibatch: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.015
    hidden: int = 64

    def __post_init__(self) -> None:
        if self.scenario not in ("agriculture", "refuge", "manure", "farmer_mixed"):
            raise ValueError("Unrecognized farmer training scenario")
        if self.steps < 1 or self.max_steps < 1 or self.worlds < 1:
            raise ValueError("Steps, max_steps and worlds must be positive")
        if self.rollout < self.worlds or self.rollout % self.worlds:
            raise ValueError("Rollout must be divisible by worlds")
        if min(self.epochs, self.minibatch, self.hidden) < 1:
            raise ValueError("PPO architecture and batch sizes must be positive")
        if not (0 < self.lr and 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip < 1 and self.entropy >= 0):
            raise ValueError("Invalid PPO hyperparameters")


def _fresh_env(scenario: str, max_steps: int) -> QoxCoreEnv:
    return QoxCoreEnv(scenario=scenario, max_steps=max_steps, scripted_others=True)


def _first_farmer(env: QoxCoreEnv) -> str:
    return next(name for name in env.possible_agents if name.startswith("farmer_"))


def train_farmer(options: FarmerTrainOptions, output: str | Path) -> dict:
    """Perform PPO updates for at least the requested number of farmer actions."""
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    model = FarmerPolicy(options.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)

    envs = [_fresh_env(options.scenario, options.max_steps) for _ in range(options.worlds)]
    names = [_first_farmer(env) for env in envs]
    episode_counts = [0] * options.worlds
    states = [env.reset(seed=options.seed + 1009 * i)[0][names[i]]
              for i, env in enumerate(envs)]
    active_returns = [0.0] * options.worlds
    completed_returns: list[float] = []
    steps = 0
    history = []
    while steps < options.steps:
        observations, masks, actions, logps, values, rewards, dones = [], [], [], [], [], [], []
        for _ in range(options.rollout // options.worlds):
            inputs = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                     dtype=torch.float32)
            valid = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                    dtype=torch.float32)
            with torch.no_grad():
                distribution, value = model(inputs, valid)
                chosen = distribution.sample()
                old_logp = distribution.log_prob(chosen)
            step_rewards, step_dones = [], []
            for i, env in enumerate(envs):
                next_obs, earned, terminated, truncated, _ = env.step({
                    names[i]: int(chosen[i].item())})
                reward = float(earned[names[i]])
                done = bool(terminated[names[i]] or truncated[names[i]])
                active_returns[i] += reward
                step_rewards.append(reward)
                step_dones.append(float(done))
                if done:
                    completed_returns.append(active_returns[i])
                    active_returns[i] = 0.0
                    episode_counts[i] += 1
                    next_seed = (options.seed + 1009 * i +
                                 100_003 * episode_counts[i])
                    states[i] = env.reset(seed=next_seed)[0][names[i]]
                else:
                    states[i] = next_obs[names[i]]
            observations.append(inputs)
            masks.append(valid)
            actions.append(chosen)
            logps.append(old_logp)
            values.append(value)
            rewards.append(torch.as_tensor(step_rewards, dtype=torch.float32))
            dones.append(torch.as_tensor(step_dones, dtype=torch.float32))

        with torch.no_grad():
            last_inputs = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                          dtype=torch.float32)
            last_masks = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                         dtype=torch.float32)
            _, bootstrap = model(last_inputs, last_masks)
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)
        advantages = torch.zeros_like(reward_tensor)
        gae = torch.zeros(options.worlds)
        for t in reversed(range(len(rewards))):
            following = bootstrap if t == len(rewards) - 1 else value_tensor[t + 1]
            remaining = 1 - done_tensor[t]
            delta = reward_tensor[t] + options.gamma * following * remaining - value_tensor[t]
            gae = delta + options.gamma * options.gae_lambda * remaining * gae
            advantages[t] = gae
        target_returns = (advantages + value_tensor).flatten().detach()
        xs = torch.stack(observations).flatten(0, 1)
        ms = torch.stack(masks).flatten(0, 1)
        acts = torch.stack(actions).flatten().detach()
        old = torch.stack(logps).flatten().detach()
        adv = advantages.flatten().detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = xs.shape[0]
        last_loss = 0.0
        for _ in range(options.epochs):
            for batch in torch.randperm(n).split(options.minibatch):
                distribution, predicted = model(xs[batch], ms[batch])
                ratio = (distribution.log_prob(acts[batch]) - old[batch]).exp()
                clipped = ratio.clamp(1 - options.clip, 1 + options.clip)
                loss = (-torch.minimum(ratio * adv[batch], clipped * adv[batch]).mean()
                        + 0.5 * (target_returns[batch] - predicted).square().mean()
                        - options.entropy * distribution.entropy().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                last_loss = float(loss.detach())
        steps += n
        history.append({"farmer_steps": steps, "completed_episodes": len(completed_returns),
                        "loss": round(last_loss, 5),
                        "mean_recent_return": round(float(np.mean(completed_returns[-30:])), 4)
                        if completed_returns else None})

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "model_hidden": options.hidden, "species": "farmer",
                "obs_version": FARMER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.FARMER),
                "actions": _action_names(), "options": asdict(options),
                "completed_steps": steps}, output)
    report = {"checkpoint": str(output), "farmer_steps": steps,
              "completed_episodes": len(completed_returns),
              "recent_returns": completed_returns[-30:], "history": history}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    return report


def load_farmer(path: str | Path) -> FarmerPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("species") != "farmer" or
            data.get("obs_version") != FARMER_OBSERVATION_VERSION or
            tuple(data.get("observation_features", ())) != features_for(Species.FARMER) or
            tuple(data.get("actions", ())) != _action_names()):
        raise ValueError("Farmer checkpoint observation/action schema mismatch")
    model = FarmerPolicy(int(data["model_hidden"]))
    model.load_state_dict(data["policy_state"])
    model.eval()
    return model


def evaluate_farmer(checkpoint: str | Path | None = None, episodes: int = 20,
                    seed: int = 100_000, scenario: str = "agriculture",
                    max_steps: int = 96, mode: str = "policy") -> dict:
    """Frozen-policy evaluation on seeds distinct from training seeds."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes/max_steps must be positive")
    if scenario not in ("agriculture", "refuge", "manure", "farmer_mixed"):
        raise ValueError("Invalid farmer evaluation scenario")
    if mode not in ("policy", "scripted", "random"):
        raise ValueError("Invalid evaluation mode")
    if mode == "policy" and checkpoint is None:
        raise ValueError("Policy evaluation needs a checkpoint")
    model = load_farmer(checkpoint) if mode == "policy" else None
    chooser = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        env = _fresh_env(scenario, max_steps)
        states, _ = env.reset(seed=seed + episode)
        name = _first_farmer(env)
        aid = env.name_to_id[name]
        total_reward, sheltered_steps = 0.0, 0
        for _ in range(max_steps):
            state = states[name]
            if mode == "random":
                choice = int(chooser.choice(np.flatnonzero(state["action_mask"])))
            elif mode == "scripted":
                proposed = scripted_actions(env.world)[aid]
                choice = ACTIONS[Species.FARMER].index(proposed)
                if not state["action_mask"][choice]:
                    choice = 0
            else:
                with torch.no_grad():
                    dist, _ = model(torch.as_tensor(state["observation"]).unsqueeze(0),
                                    torch.as_tensor(state["action_mask"]).unsqueeze(0))
                    choice = int(dist.probs.argmax(dim=1).item())
            states, rewards, terminated, truncated, _ = env.step({name: choice})
            total_reward += rewards[name]
            farmer = env.world.actors.get(aid)
            if farmer and farmer.sheltered:
                sheltered_steps += 1
            if terminated[name] or truncated[name]:
                break
        actor = env.world.actors.get(aid)
        events = [e for e in env.world.events if e.actor_id == aid]
        def total(kind: str) -> float:
            return sum(e.amount for e in events if e.kind == kind)
        home = env.world.settlements[0]
        results.append({"seed": seed + episode, "return": round(total_reward, 4),
                        "alive": bool(actor), "final_health": round(actor.health, 3) if actor else 0,
                        "harvested": round(total("harvest"), 3),
                        "deposited": round(total("deposit"), 3),
                        "waste_collected": round(total("collect_waste"), 3),
                        "fertilized": round(total("fertilize"), 3),
                        "sheltered_steps": sheltered_steps,
                        "refuge_entries": sum(e.kind == "enter_refuge" for e in events),
                        "final_home_food": round(home.stored_food, 3)})
    def avg(key: str) -> float:
        return round(float(np.mean([row[key] for row in results])), 4)
    return {"mode": mode, "scenario": scenario, "seed_start": seed,
            "episodes": episodes, "mean_return": avg("return"),
            "mean_deposited": avg("deposited"), "mean_harvested": avg("harvested"),
            "mean_waste_collected": avg("waste_collected"),
            "mean_fertilized": avg("fertilized"),
            "mean_sheltered_steps": avg("sheltered_steps"),
            "mean_final_health": avg("final_health"),
            "survival_rate": avg("alive"), "results": results}
