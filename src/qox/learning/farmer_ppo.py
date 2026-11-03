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

FARMER_SCENARIOS = (
    "agriculture", "refuge", "manure", "farmer_mixed",
    "farmstead", "farmstead_threat", "farmstead_mixed",
)


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
    survival_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.scenario not in FARMER_SCENARIOS:
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
        if not 1.0 <= self.survival_weight <= 20.0:
            raise ValueError("Farmer survival weight must be between 1 and 20")


def _fresh_env(scenario: str, max_steps: int) -> QoxCoreEnv:
    return QoxCoreEnv(scenario=scenario, max_steps=max_steps, scripted_others=True)


def _first_farmer(env: QoxCoreEnv) -> str:
    return next(name for name in env.possible_agents if name.startswith("farmer_"))


def _safety_adjusted_reward(reward: float, components: dict[str, float],
                            weight: float) -> float:
    """Training-only scaling of verified injury/death penalties.

    No change to the world's physical outcomes or reported evaluation reward.
    """
    return reward + (weight - 1.0) * (components.get("injury", 0.0) +
                                     components.get("death", 0.0))


def train_farmer(options: FarmerTrainOptions, output: str | Path,
                 init_checkpoint: str | Path | None = None, *,
                 progress_callback=None, cancel_requested=None) -> dict:
    """Perform PPO updates for at least the requested number of farmer actions."""
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    model = (load_farmer(init_checkpoint) if init_checkpoint is not None else
             FarmerPolicy(options.hidden))
    if model.encoder[0].out_features != options.hidden:
        raise ValueError("Initial farmer checkpoint hidden size mismatch")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)
    if init_checkpoint is not None:
        # A continuation uses the prior optimizer state when available.
        # RNG/rollout trajectories intentionally begin from the new run seed;
        # this is not a bit-identical mid-episode resume.
        from_checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        if from_checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(from_checkpoint["optimizer_state"])


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
        if cancel_requested is not None and cancel_requested():
            from ..training_lab import TrainingCancelled
            raise TrainingCancelled("Training cancelled at rollout boundary")

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
                next_obs, earned, terminated, truncated, infos = env.step({
                    names[i]: int(chosen[i].item())})
                reward = _safety_adjusted_reward(
                    float(earned[names[i]]),
                    infos[names[i]]["reward_components"], options.survival_weight)
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
        if progress_callback is not None:
            point = history[-1]
            progress_callback({"steps": steps, "total": options.steps,
                               "completed_episodes": point["completed_episodes"],
                               "loss": point["loss"],
                               "mean_return": next((float(v) for k, v in point.items()
                                                    if "return" in k and v is not None), None)})

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
    if scenario not in FARMER_SCENARIOS:
        raise ValueError("Invalid farmer evaluation scenario")
    if mode not in ("policy", "scripted", "random",
                    "policy_no_fertilize", "policy_no_refuge"):
        raise ValueError("Invalid evaluation mode")
    if mode.startswith("policy") and checkpoint is None:
        raise ValueError("Policy evaluation needs a checkpoint")
    model = load_farmer(checkpoint) if mode.startswith("policy") else None
    chooser = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        env = _fresh_env(scenario, max_steps)
        states, _ = env.reset(seed=seed + episode)
        initial_home_food = env.world.settlements[0].stored_food
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
                mask = state["action_mask"].copy()
                restricted = ({"policy_no_fertilize": {Kind.FERTILIZE},
                               "policy_no_refuge": {Kind.ENTER_REFUGE}}
                              .get(mode, set()))
                for index, action in enumerate(ACTIONS[Species.FARMER]):
                    if action.kind in restricted:
                        mask[index] = 0
                with torch.no_grad():
                    dist, _ = model(torch.as_tensor(state["observation"]).unsqueeze(0),
                                    torch.as_tensor(mask).unsqueeze(0))
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
        totals = {kind: sum(e.amount for e in events if e.kind == kind)
                  for kind in ("harvest", "deposit", "collect_waste", "fertilize")}
        home = env.world.settlements[0]
        crop_eaten = sum(e.amount for e in env.world.events
                         if e.kind == "graze" and e.detail == "crop")
        results.append({"seed": seed + episode, "return": round(total_reward, 4),
                        "alive": bool(actor), "final_health": round(actor.health, 3) if actor else 0,
                        "harvested": round(totals["harvest"], 3),
                        "deposited": round(totals["deposit"], 3),
                        "waste_collected": round(totals["collect_waste"], 3),
                        "fertilized": round(totals["fertilize"], 3),
                        "sheltered_steps": sheltered_steps,
                        "refuge_entries": sum(e.kind == "enter_refuge" for e in events),
                        "final_home_food": round(home.stored_food, 3),
                        "net_store_change": round(home.stored_food - initial_home_food, 3),
                        "crop_eaten_by_cows": round(crop_eaten, 3)})
    def avg(key: str) -> float:
        return round(float(np.mean([row[key] for row in results])), 4)
    return {"mode": mode, "scenario": scenario, "seed_start": seed,
            "episodes": episodes, "mean_return": avg("return"),
            "mean_deposited": avg("deposited"), "mean_harvested": avg("harvested"),
            "mean_waste_collected": avg("waste_collected"),
            "mean_fertilized": avg("fertilized"),
            "mean_net_store_change": avg("net_store_change"),
            "mean_crop_eaten_by_cows": avg("crop_eaten_by_cows"),
            "mean_sheltered_steps": avg("sheltered_steps"),
            "mean_final_health": avg("final_health"),
            "survival_rate": avg("alive"), "results": results}
