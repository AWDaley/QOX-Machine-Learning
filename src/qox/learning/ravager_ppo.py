"""M6 CPU-friendly PPO for ravager hunting and feeding curricula.

Physics (sneak/charge/drag/hibernate) belongs to World. This module only
chooses legal primitive actions and trains against their actual consequences.
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
from ..perception import ACTIONS, RAVAGER_OBSERVATION_VERSION, features_for, observation_size


def action_names() -> tuple[str, ...]:
    return tuple(a.kind.value + (f"_{a.dx}_{a.dy}" if a.kind in
                 (Kind.MOVE, Kind.DASH, Kind.SNEAK, Kind.CHARGE) else "")
                 for a in ACTIONS[Species.RAVAGER])


class RavagerPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size(Species.RAVAGER), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(ACTIONS[Species.RAVAGER]))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        encoded = self.encoder(observations)
        return (Categorical(logits=self.actor(encoded).masked_fill(masks <= 0, -1e9)),
                self.critic(encoded).squeeze(-1))


@dataclass(frozen=True)
class RavagerTrainOptions:
    steps: int = 8192
    seed: int = 42
    scenario: str = "ravager_mixed"
    max_steps: int = 64
    worlds: int = 4
    rollout: int = 256
    epochs: int = 4
    minibatch: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.015
    hidden: int = 64

    def __post_init__(self):
        if self.scenario not in ("ravager_ambush", "ravager_carcass",
                                 "ravager_hibernate", "ravager_mixed"):
            raise ValueError("Unknown ravager curriculum")
        if min(self.steps, self.max_steps, self.worlds, self.rollout,
               self.epochs, self.minibatch, self.hidden) <= 0:
            raise ValueError("Training sizes must be positive")
        if self.rollout % self.worlds:
            raise ValueError("rollout must be divisible by worlds")
        if not (self.lr > 0 and 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip < 1 and self.entropy >= 0):
            raise ValueError("Invalid PPO hyperparameters")


def _name(env: QoxCoreEnv) -> str:
    return next(name for name in env.possible_agents if name.startswith("ravager_"))


def train_ravager(options: RavagerTrainOptions, output: str | Path,
                  init_checkpoint: str | Path | None = None, *,
                 progress_callback=None, cancel_requested=None) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    model = load_ravager(init_checkpoint) if init_checkpoint else RavagerPolicy(options.hidden)
    if model.encoder[0].out_features != options.hidden:
        raise ValueError("Checkpoint hidden layer width mismatch")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)
    if init_checkpoint is not None:
        # A continuation uses the prior optimizer state when available.
        # RNG/rollout trajectories intentionally begin from the new run seed;
        # this is not a bit-identical mid-episode resume.
        from_checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        if from_checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(from_checkpoint["optimizer_state"])

    envs = [QoxCoreEnv(scenario=options.scenario, max_steps=options.max_steps,
                       scripted_others=True) for _ in range(options.worlds)]
    episodes = [0] * options.worlds
    reset_states = [env.reset(seed=options.seed + 1009 * i)[0]
                    for i, env in enumerate(envs)]
    names = [_name(env) for env in envs]
    states = [reset_states[i][name] for i, name in enumerate(names)]
    returns = [0.0] * options.worlds
    completed: list[float] = []
    history: list[dict] = []
    steps = 0
    while steps < options.steps:
        if cancel_requested is not None and cancel_requested():
            from ..training_lab import TrainingCancelled
            raise TrainingCancelled("Training cancelled at rollout boundary")

        obs, masks, choices, logps, values, rewards, dones = [], [], [], [], [], [], []
        for _ in range(options.rollout // options.worlds):
            x = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                dtype=torch.float32)
            m = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                dtype=torch.float32)
            with torch.no_grad():
                distribution, value = model(x, m)
                choice = distribution.sample()
                logp = distribution.log_prob(choice)
            earned, finished = [], []
            for i, env in enumerate(envs):
                new_state, rew, term, trunc, _ = env.step({names[i]: int(choice[i].item())})
                reward = float(rew[names[i]])
                done = bool(term[names[i]] or trunc[names[i]])
                returns[i] += reward
                earned.append(reward)
                finished.append(float(done))
                if done:
                    completed.append(returns[i])
                    returns[i] = 0.0
                    episodes[i] += 1
                    new_seed = options.seed + 1009 * i + 100_003 * episodes[i]
                    next_obs = env.reset(seed=new_seed)[0]
                    names[i] = _name(env)
                    states[i] = next_obs[names[i]]
                else:
                    states[i] = new_state[names[i]]
            obs.append(x)
            masks.append(m)
            choices.append(choice)
            logps.append(logp)
            values.append(value)
            rewards.append(torch.as_tensor(earned, dtype=torch.float32))
            dones.append(torch.as_tensor(finished, dtype=torch.float32))
        with torch.no_grad():
            x = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                dtype=torch.float32)
            m = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                dtype=torch.float32)
            _, bootstrap = model(x, m)
        rewards_t, dones_t, values_t = torch.stack(rewards), torch.stack(dones), torch.stack(values)
        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(options.worlds)
        for t in reversed(range(len(rewards))):
            following = bootstrap if t == len(rewards) - 1 else values_t[t + 1]
            remaining = 1 - dones_t[t]
            delta = rewards_t[t] + options.gamma * following * remaining - values_t[t]
            gae = delta + options.gamma * options.gae_lambda * remaining * gae
            advantages[t] = gae
        targets = (advantages + values_t).flatten().detach()
        xs = torch.stack(obs).flatten(0, 1)
        ms = torch.stack(masks).flatten(0, 1)
        acts = torch.stack(choices).flatten().detach()
        old = torch.stack(logps).flatten().detach()
        adv = advantages.flatten().detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        loss_value = 0.0
        for _ in range(options.epochs):
            for batch in torch.randperm(xs.shape[0]).split(options.minibatch):
                dist, predicted = model(xs[batch], ms[batch])
                ratio = (dist.log_prob(acts[batch]) - old[batch]).exp()
                clipped = ratio.clamp(1 - options.clip, 1 + options.clip)
                loss = (-torch.minimum(ratio * adv[batch], clipped * adv[batch]).mean()
                        + 0.5 * (targets[batch] - predicted).square().mean()
                        - options.entropy * dist.entropy().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                loss_value = float(loss.detach())
        steps += xs.shape[0]
        history.append({"steps": steps, "completed_episodes": len(completed),
                        "loss": round(loss_value, 5),
                        "recent_return": round(float(np.mean(completed[-30:])), 4)
                        if completed else None})
        if progress_callback is not None:
            point = history[-1]
            progress_callback({"steps": steps, "total": options.steps,
                               "completed_episodes": point["completed_episodes"],
                               "loss": point["loss"],
                               "mean_return": next((float(v) for k, v in point.items()
                                                    if "return" in k and v is not None), None)})
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_hidden": options.hidden, "species": "ravager",
                "obs_version": RAVAGER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.RAVAGER),
                "actions": action_names(), "options": asdict(options),
                "completed_steps": steps}, dest)
    report = {"checkpoint": str(dest), "steps": steps,
              "completed_episodes": len(completed),
              "recent_returns": completed[-30:], "history": history}
    dest.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    return report


def load_ravager(path: str | Path) -> RavagerPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("species") != "ravager"
            or data.get("obs_version") != RAVAGER_OBSERVATION_VERSION
            or tuple(data.get("observation_features", ())) != features_for(Species.RAVAGER)
            or tuple(data.get("actions", ())) != action_names()):
        raise ValueError("Ravager checkpoint schema mismatch")
    model = RavagerPolicy(int(data["model_hidden"]))
    model.load_state_dict(data["policy_state"])
    model.eval()
    return model


def evaluate_ravager(checkpoint: str | Path | None = None, *, episodes: int = 16,
                     seed: int = 100_000, scenario: str = "ravager_ambush",
                     max_steps: int = 64, mode: str = "policy_greedy") -> dict:
    """Held-out seeded evaluation; per-episode records expose critical behavior gaps."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("Invalid evaluation budget")
    modes = ("policy", "policy_greedy", "random", "scripted",
             "policy_nocharge", "policy_nosneak", "policy_nohibernate")
    if mode not in modes:
        raise ValueError("Unknown evaluation mode")
    if scenario not in ("ravager_ambush", "ravager_carcass",
                        "ravager_hibernate", "ravager_mixed"):
        raise ValueError("Unknown scenario")
    model = load_ravager(checkpoint) if mode.startswith("policy") else None
    chooser = np.random.default_rng(seed ^ 0x9D274E)
    results = []
    for episode in range(episodes):
        env = QoxCoreEnv(scenario=scenario, max_steps=max_steps, scripted_others=True)
        obs, _ = env.reset(seed=seed + episode)
        name = _name(env)
        aid = env.name_to_id[name]
        original_prey = sum(a.species == Species.COW for a in env.world.actors.values())
        total = 0.0
        sleep_ticks = 0
        for _ in range(max_steps):
            if name not in obs:
                break
            state = obs[name]
            legal = state["action_mask"].copy()
            if mode == "policy_nocharge":
                for i, action in enumerate(ACTIONS[Species.RAVAGER]):
                    if action.kind == Kind.CHARGE:
                        legal[i] = 0
            elif mode == "policy_nosneak":
                for i, action in enumerate(ACTIONS[Species.RAVAGER]):
                    if action.kind == Kind.SNEAK:
                        legal[i] = 0
            elif mode == "policy_nohibernate":
                for i, action in enumerate(ACTIONS[Species.RAVAGER]):
                    if action.kind == Kind.HIBERNATE:
                        legal[i] = 0
            if mode == "random":
                choice = int(chooser.choice(np.flatnonzero(legal)))
            elif mode == "scripted":
                act = scripted_actions(env.world)[aid]
                choice = ACTIONS[Species.RAVAGER].index(act)
                if not legal[choice]:
                    choice = 0
            else:
                with torch.no_grad():
                    dist, _ = model(torch.as_tensor(state["observation"]).unsqueeze(0),
                                    torch.as_tensor(legal).unsqueeze(0))
                    choice = (int(dist.probs.argmax(dim=-1).item())
                              if mode != "policy" else
                              int(chooser.choice(len(ACTIONS[Species.RAVAGER]),
                                                 p=dist.probs.squeeze(0).numpy())))
            obs, reward, term, trunc, _ = env.step({name: choice})
            total += float(reward[name])
            if aid in env.world.actors and env.world.actors[aid].hibernating:
                sleep_ticks += 1
            if term[name] or trunc[name]:
                break
        pred = env.world.actors.get(aid)
        events = [e for e in env.world.events if e.actor_id == aid]
        kinds = [e.kind for e in events]
        results.append({"seed": seed + episode, "return": round(total, 4),
                        "prey_alive": sum(a.species == Species.COW
                                          for a in env.world.actors.values()),
                        "original_prey": original_prey,
                        "charge_hits": kinds.count("charge_hit"),
                        "charge_misses": sum(e.kind == "charge_resolved"
                                              and e.detail != "hit" for e in events),
                        "sneak_steps": kinds.count("sneak"),
                        "drag_steps": kinds.count("drag_carcass"),
                        "carcasses_grabbed": kinds.count("grab_carcass"),
                        "meat_eaten": round(sum(e.amount for e in events
                                                if e.kind == "eat_carcass"), 4),
                        "hibernation_ticks": sleep_ticks,
                        "final_energy": round(pred.energy, 4) if pred else 0.0,
                        "ravager_alive": pred is not None})
    def average(key: str) -> float:
        return round(float(np.mean([r[key] for r in results])), 4)
    report = {"mode": mode, "scenario": scenario, "episodes": episodes,
              "seed_start": seed, "mean_return": average("return"),
              "prey_survival_rate": (average("prey_alive") if scenario != "ravager_mixed"
                                     else None),
              "mean_charge_hits": average("charge_hits"),
              "mean_charge_misses": average("charge_misses"),
              "mean_sneak_steps": average("sneak_steps"),
              "mean_drag_steps": average("drag_steps"),
              "mean_meat_eaten": average("meat_eaten"),
              "mean_hibernation_ticks": average("hibernation_ticks"),
              "mean_final_energy": average("final_energy"),
              "ravager_survival_rate": average("ravager_alive"), "results": results}
    return report
