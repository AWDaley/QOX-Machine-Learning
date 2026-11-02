"""M7 CPU-friendly PPO for finite waste and carrion scavenging curricula.

Resource accounting and perception are defined by World, not PPO rewards.
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

from ..actions import Action, Kind
from ..agents import scripted_actions
from ..entities import Species
from ..envs.core import QoxCoreEnv
from ..perception import ACTIONS, SCAVENGER_OBSERVATION_VERSION, features_for, observation_size


def action_names() -> tuple[str, ...]:
    return tuple(a.kind.value + (f"_{a.dx}_{a.dy}" if a.kind in
                 (Kind.MOVE, Kind.DASH, Kind.SNEAK, Kind.CHARGE) else "")
                 for a in ACTIONS[Species.SCAVENGER])


class ScavengerPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size(Species.SCAVENGER), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(ACTIONS[Species.SCAVENGER]))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        encoded = self.encoder(observations)
        return (Categorical(logits=self.actor(encoded).masked_fill(masks <= 0, -1e9)),
                self.critic(encoded).squeeze(-1))


@dataclass(frozen=True)
class ScavengerTrainOptions:
    steps: int = 8192
    seed: int = 42
    scenario: str = "scavenger_mixed"
    max_steps: int = 48
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
        if self.scenario not in ("scavenger_waste", "scavenger_carrion",
                                 "scavenger_competition", "scavenger_mixed"):
            raise ValueError("Unknown scavenger curriculum")
        if min(self.steps, self.max_steps, self.worlds, self.rollout,
               self.epochs, self.minibatch, self.hidden) <= 0:
            raise ValueError("Training sizes must be positive")
        if self.rollout % self.worlds:
            raise ValueError("rollout must be divisible by worlds")
        if not (self.lr > 0 and 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip < 1 and self.entropy >= 0):
            raise ValueError("Invalid PPO hyperparameters")


def _name(env: QoxCoreEnv) -> str:
    return next(name for name in env.possible_agents if name.startswith("scavenger_"))


def train_scavenger(options: ScavengerTrainOptions, output: str | Path,
                  init_checkpoint: str | Path | None = None, *,
                 progress_callback=None, cancel_requested=None) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    model = load_scavenger(init_checkpoint) if init_checkpoint else ScavengerPolicy(options.hidden)
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
                "model_hidden": options.hidden, "species": "scavenger",
                "obs_version": SCAVENGER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.SCAVENGER),
                "actions": action_names(), "options": asdict(options),
                "completed_steps": steps}, dest)
    report = {"checkpoint": str(dest), "steps": steps,
              "completed_episodes": len(completed),
              "recent_returns": completed[-30:], "history": history}
    dest.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    return report


def load_scavenger(path: str | Path) -> ScavengerPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data.get("species") != "scavenger"
            or data.get("obs_version") != SCAVENGER_OBSERVATION_VERSION
            or tuple(data.get("observation_features", ())) != features_for(Species.SCAVENGER)
            or tuple(data.get("actions", ())) != action_names()):
        raise ValueError("Scavenger checkpoint schema mismatch")
    model = ScavengerPolicy(int(data["model_hidden"]))
    model.load_state_dict(data["policy_state"])
    model.eval()
    return model


def evaluate_scavenger(checkpoint: str | Path | None = None, *, episodes: int = 16,
                       seed: int = 100_000, scenario: str = "scavenger_waste",
                       max_steps: int = 48, mode: str = "policy_greedy") -> dict:
    """Held-out paired baseline and food-ablation evaluation with event accounting."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("Invalid evaluation budget")
    modes = ("policy", "policy_greedy", "random", "scripted",
             "policy_nowaste", "policy_nocarrion")
    if mode not in modes:
        raise ValueError("Unknown evaluation mode")
    if scenario not in ("scavenger_waste", "scavenger_carrion",
                        "scavenger_competition", "scavenger_mixed"):
        raise ValueError("Unknown scavenger scenario")
    model = load_scavenger(checkpoint) if mode.startswith("policy") else None
    rng = np.random.default_rng(seed ^ 0x9D274E)
    results = []
    for episode in range(episodes):
        env = QoxCoreEnv(scenario=scenario, max_steps=max_steps, scripted_others=True)
        obs, _ = env.reset(seed=seed + episode)
        name = _name(env)
        aid = env.name_to_id[name]
        total = 0.0
        for _ in range(max_steps):
            if name not in obs:
                break
            state = obs[name]
            legal = state["action_mask"].copy()
            if mode == "policy_nowaste":
                legal[ACTIONS[Species.SCAVENGER].index(Action(Kind.EAT))] = 0
            elif mode == "policy_nocarrion":
                legal[ACTIONS[Species.SCAVENGER].index(Action(Kind.EAT_CARCASS))] = 0
            if mode == "random":
                choice = int(rng.choice(np.flatnonzero(legal)))
            elif mode == "scripted":
                action = scripted_actions(env.world)[aid]
                choice = ACTIONS[Species.SCAVENGER].index(action)
                if not legal[choice]:
                    choice = 0
            else:
                with torch.no_grad():
                    dist, _ = model(torch.as_tensor(state["observation"]).unsqueeze(0),
                                    torch.as_tensor(legal).unsqueeze(0))
                    probs = dist.probs.squeeze(0).numpy()
                    choice = (int(probs.argmax()) if mode != "policy"
                              else int(rng.choice(len(probs), p=probs)))
            obs, rewards, terms, truncs, _ = env.step({name: choice})
            total += float(rewards[name])
            if terms[name] or truncs[name]:
                break
        actual = env.world.actors.get(aid)
        events = env.world.events
        own = [e for e in events if e.actor_id == aid]
        results.append({
            "seed": seed + episode,
            "return": round(total, 4),
            "waste_eaten": round(sum(e.amount for e in own if e.kind == "eat_waste"), 4),
            "carrion_eaten": round(sum(e.amount for e in own if e.kind == "eat_carcass"), 4),
            "farmer_manure_collected": round(sum(e.amount for e in events
                                                 if e.kind == "collect_waste"), 4),
            "fertilizer_applied": round(sum(e.amount for e in events
                                            if e.kind == "fertilize"), 4),
            "recycled_residue": round(sum(e.amount for e in own
                                          if e.kind == "scavenger_residue"), 4),
            "carcass_decay": round(sum(e.amount for e in events
                                       if e.kind == "carcass_decay"), 4),
            "final_energy": round(actual.energy, 4) if actual else 0.0,
            "alive": actual is not None,
        })
    def average(key: str) -> float:
        return round(float(np.mean([result[key] for result in results])), 4)
    return {
        "mode": mode, "scenario": scenario, "episodes": episodes,
        "seed_start": seed, "mean_return": average("return"),
        "mean_waste_eaten": average("waste_eaten"),
        "mean_carrion_eaten": average("carrion_eaten"),
        "mean_farmer_manure_collected": average("farmer_manure_collected"),
        "mean_fertilizer_applied": average("fertilizer_applied"),
        "mean_recycled_residue": average("recycled_residue"),
        "mean_final_energy": average("final_energy"),
        "survival_rate": average("alive"),
        "results": results,
    }
