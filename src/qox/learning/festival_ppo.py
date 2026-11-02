"""M8: masked PPO for genuine food commitment and festival timing.

This is deliberately a small curriculum, not a claim of learned diplomacy.
The policy's only privileged input is its own local, versioned observation.
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
from ..perception import action_catalog, features_for, observation_size

SCENARIOS = ("festival_farmer", "festival_nomad", "festival_combined", "festival_ravager")
START_ACTIONS = {Kind.START_FARMER_FESTIVAL, Kind.START_NOMAD_FESTIVAL,
                 Kind.START_COMBINED_FESTIVAL}


def species_for(scenario: str) -> Species:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported festival scenario: {scenario}")
    return Species.NOMAD if scenario == "festival_nomad" else Species.FARMER


def catalog_names(species: Species) -> tuple[str, ...]:
    return tuple(a.kind.value + (f"_{a.dx}_{a.dy}" if a.kind == Kind.MOVE else "")
                 for a in action_catalog(species, festivals=True))


class FestivalPolicy(nn.Module):
    def __init__(self, species: Species, hidden: int = 64):
        super().__init__()
        self.species = species
        self.encoder = nn.Sequential(
            nn.Linear(observation_size(species, festivals=True), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(action_catalog(species, festivals=True)))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        encoded = self.encoder(observations)
        logits = self.actor(encoded).masked_fill(masks <= 0, -1e9)
        return Categorical(logits=logits), self.critic(encoded).squeeze(-1)


@dataclass(frozen=True)
class FestivalTrainOptions:
    scenario: str = "festival_farmer"
    steps: int = 8192
    seed: int = 42
    worlds: int = 4
    rollout: int = 256
    max_steps: int = 16
    epochs: int = 3
    minibatch: int = 128
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.015

    def __post_init__(self):
        species_for(self.scenario)
        if min(self.steps, self.worlds, self.rollout, self.max_steps, self.epochs,
               self.minibatch, self.hidden) < 1 or self.rollout % self.worlds:
            raise ValueError("Invalid festival training sizes")
        if not (self.lr > 0 and 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip < 1 and self.entropy >= 0):
            raise ValueError("Invalid PPO hyperparameters")


def _hero(env: QoxCoreEnv) -> str:
    prefix = species_for(env.scenario).value + "_"
    return next(n for n in env.possible_agents if n.startswith(prefix))


def train_festival(options: FestivalTrainOptions, output: str | Path) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    species = species_for(options.scenario)
    model = FestivalPolicy(species, options.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, eps=1e-5)
    envs = [QoxCoreEnv(scenario=options.scenario, max_steps=options.max_steps,
                       scripted_others=True) for _ in range(options.worlds)]
    counts = [0] * options.worlds
    names = [_hero(e) for e in envs]
    states = [e.reset(seed=options.seed + 113 * i)[0][names[i]]
              for i, e in enumerate(envs)]
    returns = [0.0] * options.worlds
    completed = []
    history = []
    steps = 0
    while steps < options.steps:
        obs, masks, choices, logps, values, rewards, dones = [], [], [], [], [], [], []
        for _ in range(options.rollout // options.worlds):
            x = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                dtype=torch.float32)
            m = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                dtype=torch.float32)
            with torch.no_grad():
                dist, value = model(x, m)
                choice = dist.sample()
                logp = dist.log_prob(choice)
            earned, finished = [], []
            for i, env in enumerate(envs):
                state, rs, terms, truncs, _ = env.step({names[i]: int(choice[i])})
                reward = float(rs[names[i]])
                done = bool(terms[names[i]] or truncs[names[i]])
                returns[i] += reward
                earned.append(reward)
                finished.append(float(done))
                if done:
                    completed.append(returns[i])
                    returns[i] = 0.0
                    counts[i] += 1
                    obs_next = env.reset(seed=options.seed + 113 * i +
                                         1009 * counts[i])[0]
                    names[i] = _hero(env)
                    states[i] = obs_next[names[i]]
                else:
                    states[i] = state[names[i]]
            obs.append(x)
            masks.append(m)
            choices.append(choice)
            logps.append(logp)
            values.append(value)
            rewards.append(torch.tensor(earned, dtype=torch.float32))
            dones.append(torch.tensor(finished, dtype=torch.float32))
        with torch.no_grad():
            x = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                dtype=torch.float32)
            m = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                dtype=torch.float32)
            _, bootstrap = model(x, m)
        rewards_t = torch.stack(rewards)
        dones_t = torch.stack(dones)
        values_t = torch.stack(values)
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
        last_loss = 0.0
        for _ in range(options.epochs):
            for batch in torch.randperm(xs.shape[0]).split(options.minibatch):
                dist, predicted = model(xs[batch], ms[batch])
                ratio = (dist.log_prob(acts[batch]) - old[batch]).exp()
                loss = (-torch.minimum(ratio * adv[batch],
                                       ratio.clamp(1 - options.clip,
                                                   1 + options.clip) * adv[batch]).mean()
                        + 0.5 * (targets[batch] - predicted).square().mean()
                        - options.entropy * dist.entropy().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                last_loss = float(loss.detach())
        steps += xs.shape[0]
        history.append({"steps": steps, "episodes": len(completed),
                        "loss": round(last_loss, 5),
                        "recent_return": round(float(np.mean(completed[-30:])), 4)
                        if completed else None})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": model.state_dict(), "model_hidden": options.hidden,
                "species": species.value, "scenario": options.scenario,
                "festival_contract": 1, "observation_features": features_for(species, festivals=True),
                "actions": catalog_names(species), "options": asdict(options),
                "completed_steps": steps}, output)
    report = {"checkpoint": str(output), "scenario": options.scenario, "steps": steps,
              "episodes": len(completed), "history": history,
              "recent_returns": completed[-30:]}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    return report


def load_festival(path: str | Path) -> tuple[FestivalPolicy, str]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    scenario = str(data["scenario"])
    species = species_for(scenario)
    if (data.get("festival_contract") != 1 or data.get("species") != species.value
            or tuple(data.get("observation_features", ())) !=
            features_for(species, festivals=True)
            or tuple(data.get("actions", ())) != catalog_names(species)):
        raise ValueError("Festival checkpoint schema mismatch")
    model = FestivalPolicy(species, int(data["model_hidden"]))
    model.load_state_dict(data["policy_state"])
    model.eval()
    return model, scenario


def evaluate_festival(checkpoint: str | Path | None = None, *,
                      scenario: str = "festival_farmer", episodes: int = 24,
                      max_steps: int = 16, seed: int = 100_000,
                      mode: str = "policy_greedy") -> dict:
    species = species_for(scenario)
    if episodes < 1 or max_steps < 1:
        raise ValueError("Invalid evaluation budget")
    if mode not in ("policy", "policy_greedy", "random", "scripted", "no_start"):
        raise ValueError("Unknown festival evaluation mode")
    model = None
    if mode.startswith("policy") and checkpoint is None:
        raise ValueError("A festival checkpoint is required for policy evaluation")
    if mode.startswith("policy"):
        model, trained = load_festival(checkpoint)
        if trained != scenario:
            raise ValueError("Checkpoint curriculum mismatch")
    rng = np.random.default_rng(seed ^ 0xCABEC0DE)
    catalog = action_catalog(species, festivals=True)
    results = []
    for i in range(episodes):
        env = QoxCoreEnv(scenario=scenario, max_steps=max_steps, scripted_others=True)
        observations, _ = env.reset(seed=seed + i)
        name = _hero(env)
        aid = env.name_to_id[name]
        episode_return = 0.0
        for _ in range(max_steps):
            if name not in observations:
                break
            state = observations[name]
            legal = state["action_mask"].copy()
            if mode == "no_start":
                for idx, action in enumerate(catalog):
                    if action.kind in START_ACTIONS:
                        legal[idx] = 0
            if mode in ("random", "no_start"):
                choice = int(rng.choice(np.flatnonzero(legal)))
            elif mode == "scripted":
                chosen = scripted_actions(env.world)[aid]
                choice = catalog.index(chosen) if chosen in catalog else 0
                if not legal[choice]:
                    choice = 0
            else:
                with torch.no_grad():
                    dist, _ = model(torch.as_tensor(state["observation"]).unsqueeze(0),
                                    torch.as_tensor(legal).unsqueeze(0))
                    p = dist.probs.squeeze(0).numpy()
                    choice = (int(np.argmax(p)) if mode == "policy_greedy"
                              else int(rng.choice(len(p), p=p)))
            observations, rewards, terms, truncs, _ = env.step({name: choice})
            episode_return += float(rewards[name])
            if terms[name] or truncs[name]:
                break
        world = env.world
        events = world.events
        results.append({"seed": seed + i, "return": round(episode_return, 4),
                        "successes": world.summary()["festival_successes"],
                        "failures": world.summary()["festival_failures"],
                        "proposals": sum(e.kind == "festival_started" for e in events),
                        "contributed_food": round(sum(e.amount for e in events
                                                          if e.kind == "festival_contribution"
                                                          and e.detail.startswith("farmer")), 3),
                        "contributed_milk": round(sum(e.amount for e in events
                                                          if e.kind == "festival_contribution"
                                                          and e.detail.startswith("nomad")), 3),
                        "final_survival_store": round(world.settlements[0].stored_food, 3),
                        "ravager_nearby": sum(e.kind == "festival_failed"
                                              and "predator" in e.detail for e in events)})
    def avg(key):
        return round(float(np.mean([r[key] for r in results])), 4)
    return {"mode": mode, "scenario": scenario, "episodes": episodes,
            "mean_return": avg("return"), "success_rate": avg("successes"),
            "failure_rate": avg("failures"), "mean_proposals": avg("proposals"),
            "mean_grain_contributed": avg("contributed_food"),
            "mean_milk_contributed": avg("contributed_milk"),
            "mean_final_survival_store": avg("final_survival_store"),
            "results": results}
