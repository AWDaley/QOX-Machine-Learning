"""M5 CPU PPO for one nomad policy, with herd migration/defense evaluation.

Cattle use a *scripted*, perceptual call responder during the first nomad
curriculum. This does not claim to train the cattle call-response policy.
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
from ..perception import (ACTIONS, NOMAD_OBSERVATION_VERSION,
                          NOMAD_RELATIONAL_OBSERVATION_VERSION, features_for,
                          observation_size)


def action_names() -> tuple[str, ...]:
    return tuple(a.kind.value + (f"_{a.dx}_{a.dy}" if a.kind in (Kind.MOVE, Kind.DASH)
                                  else "") for a in ACTIONS[Species.NOMAD])


class NomadPolicy(nn.Module):
    def __init__(self, hidden: int = 64, *, nomad_v2: bool = False):
        super().__init__()
        self.nomad_v2 = nomad_v2
        self.encoder = nn.Sequential(
            nn.Linear(observation_size(Species.NOMAD, nomad_v2=nomad_v2), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, len(ACTIONS[Species.NOMAD]))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor, masks: torch.Tensor):
        features = self.encoder(observations)
        return (Categorical(logits=self.actor(features).masked_fill(masks <= 0, -1e9)),
                self.critic(features).squeeze(-1))


@dataclass(frozen=True)
class NomadTrainOptions:
    steps: int = 16384
    seed: int = 42
    scenario: str = "nomad_herd"
    max_steps: int = 96
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
    nomad_v2: bool = False

    def __post_init__(self):
        if self.scenario not in ("nomad_signal", "nomad_herd", "nomad_defend", "nomad_mixed"):
            raise ValueError("Unknown nomad training scenario")
        if min(self.steps, self.max_steps, self.worlds, self.rollout,
               self.epochs, self.minibatch, self.hidden) <= 0:
            raise ValueError("Training budgets and sizes must be positive")
        if self.rollout % self.worlds or self.rollout < self.worlds:
            raise ValueError("rollout must be divisible by worlds")
        if not (self.lr > 0 and 0 < self.gamma <= 1 and
                0 <= self.gae_lambda <= 1 and 0 < self.clip < 1 and self.entropy >= 0):
            raise ValueError("Invalid PPO hyperparameters")


def _nomad(env: QoxCoreEnv) -> str:
    return next(name for name in env.possible_agents if name.startswith("nomad_"))


def train_nomad(options: NomadTrainOptions, output: str | Path,
                init_checkpoint: str | Path | None = None, *,
                 progress_callback=None, cancel_requested=None) -> dict:
    """Train a role-shared nomad against responsive scripted cattle and ravager."""
    torch.set_num_threads(1)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)
    random.seed(options.seed)
    model = (load_nomad(init_checkpoint) if init_checkpoint is not None else
             NomadPolicy(options.hidden, nomad_v2=options.nomad_v2))
    if model.nomad_v2 != options.nomad_v2:
        raise ValueError("Initial nomad checkpoint observation version mismatch")
    if model.encoder[0].out_features != options.hidden:
        raise ValueError("Initial nomad checkpoint hidden size does not match requested size")
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
                       scripted_others=True, cow_calls=True,
                       nomad_v2=options.nomad_v2)
            for _ in range(options.worlds)]
    names = [_nomad(env) for env in envs]
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

        obs, masks, actions, logps, values, rewards, dones = [], [], [], [], [], [], []
        for _ in range(options.rollout // options.worlds):
            x = torch.as_tensor(np.stack([s["observation"] for s in states]),
                                dtype=torch.float32)
            m = torch.as_tensor(np.stack([s["action_mask"] for s in states]),
                                dtype=torch.float32)
            with torch.no_grad():
                dist, value = model(x, m)
                choice = dist.sample()
                logp = dist.log_prob(choice)
            step_rewards, step_dones = [], []
            for i, env in enumerate(envs):
                next_obs, earned, terminated, truncated, _ = env.step({
                    names[i]: int(choice[i].item())})
                reward = float(earned[names[i]])
                done = bool(terminated[names[i]] or truncated[names[i]])
                active_returns[i] += reward
                step_rewards.append(reward)
                step_dones.append(float(done))
                if done:
                    completed_returns.append(active_returns[i])
                    active_returns[i] = 0.0
                    episode_counts[i] += 1
                    next_seed = options.seed + 1009 * i + 100_003 * episode_counts[i]
                    states[i] = env.reset(seed=next_seed)[0][names[i]]
                else:
                    states[i] = next_obs[names[i]]
            obs.append(x)
            masks.append(m)
            actions.append(choice)
            logps.append(logp)
            values.append(value)
            rewards.append(torch.as_tensor(step_rewards, dtype=torch.float32))
            dones.append(torch.as_tensor(step_dones, dtype=torch.float32))
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
        target_returns = (advantages + values_t).flatten().detach()
        xs = torch.stack(obs).flatten(0, 1)
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
        history.append({"nomad_steps": steps, "completed_episodes": len(completed_returns),
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
                "model_hidden": options.hidden, "species": "nomad",
                "obs_version": (NOMAD_RELATIONAL_OBSERVATION_VERSION
                                if options.nomad_v2 else NOMAD_OBSERVATION_VERSION),
                "observation_features": features_for(Species.NOMAD,
                                                      nomad_v2=options.nomad_v2),
                "actions": action_names(), "options": asdict(options),
                "completed_steps": steps,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None}, output)
    report = {"checkpoint": str(output), "initial_checkpoint": str(init_checkpoint)
              if init_checkpoint is not None else None, "nomad_steps": steps,
              "completed_episodes": len(completed_returns),
              "recent_returns": completed_returns[-30:], "history": history}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    return report


def load_nomad(path: str | Path) -> NomadPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    version = data.get("obs_version")
    if version not in (NOMAD_OBSERVATION_VERSION, NOMAD_RELATIONAL_OBSERVATION_VERSION):
        raise ValueError("Nomad checkpoint observation version mismatch")
    nomad_v2 = version == NOMAD_RELATIONAL_OBSERVATION_VERSION
    if (data.get("species") != "nomad"
            or tuple(data.get("observation_features", ())) != features_for(
                Species.NOMAD, nomad_v2=nomad_v2)
            or tuple(data.get("actions", ())) != action_names()):
        raise ValueError("Nomad checkpoint schema mismatch")
    model = NomadPolicy(int(data["model_hidden"]), nomad_v2=nomad_v2)
    model.load_state_dict(data["policy_state"])
    model.eval()
    return model


def evaluate_nomad(checkpoint: str | Path | None = None, *, episodes: int = 12,
                   seed: int = 100_000, scenario: str = "nomad_herd",
                   max_steps: int = 96, mode: str = "policy",
                   cow_checkpoint: str | Path | None = None) -> dict:
    """Evaluate a *frozen* nomad policy with paired call/no-call/flip controls.

    ``policy_greedy`` means deterministic actor actions; stochastic ``policy``
    and ``random`` use their own RNG independent of world physics. In the
    counterfactual modes, the selected friendly/scary action is swapped before
    execution without changing the goal or positions.
    """
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if scenario not in ("nomad_signal", "nomad_herd", "nomad_defend", "nomad_mixed"):
        raise ValueError("Unknown evaluation scenario")
    modes = ("policy", "policy_greedy", "policy_nocalls", "policy_greedy_nocalls",
             "policy_greedy_flipped", "random", "scripted", "scripted_nocalls",
             "scripted_flipped")
    if mode not in modes:
        raise ValueError("Unknown evaluation mode")
    if mode.startswith("policy") and checkpoint is None:
        raise ValueError("Policy evaluation requires a checkpoint")
    model = load_nomad(checkpoint) if mode.startswith("policy") else None
    cow_model = None
    if cow_checkpoint is not None:
        from .cow_ppo import load_cow
        cow_model = load_cow(cow_checkpoint)
        if not cow_model.cow_calls:
            raise ValueError("Herd evaluation needs a call-aware 66-feature cow checkpoint")
    chooser = np.random.default_rng(seed)
    results = []
    for episode in range(episodes):
        env = QoxCoreEnv(scenario=scenario, max_steps=max_steps, scripted_others=True,
                         cow_calls=True, allow_nomad_calls=not mode.endswith("nocalls"),
                         nomad_v2=model.nomad_v2 if model is not None else False)
        observations, _ = env.reset(seed=seed + episode)
        name = _nomad(env)
        original_cow_count = sum(a.species == Species.COW for a in env.world.actors.values())
        cumulative_reward = 0.0
        for _ in range(max_steps):
            ob = observations[name]
            if mode == "random":
                idx = int(chooser.choice(np.flatnonzero(ob["action_mask"])))
            elif mode.startswith("scripted"):
                action = scripted_actions(env.world)[env.name_to_id[name]]
                idx = ACTIONS[Species.NOMAD].index(action)
                if not ob["action_mask"][idx]:
                    idx = 0
            else:
                with torch.no_grad():
                    x = torch.as_tensor(ob["observation"]).unsqueeze(0)
                    m = torch.as_tensor(ob["action_mask"]).unsqueeze(0)
                    dist, _ = model(x, m)
                    if "greedy" in mode:
                        idx = int(dist.probs.argmax(dim=1).item())
                    else:
                        idx = int(chooser.choice(len(ACTIONS[Species.NOMAD]),
                                                 p=dist.probs.squeeze(0).numpy()))
            if mode.endswith("flipped"):
                kind = ACTIONS[Species.NOMAD][idx].kind
                if kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL):
                    swapped = Kind.SCARY_CALL if kind == Kind.FRIENDLY_CALL else Kind.FRIENDLY_CALL
                    idx = ACTIONS[Species.NOMAD].index(Action(swapped))
                    if not ob["action_mask"][idx]:
                        idx = 0
            orders = {name: idx}
            if cow_model is not None:
                for cow_name in observations:
                    if not cow_name.startswith("cow_"):
                        continue
                    cow_obs = observations[cow_name]
                    with torch.no_grad():
                        cow_dist, _ = cow_model(
                            torch.as_tensor(cow_obs["observation"]).unsqueeze(0),
                            torch.as_tensor(cow_obs["action_mask"]).unsqueeze(0))
                        orders[cow_name] = int(cow_dist.probs.argmax(dim=1).item())
            observations, rewards, term, trunc, _ = env.step(orders)
            cumulative_reward += rewards[name]
            if term[name] or trunc[name]:
                break
        nomad = env.world.actors.get(env.name_to_id[name])
        cattle = [a for a in env.world.actors.values() if a.species == Species.COW]
        events = env.world.events
        calls = [e.kind for e in events if e.kind in (Kind.FRIENDLY_CALL.value,
                                                      Kind.SCARY_CALL.value)
                 and e.actor_id == env.name_to_id[name]]
        expected = str(env.signal_kind) if scenario == "nomad_signal" else None
        results.append({"seed": seed + episode, "return": round(cumulative_reward, 4),
                        "expected_signal": expected,
                        "correct_calls": calls.count(expected) if expected else None,
                        "wrong_calls": (len(calls) - calls.count(expected)) if expected else None,
                        "cows_in_pasture": len(env.reached_goal_ids),
                        "cows_alive": len(cattle), "original_cows": original_cow_count,
                        "mean_cow_health": (round(sum(c.health for c in cattle) /
                                                  original_cow_count, 3)
                                            if original_cow_count else 0),
                        "calls": len(calls),
                        "nomad_attacks": sum(e.kind == "nomad_attack" for e in events),
                        "nomad_alive": nomad is not None})
    def mean(key):
        return round(float(np.mean([entry[key] for entry in results])), 4)
    report = {"mode": mode, "scenario": scenario, "episodes": episodes,
              "cow_controller": ("learned_greedy" if cow_checkpoint is not None
                                 else "scripted"),
              "seed_start": seed, "mean_return": mean("return"),
              "mean_cows_in_pasture": mean("cows_in_pasture"),
              "mean_cows_alive": mean("cows_alive"),
              "mean_cow_health": mean("mean_cow_health"),
              "mean_calls": mean("calls"), "mean_nomad_attacks": mean("nomad_attacks"),
              "nomad_survival_rate": mean("nomad_alive"), "results": results}
    if scenario == "nomad_signal":
        report["mean_correct_calls"] = mean("correct_calls")
        report["mean_wrong_calls"] = mean("wrong_calls")
        report["friendly_target_rate"] = round(float(np.mean([
            x["cows_in_pasture"] for x in results if x["expected_signal"] == "friendly_call"
        ])), 4) if any(x["expected_signal"] == "friendly_call" for x in results) else None
        report["scary_target_rate"] = round(float(np.mean([
            x["cows_in_pasture"] for x in results if x["expected_signal"] == "scary_call"
        ])), 4) if any(x["expected_signal"] == "scary_call" for x in results) else None
    return report
