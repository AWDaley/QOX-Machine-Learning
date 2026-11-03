"""M5c: alternating frozen-partner PPO for shared cow and nomad policies.

In a cow phase all three cows act from a *shared* trainable policy, while the
nomad's policy is frozen. The following nomad phase trains against the frozen
cow policy. Neither policy sees the other's hidden internal state. Herd goal
progress is used only by the environment's reward function, not cow inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from ..entities import Species
from ..envs.core import QoxCoreEnv
from ..perception import (ACTIONS, COW_CALL_OBSERVATION_VERSION,
                          NOMAD_OBSERVATION_VERSION, NOMAD_RELATIONAL_OBSERVATION_VERSION,
                          features_for)
from .cow_ppo import CowPolicy, load_cow
from .nomad_ppo import NomadPolicy, action_names, evaluate_nomad, load_nomad


@dataclass(frozen=True)
class JointTrainOptions:
    """Budgets count *individual actor decisions*, not world ticks."""

    rounds: int = 1
    scenario: str = "nomad_herd"
    partner_mode: str = "greedy"
    cow_steps: int = 12_288
    nomad_steps: int = 4_096
    seed: int = 407
    worlds: int = 4
    rollout: int = 384
    max_steps: int = 96
    epochs: int = 3
    minibatch: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01

    def __post_init__(self):
        if min(self.rounds, self.cow_steps, self.nomad_steps, self.worlds,
               self.rollout, self.max_steps, self.epochs, self.minibatch) <= 0:
            raise ValueError("All training sizes and budgets must be positive")
        if self.rollout % (self.worlds * 3):
            raise ValueError("rollout must be divisible by three cows per world")
        if self.scenario not in ("nomad_herd", "nomad_defend", "nomad_mixed", "nomad_convoy", "nomad_convoy_mixed"):
            raise ValueError("Joint scenario must be one of the supported migration/defense/convoy tasks")
        if self.partner_mode not in ("greedy", "sample"):
            raise ValueError("Partner mode must be greedy or sample")
        if not (self.lr > 0 and 0 < self.gamma <= 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip < 1 and self.entropy >= 0):
            raise ValueError("Invalid PPO settings")


def _names(env: QoxCoreEnv, species: Species) -> list[str]:
    names = sorted((name for name in env.possible_agents
                    if name.startswith(f"{species.value}_")),
                   key=lambda name: env.name_to_id[name])
    expected = 3 if species == Species.COW else 1
    if len(names) != expected:
        raise ValueError(f"M5c requires exactly {expected} {species.value} agents per world")
    return names


def _sample_actions(model: nn.Module, observations: list[dict], *,
                    greedy: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.as_tensor(np.stack([o["observation"] for o in observations]),
                         dtype=torch.float32)
    masks = torch.as_tensor(np.stack([o["action_mask"] for o in observations]),
                            dtype=torch.float32)
    with torch.no_grad():
        distribution, values = model(xs, masks)
        choices = (distribution.probs.argmax(dim=-1) if greedy
                   else distribution.sample())
        logprob = distribution.log_prob(choices)
    return choices, logprob, values


def _save_model(model: nn.Module, optimizer: torch.optim.Optimizer,
                species: Species, path: Path, options: JointTrainOptions,
                step_count: int, round_index: int) -> None:
    hidden = model.encoder[0].out_features
    if species == Species.COW:
        # Use the exact existing schema so old load_cow / inference stay compatible.
        action_names_for_checkpoint = tuple(
            a.to_dict()["kind"] + (f"_{a.dx}_{a.dy}" if a.kind.value == "move" else "")
            for a in ACTIONS[Species.COW])
        metadata = {
            "obs_version": COW_CALL_OBSERVATION_VERSION,
            "observation_features": features_for(Species.COW, cow_calls=True),
            "actions": action_names_for_checkpoint,
        }
    else:
        metadata = {
            "obs_version": (NOMAD_RELATIONAL_OBSERVATION_VERSION if model.nomad_v2
                            else NOMAD_OBSERVATION_VERSION),
            "observation_features": features_for(Species.NOMAD,
                                                  nomad_v2=model.nomad_v2),
            "actions": action_names(), "species": "nomad",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_hidden": hidden, "completed_steps": step_count,
                "joint_phase": species.value, "round": round_index,
                "options": asdict(options), **metadata}, path)


def train_phase(species: Species, cow_model: CowPolicy, nomad_model: NomadPolicy,
                options: JointTrainOptions, output: str | Path, *,
                round_index: int = 1, seed: int | None = None) -> dict:
    """Train all current-species actors against frozen partner decisions.

    Each partner acts on its own observation. It cannot see the hidden pasture
    goal if it is a cow. There are no imitation labels or action rewrites.
    The selected scenario can alternate migration and predator-defense tasks.
    A world-level reset cuts all training trajectories when *any* actor dies,
    so advantages cannot leak across independent episodes.
    """
    if species not in (Species.COW, Species.NOMAD):
        raise ValueError("Only cow and nomad partner training are supported")
    if not cow_model.cow_calls:
        raise ValueError("Cow partner needs 66-feature call-aware policy")
    torch.set_num_threads(1)
    phase_seed = options.seed if seed is None else seed
    torch.manual_seed(phase_seed)
    np.random.seed(phase_seed)
    random.seed(phase_seed)
    trainable = cow_model if species == Species.COW else nomad_model
    frozen = nomad_model if species == Species.COW else cow_model
    trainable.train()
    frozen.eval()
    optimizer = torch.optim.Adam(trainable.parameters(), lr=options.lr, eps=1e-5)
    other = Species.NOMAD if species == Species.COW else Species.COW
    envs = [QoxCoreEnv(scenario=options.scenario, max_steps=options.max_steps,
                       scripted_others=True, cow_calls=True,
                       nomad_v2=nomad_model.nomad_v2)
            for _ in range(options.worlds)]
    names = [_names(env, species) for env in envs]
    other_names = [_names(env, other) for env in envs]
    actors_per_world = len(names[0])
    width = options.worlds * actors_per_world
    episodes = [0 for _ in envs]
    state = [env.reset(seed=phase_seed + 1019 * i)[0]
             for i, env in enumerate(envs)]
    returns = [0.0 for _ in envs]
    completed_returns: list[float] = []
    samples = 0
    history: list[dict] = []
    while samples < (options.cow_steps if species == Species.COW else options.nomad_steps):
        collected: dict[str, list] = {k: [] for k in
                                      ("obs", "mask", "choice", "logp", "value",
                                       "reward", "done")}
        for _ in range(options.rollout // width):
            training_obs = [state[wi][name] for wi in range(len(envs))
                            for name in names[wi]]
            partner_obs = [state[wi][name] for wi in range(len(envs))
                           for name in other_names[wi]]
            x = torch.as_tensor(np.stack([o["observation"] for o in training_obs]),
                                dtype=torch.float32)
            mask = torch.as_tensor(np.stack([o["action_mask"] for o in training_obs]),
                                   dtype=torch.float32)
            choice, logp, value = _sample_actions(trainable, training_obs)
            partner_choice, _, _ = _sample_actions(
                frozen, partner_obs, greedy=(options.partner_mode == "greedy"))
            rewards_flat: list[float] = []
            dones_flat: list[float] = []
            train_offset, partner_offset = 0, 0
            for wi, env in enumerate(envs):
                orders = {name: int(partner_choice[partner_offset + j].item())
                          for j, name in enumerate(other_names[wi])}
                orders.update({name: int(choice[train_offset + j].item())
                               for j, name in enumerate(names[wi])})
                partner_offset += len(other_names[wi])
                train_offset += len(names[wi])
                observation, rewards, terminated, truncated, _ = env.step(orders)
                rewards_flat.extend(float(rewards[name]) for name in names[wi])
                done = any(terminated[name] or truncated[name]
                           for name in names[wi] + other_names[wi])
                # Even surviving actors' trajectories end when the entire world
                # resets after another animal dies. Mark them done for GAE.
                dones_flat.extend([float(done)] * len(names[wi]))
                returns[wi] += sum(float(rewards[name]) for name in names[wi])
                if done:
                    completed_returns.append(returns[wi])
                    returns[wi] = 0.0
                    episodes[wi] += 1
                    state[wi] = env.reset(
                        seed=phase_seed + 1019 * wi + 100_003 * episodes[wi])[0]
                else:
                    state[wi] = observation
            collected["obs"].append(x)
            collected["mask"].append(mask)
            collected["choice"].append(choice)
            collected["logp"].append(logp)
            collected["value"].append(value)
            collected["reward"].append(torch.tensor(rewards_flat, dtype=torch.float32))
            collected["done"].append(torch.tensor(dones_flat, dtype=torch.float32))
        with torch.no_grad():
            bootstrap_obs = [state[wi][name] for wi in range(len(envs))
                             for name in names[wi]]
            xs = torch.as_tensor(np.stack([o["observation"] for o in bootstrap_obs]),
                                 dtype=torch.float32)
            ms = torch.as_tensor(np.stack([o["action_mask"] for o in bootstrap_obs]),
                                 dtype=torch.float32)
            _, bootstrap = trainable(xs, ms)
        reward_tensor = torch.stack(collected["reward"])
        done_tensor = torch.stack(collected["done"])
        value_tensor = torch.stack(collected["value"])
        advantages = torch.zeros_like(reward_tensor)
        gae = torch.zeros(width)
        for t in reversed(range(reward_tensor.shape[0])):
            following = bootstrap if t == reward_tensor.shape[0] - 1 else value_tensor[t + 1]
            nonterminal = 1 - done_tensor[t]
            delta = reward_tensor[t] + options.gamma * following * nonterminal - value_tensor[t]
            gae = delta + options.gamma * options.gae_lambda * nonterminal * gae
            advantages[t] = gae
        targets = (advantages + value_tensor).flatten().detach()
        obs_batch = torch.stack(collected["obs"]).flatten(0, 1)
        mask_batch = torch.stack(collected["mask"]).flatten(0, 1)
        actions_batch = torch.stack(collected["choice"]).flatten().detach()
        old_logp = torch.stack(collected["logp"]).flatten().detach()
        adv = advantages.flatten().detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        loss_value = 0.0
        for _ in range(options.epochs):
            for batch in torch.randperm(len(obs_batch)).split(options.minibatch):
                dist, predicted = trainable(obs_batch[batch], mask_batch[batch])
                ratio = (dist.log_prob(actions_batch[batch]) - old_logp[batch]).exp()
                loss = (-torch.minimum(ratio * adv[batch],
                                       ratio.clamp(1-options.clip,
                                                   1+options.clip) * adv[batch]).mean()
                        + .5 * (targets[batch] - predicted).square().mean()
                        - options.entropy * dist.entropy().mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), 0.5)
                optimizer.step()
                loss_value = float(loss.detach())
        samples += len(obs_batch)
        history.append({"species_steps": samples,
                        "complete_episodes": sum(episodes),
                        "loss": round(loss_value, 6),
                        "recent_mean_episode_return": round(float(
                            np.mean(completed_returns[-20:])), 3)
                            if completed_returns else None})
    trainable.eval()
    output = Path(output)
    _save_model(trainable, optimizer, species, output, options, samples, round_index)
    return {"species": species.value, "checkpoint": str(output),
            "completed_steps": samples, "episodes": sum(episodes), "history": history,
            "final_loss": round(loss_value, 6), "scenario": options.scenario,
            "partner_mode": options.partner_mode}


def train_joint(options: JointTrainOptions, cow_checkpoint: str | Path,
                nomad_checkpoint: str | Path, output_dir: str | Path,
                *, phase: str = "alternate") -> dict:
    """Train cows, nomads, or alternate both, preserving each intermediate pair.

    A cow-only phase is useful when a good nomad policy should remain fixed.
    This routine **does not** imply convergence; evaluate all intermediate
    checkpoints on held-out worlds before choosing a deployed policy pair.
    """
    if phase not in ("alternate", "cow", "nomad"):
        raise ValueError("phase must be alternate, cow or nomad")
    cow_model = load_cow(cow_checkpoint)
    nomad_model = load_nomad(nomad_checkpoint)
    if not cow_model.cow_calls:
        raise ValueError("Joint herding requires call-aware 66-feature cow checkpoint")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    final_cow = str(cow_checkpoint)
    final_nomad = str(nomad_checkpoint)
    for round_index in range(1, options.rounds + 1):
        if phase in ("alternate", "cow"):
            cow_result = train_phase(Species.COW, cow_model, nomad_model, options,
                                     out / f"cow_round{round_index}.pt",
                                     round_index=round_index,
                                     seed=(options.seed if phase == "cow" and round_index == 1
                                           else options.seed + 15_001 * round_index))
            reports.append(cow_result)
            final_cow = cow_result["checkpoint"]
        if phase in ("alternate", "nomad"):
            nomad_result = train_phase(Species.NOMAD, cow_model, nomad_model, options,
                                       out / f"nomad_round{round_index}.pt",
                                       round_index=round_index,
                                       seed=(options.seed if phase == "nomad" and round_index == 1
                                             else options.seed + 25_003 * round_index))
            reports.append(nomad_result)
            final_nomad = nomad_result["checkpoint"]
    result = {"options": asdict(options), "phase": phase,
              "initial_cow_checkpoint": str(cow_checkpoint),
              "initial_nomad_checkpoint": str(nomad_checkpoint),
              "rounds": reports, "final_cow_checkpoint": final_cow,
              "final_nomad_checkpoint": final_nomad}
    (out / "joint_training.json").write_text(json.dumps(result, indent=2),
                                               encoding="utf-8")
    return result


def benchmark_joint(cow_checkpoint: str | Path, nomad_checkpoint: str | Path, *,
                    episodes: int = 24, seed: int = 340_000,
                    max_steps: int = 96, scenario: str = "nomad_herd") -> dict:
    """Paired held-out counterfactuals with learned cows and nomad.

    Data include raw per-seed outcomes. The deterministic counterfactuals
    change only whether/which call gets emitted, not the map or reward rules.
    """
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    if scenario not in ("nomad_herd", "nomad_defend", "nomad_mixed", "nomad_convoy", "nomad_convoy_mixed"):
        raise ValueError("Unsupported joint herding evaluation scenario")
    modes = {"learned": "policy_greedy",
             "no_calls": "policy_greedy_nocalls",
             "reversed_calls": "policy_greedy_flipped",
             "scripted_nomad": "scripted",
             "random_nomad": "random"}
    runs = {key: evaluate_nomad(nomad_checkpoint, episodes=episodes, seed=seed,
                                 scenario=scenario, max_steps=max_steps,
                                 mode=value, cow_checkpoint=cow_checkpoint)
            for key, value in modes.items()}
    original = evaluate_nomad(nomad_checkpoint, episodes=episodes, seed=seed,
                              scenario=scenario, max_steps=max_steps,
                              mode="policy_greedy")
    summary = {key: {field: report[field] for field in
                     ("mean_cows_in_pasture", "mean_cows_alive", "mean_cow_health",
                      "nomad_survival_rate", "mean_nomad_attacks", "mean_calls")}
               for key, report in runs.items()}
    summary["scripted_cows"] = {
        "mean_cows_in_pasture": original["mean_cows_in_pasture"],
        "mean_cows_alive": original["mean_cows_alive"],
        "mean_cow_health": original["mean_cow_health"],
        "nomad_survival_rate": original["nomad_survival_rate"],
        "mean_nomad_attacks": original["mean_nomad_attacks"],
        "mean_calls": original["mean_calls"],
    }
    per_seed = []
    for i in range(episodes):
        per_seed.append({"seed": seed+i,
                         **{k: v["results"][i]["cows_in_pasture"] for k, v in runs.items()},
                         "cow_health": runs["learned"]["results"][i]["mean_cow_health"],
                         "nomad_alive": runs["learned"]["results"][i]["nomad_alive"],
                         "scripted_cows": original["results"][i]["cows_in_pasture"]})
    return {"episodes": episodes, "seed_start": seed, "scenario": scenario,
            "cow_checkpoint": str(cow_checkpoint),
            "nomad_checkpoint": str(nomad_checkpoint),
            "summary": summary, "paired_results": per_seed,
            "mean_gain_over_no_calls": round(float(np.mean([
                row["learned"] - row["no_calls"] for row in per_seed])), 4)}


def _paired_interval(differences: list[float], *, seed: int) -> tuple[float, float]:
    """Bootstrap paired differences; uncertainty, not proof of generalization."""
    if not differences:
        raise ValueError("Expected at least one paired outcome")
    values = np.asarray(differences, dtype=np.float64)
    random_source = np.random.default_rng(seed)
    sample_indices = random_source.integers(0, len(values), size=(500, len(values)))
    means = values[sample_indices].mean(axis=1)
    return tuple(round(float(x), 4) for x in np.percentile(means, [2.5, 97.5]))


def compare_joint(baseline_cow: str | Path, baseline_nomad: str | Path,
                  candidate_cow: str | Path, candidate_nomad: str | Path, *,
                  episodes: int = 48, validation_seed: int = 660_000,
                  max_steps: int = 96) -> dict:
    """Counterfactual validation on common migration and defense seeds.

    Only the modes needed for comparison are run; there is no scripted control
    in this fast candidate gate. The full benchmark_joint still reports those
    baselines separately. Neither checkpoints nor the active session mutate.
    """
    if episodes < 1 or max_steps < 1 or validation_seed < 0:
        raise ValueError("episodes/max_steps must be positive and seed nonnegative")
    # CPU inference with many tiny networks is much faster without oversubscription.
    torch.set_num_threads(1)
    pairs = ((baseline_cow, baseline_nomad), (candidate_cow, candidate_nomad))
    runs: dict[str, dict] = {}
    for label, pair in zip(("baseline", "candidate"), pairs, strict=True):
        migration = evaluate_nomad(pair[1], cow_checkpoint=pair[0],
                                   episodes=episodes, seed=validation_seed,
                                   max_steps=max_steps, scenario="nomad_herd",
                                   mode="policy_greedy")
        no_calls = evaluate_nomad(pair[1], cow_checkpoint=pair[0],
                                  episodes=episodes, seed=validation_seed,
                                  max_steps=max_steps, scenario="nomad_herd",
                                  mode="policy_greedy_nocalls")
        defense = evaluate_nomad(pair[1], cow_checkpoint=pair[0],
                                 episodes=episodes, seed=validation_seed + 200_000,
                                 max_steps=max_steps, scenario="nomad_defend",
                                 mode="policy_greedy")
        runs[label] = {"migration": migration, "no_calls": no_calls,
                       "defense": defense}
    old, new = runs["baseline"], runs["candidate"]
    arrivals = [c["cows_in_pasture"] - b["cows_in_pasture"]
                for b, c in zip(old["migration"]["results"],
                                new["migration"]["results"], strict=True)]
    health = [c["mean_cow_health"] - b["mean_cow_health"]
              for b, c in zip(old["defense"]["results"],
                              new["defense"]["results"], strict=True)]
    gain = round(float(np.mean(arrivals)), 4)
    health_gain = round(float(np.mean(health)), 4)
    arrival_interval = _paired_interval(arrivals, seed=validation_seed + 7)
    health_interval = _paired_interval(health, seed=validation_seed + 11)
    call_lifts = {label: round(report["migration"]["mean_cows_in_pasture"] -
                               report["no_calls"]["mean_cows_in_pasture"], 4)
                  for label, report in runs.items()}
    checks = {
        "enough_validation_seeds": episodes >= 24,
        "arrival_improvement": gain > 0 and arrival_interval[0] > 0,
        "correct_calls_help_candidate": call_lifts["candidate"] > 0,
        "defense_health_not_worse_than_five_points": health_gain >= -5.0,
        "defense_survival_not_worse": (
            new["defense"]["mean_cows_alive"] >=
            old["defense"]["mean_cows_alive"]),
    }
    metrics = ("mean_cows_in_pasture", "mean_cows_alive", "mean_cow_health",
               "nomad_survival_rate", "mean_nomad_attacks", "mean_calls")
    def selection(name: str, mode: str) -> dict:
        return {key: runs[name][mode][key] for key in metrics}

    return {
        "validation_seed": validation_seed, "episodes_per_scenario": episodes,
        "max_steps": max_steps,
        "pairs": {label: {"cow_checkpoint": str(pair[0]),
                          "nomad_checkpoint": str(pair[1])}
                  for label, pair in zip(("baseline", "candidate"), pairs, strict=True)},
        "migration": {
            "baseline": selection("baseline", "migration"),
            "candidate": selection("candidate", "migration"),
            "baseline_no_calls": selection("baseline", "no_calls"),
            "candidate_no_calls": selection("candidate", "no_calls"),
            "arrival_gain_per_world": gain,
            "arrival_gain_bootstrap_95pct": arrival_interval,
            "baseline_correct_call_lift": call_lifts["baseline"],
            "candidate_correct_call_lift": call_lifts["candidate"],
        },
        "defense": {
            "baseline": selection("baseline", "defense"),
            "candidate": selection("candidate", "defense"),
            "cow_health_gain": health_gain,
            "cow_health_gain_bootstrap_95pct": health_interval,
        },
        "diagnostic_checks": checks,
        "candidate_eligible_for_manual_review": all(checks.values()),
        "notes": ("No checkpoint promoted. Validation is not a final blind test. "
                  "Predator in defense worlds is scripted; compare on untouched "
                  "test seeds before interpreting policy improvements."),
        "paired_migration": [{"seed": validation_seed + i,
                              "baseline": b["cows_in_pasture"],
                              "candidate": c["cows_in_pasture"],
                              "difference": arrivals[i]}
                             for i, (b, c) in enumerate(zip(
                                 old["migration"]["results"],
                                 new["migration"]["results"], strict=True))],
        "paired_defense": [{"seed": validation_seed + 200_000 + i,
                            "baseline_cow_health": b["mean_cow_health"],
                            "candidate_cow_health": c["mean_cow_health"]}
                           for i, (b, c) in enumerate(zip(
                               old["defense"]["results"],
                               new["defense"]["results"], strict=True))],
    }
