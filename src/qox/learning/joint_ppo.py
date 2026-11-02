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
    This phase deliberately targets only predator-free nomad_herd curriculum.
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
    envs = [QoxCoreEnv(scenario="nomad_herd", max_steps=options.max_steps,
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
            partner_choice, _, _ = _sample_actions(frozen, partner_obs, greedy=True)
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
                dones_flat.extend(float(terminated[name] or truncated[name])
                                  for name in names[wi])
                returns[wi] += sum(float(rewards[name]) for name in names[wi])
                done = any(terminated[name] or truncated[name]
                           for name in names[wi] + other_names[wi])
                if done:
                    # In M5c the nomad_herd environment has no predators.
                    # Reset *whole* world on any terminal to keep partners aligned.
                    if not all(truncated.get(name, False)
                               for name in names[wi] + other_names[wi]):
                        raise RuntimeError("Unexpected early death in M5c no-predator scenario")
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
            "final_loss": round(loss_value, 6)}


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
                    max_steps: int = 96) -> dict:
    """Paired held-out counterfactuals with learned cows and nomad.

    Data include raw per-seed outcomes. The deterministic counterfactuals
    change only whether/which call gets emitted, not the map or reward rules.
    """
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    modes = {"learned": "policy_greedy",
             "no_calls": "policy_greedy_nocalls",
             "reversed_calls": "policy_greedy_flipped",
             "scripted_nomad": "scripted",
             "random_nomad": "random"}
    runs = {key: evaluate_nomad(nomad_checkpoint, episodes=episodes, seed=seed,
                                 scenario="nomad_herd", max_steps=max_steps,
                                 mode=value, cow_checkpoint=cow_checkpoint)
            for key, value in modes.items()}
    original = evaluate_nomad(nomad_checkpoint, episodes=episodes, seed=seed,
                              scenario="nomad_herd", max_steps=max_steps,
                              mode="policy_greedy")
    summary = {key: {field: report[field] for field in
                     ("mean_cows_in_pasture", "mean_cows_alive", "mean_calls")}
               for key, report in runs.items()}
    summary["scripted_cows"] = {
        "mean_cows_in_pasture": original["mean_cows_in_pasture"],
        "mean_cows_alive": original["mean_cows_alive"],
        "mean_calls": original["mean_calls"],
    }
    per_seed = []
    for i in range(episodes):
        per_seed.append({"seed": seed+i,
                         **{k: v["results"][i]["cows_in_pasture"] for k, v in runs.items()},
                         "scripted_cows": original["results"][i]["cows_in_pasture"]})
    return {"episodes": episodes, "seed_start": seed,
            "cow_checkpoint": str(cow_checkpoint),
            "nomad_checkpoint": str(nomad_checkpoint),
            "summary": summary, "paired_results": per_seed,
            "mean_gain_over_no_calls": round(float(np.mean([
                row["learned"] - row["no_calls"] for row in per_seed])), 4)}
