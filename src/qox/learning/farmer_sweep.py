"""M16f: paired farmer hyperparameter sweep with a single held-out audit.

All weights and stage budgets share development seeds. Candidate selection is
finished *before* the separate audit range is accessed, and nothing promotes
or replaces an active neural-network checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from .farmer_eval import SCENARIOS, compare_farmers
from .farmer_ppo import evaluate_farmer, load_farmer
from .farmer_staged import (STAGES, StagedFarmerOptions, _gate,
                            _sha, _write_report, train_farmer_staged)


@dataclass(frozen=True)
class FarmerSweepOptions:
    survival_weights: tuple[float, ...] = (1.0, 3.0, 6.0)
    stage_budgets: tuple[int, ...] = (1024, 4096)
    rehearsal_kls: tuple[float, ...] = (0.0,)
    interleave_cycles: tuple[int, ...] = (1,)
    manure_credits: tuple[float, ...] = (0.0,)
    training_seeds: tuple[int, ...] = (17, 37, 59)
    worlds: int = 4
    rollout: int = 256
    epochs: int = 4
    max_steps_override: int | None = None
    development_episodes: int = 48
    audit_episodes: int = 48
    development_seed: int = 1_600_000
    audit_seed: int = 2_600_000
    rehearsal_seed: int = 1_200_000
    rehearsal_episodes: int = 4
    rehearsal_max_steps: int = 96

    def __post_init__(self) -> None:
        if (not self.survival_weights or not self.stage_budgets or not self.rehearsal_kls or
                not self.interleave_cycles or not self.manure_credits or
                len(set(self.survival_weights)) != len(self.survival_weights) or
                len(set(self.stage_budgets)) != len(self.stage_budgets) or
                len(set(self.rehearsal_kls)) != len(self.rehearsal_kls) or
                len(set(self.interleave_cycles)) != len(self.interleave_cycles) or
                len(set(self.manure_credits)) != len(self.manure_credits)):
            raise ValueError("Sweep settings must be nonempty and distinct")
        configs = (len(self.survival_weights) * len(self.stage_budgets) *
                   len(self.rehearsal_kls) * len(self.interleave_cycles) *
                   len(self.manure_credits))
        if configs > 12:
            raise ValueError("Maximum twelve sweep configurations per experiment")
        for weight, budget, kl, cycles, manure_credit in product(
                self.survival_weights, self.stage_budgets, self.rehearsal_kls,
                self.interleave_cycles, self.manure_credits):
            self.stage_options(weight, budget, kl, cycles, manure_credit)

    def stage_options(self, weight: float, budget: int,
                      rehearsal_kl: float = 0.0,
                      interleave_cycles: int = 1,
                      manure_credit: float = 0.0) -> StagedFarmerOptions:
        return StagedFarmerOptions(
            training_seeds=self.training_seeds, steps_per_stage=budget,
            worlds=self.worlds, rollout=self.rollout, epochs=self.epochs,
            survival_weight=weight, max_steps_override=self.max_steps_override,
            development_episodes=self.development_episodes,
            audit_episodes=self.audit_episodes,
            development_seed=self.development_seed, audit_seed=self.audit_seed,
            rehearsal_kl=rehearsal_kl, rehearsal_seed=self.rehearsal_seed,
            rehearsal_episodes=self.rehearsal_episodes,
            rehearsal_max_steps=self.rehearsal_max_steps,
            interleave_cycles=interleave_cycles, manure_credit=manure_credit)

    def estimate(self) -> dict:
        # The PPO trainer finishes the full last rollout of each stage.
        decisions = (len(self.training_seeds) * len(STAGES) *
                     len(self.survival_weights) * len(self.rehearsal_kls) *
                     len(self.manure_credits) *
                     sum(((((budget // cycles + self.rollout - 1) // self.rollout) *
                            self.rollout) * cycles)
                         for budget in self.stage_budgets
                         for cycles in self.interleave_cycles))
        configs = (len(self.survival_weights) * len(self.stage_budgets) *
                   len(self.rehearsal_kls) * len(self.interleave_cycles) *
                   len(self.manure_credits))
        return {
            "configurations": configs,
            "training_runs": (len(self.training_seeds) * len(STAGES) *
                              len(self.survival_weights) * len(self.stage_budgets) *
                              len(self.rehearsal_kls) * len(self.manure_credits) *
                              sum(self.interleave_cycles)),
            "approximate_ppo_decisions": decisions,
            "development_world_episodes": (configs * len(self.training_seeds) *
                                            len(SCENARIOS) * 3 * self.development_episodes),
            "rehearsal_world_episodes": (len(self.survival_weights) *
                                         len(self.stage_budgets) *
                                         len(self.interleave_cycles) *
                                         len(self.manure_credits) *
                                         sum(kl > 0 for kl in self.rehearsal_kls) *
                                         3 * self.rehearsal_episodes),
            "audit_world_episodes_if_selected": (len(SCENARIOS) *
                                                 5 * self.audit_episodes),
            "audit_is_single_use": True,
        }


def _slug(weight: float, budget: int, rehearsal_kl: float = 0.0,
          interleave_cycles: int = 1, manure_credit: float = 0.0) -> str:
    base = f"safety_{weight:.17g}".replace(".", "p") + f"_steps_{budget}"
    if rehearsal_kl:
        base += f"_kl_{rehearsal_kl:.17g}".replace(".", "p")
    if interleave_cycles > 1:
        base += f"_cycles_{interleave_cycles}"
    if manure_credit:
        base += f"_manure_{manure_credit:.17g}".replace(".", "p")
    return base


def run_farmer_sweep(baseline: str | Path, output_dir: str | Path,
                     options: FarmerSweepOptions) -> dict:
    """Choose one development winner globally; audit that winner at most once.

    The baseline, active model sets, and visible world remain read-only. A
    failed experiment leaves its partial outputs for inspection.
    """
    baseline, output_dir = Path(baseline), Path(output_dir)
    if not baseline.is_file():
        raise ValueError(f"Missing farmer baseline checkpoint: {baseline}")
    load_farmer(baseline)
    if output_dir.exists():
        raise FileExistsError(f"Sweep output already exists: {output_dir}")
    original_hash = _sha(baseline)
    output_dir.mkdir(parents=True)

    configs: list[dict] = []
    eligible: list[tuple[float, int, int, Path]] = []
    for weight, budget, rehearsal_kl, cycles, manure_credit in product(
            options.survival_weights, options.stage_budgets, options.rehearsal_kls,
            options.interleave_cycles, options.manure_credits):
        location = output_dir / _slug(weight, budget, rehearsal_kl, cycles, manure_credit)
        result = train_farmer_staged(
            baseline, location,
            options.stage_options(weight, budget, rehearsal_kl, cycles, manure_credit),
            audit_enabled=False)
        config_index = len(configs)
        entries = []
        for index, replica in enumerate(result["replicas"]):
            entries.append({"training_seed": replica["training_seed"],
                            "candidate": replica["candidate"],
                            "food_gain": replica["food_gain"],
                            "development_passed": replica["development_passed"],
                            "development_checks": replica["development_checks"],
                            "report": replica["development_report"]})
            if replica["development_passed"] and len(options.training_seeds) >= 3:
                eligible.append((replica["food_gain"], config_index, index,
                                 Path(replica["candidate"])))
        configs.append({"survival_weight": weight, "stage_budget": budget,
                        "rehearsal_kl": rehearsal_kl,
                        "interleave_cycles": cycles, "manure_credit": manure_credit,
                        "rehearsal": result.get("rehearsal"),
                        "location": str(location), "replicas": entries})

    chosen = None
    audit = None
    if eligible:
        # Deterministic tie-breaking; do not choose or retry based on audit.
        score, config_index, replica_index, candidate = max(
            eligible, key=lambda item: (item[0], -item[1], -item[2]))
        config = configs[config_index]
        chosen = {"survival_weight": config["survival_weight"],
                  "stage_budget": config["stage_budget"],
                  "rehearsal_kl": config["rehearsal_kl"],
                  "interleave_cycles": config["interleave_cycles"],
                  "manure_credit": config["manure_credit"],
                  "training_seed": config["replicas"][replica_index]["training_seed"],
                  "candidate": str(candidate), "development_food_gain": score}

        comparison = compare_farmers(
            baseline, candidate, episodes=options.audit_episodes,
            seed=options.audit_seed, max_steps=options.max_steps_override or 192)
        _write_report(output_dir / "audit.json", comparison)
        checks = _gate(comparison)
        audit = {"report": str(output_dir / "audit.json"),
                 "checks": checks, "passed": all(checks.values())}

        # These are descriptive controls only: they cannot change the chosen
        # policy, the audit pass/fail decision, or a promotion threshold.
        references = {
            scenario: {
                mode: evaluate_farmer(
                    episodes=options.audit_episodes,
                    seed=options.audit_seed + offset * 100_000,
                    scenario=scenario, max_steps=options.max_steps_override or 192,
                    mode=mode)
                for mode in ("scripted", "random")}
            for offset, scenario in enumerate(SCENARIOS)
        }
        _write_report(output_dir / "audit_references.json", references)

    if _sha(baseline) != original_hash:
        raise RuntimeError("Farmer baseline checkpoint changed during the sweep")
    approved = bool(audit and audit["passed"])
    report = {
        "version": 4, "method": "farmer_safety_budget_rehearsal_cycle_manure_sweep",
        "baseline": str(baseline), "baseline_sha256": original_hash,
        "options": asdict(options), "cost_estimate": options.estimate(),
        "configurations": configs, "development_qualifiers": len(eligible),
        "selected_before_audit": chosen, "audit": audit,
        "eligible_for_manual_review": approved,
        "recommended_checkpoint": chosen["candidate"] if approved else None,
        "notes": ("All schedules use equal total per-task PPO decision budgets; "
                  "interleave cycles change the ordering and segment lengths. "
                  "Manure shaping credits verified collection (0.2x) and useful "
                  "fertilizer application (1.0x) during training only; evaluation "
                  "rewards and world physics are never changed. "
                  "Development comparisons are shared across all settings, including "
                  "the no-rehearsal control. Teacher rehearsal uses disjoint training "
                  "seeds and never draws on evaluation cases. "
                  "Only one development winner may use audit seeds. "
                  "Scripted and random reference policies are descriptive "
                  "and cannot influence selection. At least 3 training seeds "
                  "and 48 development/audit episodes per scenario are needed "
                  "for manual review. No weights are automatically installed. "
                  "Finite held-out samples do not prove generalization."),
    }
    _write_report(output_dir / "summary.json", report)
    return report
