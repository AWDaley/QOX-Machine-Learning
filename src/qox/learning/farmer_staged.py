"""M16e: multi-seed staged farmer PPO with non-promoting, survival-first gates.

Development results select at most one candidate. A disjoint audit range is
consulted only for that candidate, and no installed policy is ever replaced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from .farmer_eval import compare_farmers
from .farmer_ppo import FarmerTrainOptions, load_farmer, train_farmer


STAGES = (("agriculture", 96), ("manure", 96), ("refuge", 96),
          ("farmstead_mixed", 192))


@dataclass(frozen=True)
class StagedFarmerOptions:
    training_seeds: tuple[int, ...] = (17, 37, 59)
    steps_per_stage: int = 4096
    worlds: int = 4
    rollout: int = 256
    epochs: int = 4
    survival_weight: float = 3.0
    max_steps_override: int | None = None
    development_episodes: int = 48
    audit_episodes: int = 48
    development_seed: int = 1_600_000
    audit_seed: int = 2_600_000

    def __post_init__(self) -> None:
        seeds = self.training_seeds
        if not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds):
            raise ValueError("Provide 1–8 distinct farmer training seeds")
        if any(seed < 0 or seed > 999_999 for seed in seeds):
            raise ValueError("Training seed IDs must be between 0 and 999999")
        if min(self.steps_per_stage, self.worlds, self.rollout, self.epochs,
               self.development_episodes, self.audit_episodes) < 1:
            raise ValueError("Training and evaluation budgets must be positive")
        if self.max_steps_override is not None and self.max_steps_override < 1:
            raise ValueError("Episode cap must be positive")
        if self.rollout < self.worlds or self.rollout % self.worlds:
            raise ValueError("Rollout must be divisible by worlds")
        if not 1.0 <= self.survival_weight <= 20.0:
            raise ValueError("Survival weight must be between 1 and 20")
        # Each comparison evaluates two blocks: safe seeds, then threat seeds
        # offset by 100,000. Keep both blocks separate and below training seeds.
        dev_end = self.development_seed + 100_000 + self.development_episodes
        audit_end = self.audit_seed + 100_000 + self.audit_episodes
        if (self.development_seed < 0 or self.audit_seed < 0 or
                dev_end > self.audit_seed or audit_end >= 1_000_000_000 or
                self.development_episodes >= 100_000 or self.audit_episodes >= 100_000):
            raise ValueError("Development and audit seed ranges must be nonoverlapping")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_report(path: Path, report: dict) -> None:
    """Write complete experiment reports without exposing half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def _gate(report: dict) -> dict[str, bool]:
    """Add explicit farm damage/health checks to the M16d comparison gate."""
    safe = report["cases"]["farmstead"]["metrics"]
    threat = report["cases"]["farmstead_threat"]["metrics"]
    return {
        **report["promotion_checks"],
        "safe_health_not_worse": (safe["candidate"]["mean_final_health"] >=
                                   safe["baseline"]["mean_final_health"] - 1.0),
        "crop_damage_not_worse": (safe["candidate"]["mean_crop_eaten_by_cows"] <=
                                   safe["baseline"]["mean_crop_eaten_by_cows"] + 0.5),
        "threat_crop_damage_not_worse": (
            threat["candidate"]["mean_crop_eaten_by_cows"] <=
            threat["baseline"]["mean_crop_eaten_by_cows"] + 0.5),
        "threat_food_retained": (threat["candidate"]["mean_deposited"] >=
                                 threat["baseline"]["mean_deposited"] - 0.25),
    }


def _food_gain(report: dict) -> float:
    """Selection score applies *only after* all development gates pass."""
    gain = 0.0
    for scenario in ("farmstead", "farmstead_threat"):
        metrics = report["cases"][scenario]["metrics"]
        gain += (metrics["candidate"]["mean_deposited"] -
                 metrics["baseline"]["mean_deposited"])
    return round(gain, 4)


def _stage_seed(replica: int, stage: int, seed_id: int) -> int:
    # Train at 2B+, deliberately disjoint from the <1B evaluation namespaces.
    # Values passed to legacy np.random.seed remain under the 32-bit ceiling.
    return 2_000_000_000 + replica * 10_000_000 + stage * 100_000_000 + seed_id


def train_farmer_staged(baseline: str | Path, output_dir: str | Path,
                        options: StagedFarmerOptions) -> dict:
    """Train independent continuations and audit at most one selected candidate.

    Never mutates ``baseline``, any active QOX model set, or a visible world.
    Refuses an existing output directory so prior experiment data is preserved.
    """
    baseline, output_dir = Path(baseline), Path(output_dir)
    if not baseline.is_file():
        raise ValueError(f"Missing farmer baseline checkpoint: {baseline}")
    starting_model = load_farmer(baseline)
    hidden = int(starting_model.encoder[0].out_features)
    if output_dir.exists():
        raise FileExistsError(f"Experiment output already exists: {output_dir}")
    original_hash = _sha(baseline)
    output_dir.mkdir(parents=True)

    replicas: list[dict] = []
    candidates: list[tuple[float, int, Path]] = []
    for index, seed_id in enumerate(options.training_seeds):
        previous = baseline
        stage_records: list[dict] = []
        replica_dir = output_dir / f"seed_{seed_id}"
        for stage_index, (scenario, recommended_steps) in enumerate(STAGES):
            stage_seed = _stage_seed(index, stage_index, seed_id)
            checkpoint = replica_dir / f"stage_{stage_index + 1}_{scenario}.pt"
            stage_options = FarmerTrainOptions(
                steps=options.steps_per_stage, seed=stage_seed,
                worlds=options.worlds, rollout=options.rollout, epochs=options.epochs,
                max_steps=options.max_steps_override or recommended_steps,
                scenario=scenario, hidden=hidden,
                survival_weight=options.survival_weight if stage_index >= 2 else 1.0)
            result = train_farmer(stage_options, checkpoint, init_checkpoint=previous)
            stage_records.append({
                "scenario": scenario, "seed": stage_seed, "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "decisions": result["farmer_steps"],
                "survival_weight": stage_options.survival_weight,
                "completed_episodes": result["completed_episodes"],
            })
            previous = checkpoint

        development = compare_farmers(
            baseline, previous, episodes=options.development_episodes,
            seed=options.development_seed,
            max_steps=options.max_steps_override or 192)
        _write_report(replica_dir / "development.json", development)
        checks = _gate(development)
        eligible = all(checks.values())
        score = _food_gain(development)
        replicas.append({"training_seed": seed_id, "stages": stage_records,
                         "candidate": str(previous), "development_report": str(
                             replica_dir / "development.json"),
                         "development_checks": checks, "development_passed": eligible,
                         "food_gain": score})
        if eligible:
            candidates.append((score, index, previous))

    audit = None
    if candidates:
        # Ties resolve by input seed order. Never search over audit results.
        _, selected_index, selected_path = max(candidates, key=lambda row: (row[0], -row[1]))
        comparison = compare_farmers(
            baseline, selected_path, episodes=options.audit_episodes,
            seed=options.audit_seed, max_steps=options.max_steps_override or 192)
        _write_report(output_dir / "audit.json", comparison)
        checks = _gate(comparison)
        audit = {"training_seed": options.training_seeds[selected_index],
                 "candidate": str(selected_path), "report": str(output_dir / "audit.json"),
                 "checks": checks, "passed": all(checks.values())}

    if _sha(baseline) != original_hash:
        raise RuntimeError("Baseline checkpoint changed during the experiment")
    eligible = bool(audit and audit["passed"] and len(options.training_seeds) >= 3)
    report = {
        "version": 1, "method": "staged_farmer_multi_seed",
        "baseline": str(baseline), "baseline_sha256": original_hash,
        "options": asdict(options), "stages": [name for name, _ in STAGES],
        "replicas": replicas, "audit": audit,
        "enough_training_seeds": len(options.training_seeds) >= 3,
        "eligible_for_manual_review": eligible,
        "recommended_checkpoint": audit["candidate"] if eligible else None,
        "notes": ("No model is automatically installed. Development comparisons select at "
                  "most one candidate, evaluated once on separate audit seeds. "
                  "At least three distinct training seeds and 48 episodes per "
                  "comparison scenario are required for manual review. "
                  "A positive audit remains a research screening result, not proof "
                  "of generalization; independent replication is still needed."),
    }
    _write_report(output_dir / "summary.json", report)
    return report
