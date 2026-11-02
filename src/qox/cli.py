"""Command line entry point: run, replay, and optional interactive viewer."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import SimulationConfig
from .trace import Trace, replay, run


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QOX artificial-life simulator")
    sub = p.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("run", help="Run a headless deterministic world")
    simulate.add_argument("--ticks", type=int, default=200)
    simulate.add_argument("--seed", type=int)
    simulate.add_argument("--config", type=Path)
    simulate.add_argument("--mode", choices=("scripted", "random", "idle", "learned-cow", "learned-farmer", "learned-nomad", "learned-herd", "learned-ravager", "learned-scavenger", "learned-festival"), default="scripted")
    simulate.add_argument("--report", type=Path)
    simulate.add_argument("--trace", type=Path)
    simulate.add_argument("--checkpoint", type=Path)
    simulate.add_argument("--festivals", action="store_true", help="Enable M8 festivals")
    simulate.add_argument("--reproduction", action="store_true", help="Enable M9 births and juveniles")
    simulate.add_argument("--cow-checkpoint", type=Path,
                          help="Second checkpoint for learned-herd mode")
    check = sub.add_parser("replay", help="Verify deterministic action trace")
    check.add_argument("--trace", type=Path, required=True)
    replay_viewer = sub.add_parser("replay-viewer", help="Graphical, seekable trace replay")
    replay_viewer.add_argument("--trace", type=Path, required=True)
    view = sub.add_parser("viewer", help="View the world in Pygame (optional extra)")
    view.add_argument("--seed", type=int, default=42)
    view.add_argument("--config", type=Path)
    view.add_argument("--festival-scenario", choices=("natural", "festival_farmer",
                      "festival_nomad", "festival_combined", "festival_ravager"),
                      default="natural", help="Watch a preconfigured M8 festival")
    view.add_argument("--mode", choices=("scripted", "random", "idle", "learned-cow", "learned-farmer", "learned-nomad", "learned-herd", "learned-ravager", "learned-scavenger", "learned-festival"), default="scripted")
    view.add_argument("--checkpoint", type=Path)
    view.add_argument("--festivals", action="store_true", help="Enable M8 festivals")
    view.add_argument("--reproduction", action="store_true", help="Enable M9 births and juveniles")
    view.add_argument("--cow-checkpoint", type=Path,
                      help="Second checkpoint for learned-herd mode")
    view.add_argument("--record-trace", type=Path,
                      help="Record viewer actions for later replay-viewer inspection")
    study = sub.add_parser("population-study", help="M9: frozen-policy long-horizon multi-seed ecology evaluation")
    study.add_argument("--ticks", type=int, default=600)
    study.add_argument("--seeds", type=int, default=3)
    study.add_argument("--seed", type=int, default=200000)
    study.add_argument("--sample-every", type=int, default=60)
    study.add_argument("--mode", choices=("scripted", "random", "idle", "learned"),
                       default="scripted")
    study.add_argument("--config", type=Path)
    study.add_argument("--festivals", action="store_true")
    study.add_argument("--without-births", action="store_true",
                       help="Counterfactual: same settings, reproduction disabled")
    study.add_argument("--farmer-checkpoint", type=Path)
    study.add_argument("--cow-checkpoint", type=Path)
    study.add_argument("--nomad-checkpoint", type=Path)
    study.add_argument("--scavenger-checkpoint", type=Path)
    study.add_argument("--ravager-checkpoint", type=Path)
    study.add_argument("--festival-checkpoint", type=Path)
    study.add_argument("--report", type=Path, default=Path("runs/m9_population_study.json"))
    train = sub.add_parser("train-cow", help="Train a shared cow policy with masked PPO")
    train.add_argument("--steps", type=int, default=8192)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--worlds", type=int, default=4)
    train.add_argument("--rollout", type=int, default=512)
    train.add_argument("--max-steps", type=int, default=96)
    train.add_argument("--scenario", choices=("forage", "threat", "nomad_herd", "cow_signal"), default="forage")
    train.add_argument("--cow-calls", action="store_true",
                       help="Train v2 66-feature cow policy to interpret nomad calls")
    train.add_argument("--output", type=Path, default=Path("runs/cow.pt"))
    evaluate = sub.add_parser("eval-cow", help="Evaluate a frozen cow policy or baseline")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--mode", choices=("policy", "policy_nocalls", "policy_flipped", "random",
                                           "scripted", "scripted_nocalls", "scripted_flipped"),
                          default="policy")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=100000)
    evaluate.add_argument("--max-steps", type=int, default=96)
    evaluate.add_argument("--scenario", choices=("forage", "threat", "nomad_herd", "cow_signal"), default="forage")
    evaluate.add_argument("--report", type=Path)
    benchmark = sub.add_parser("benchmark-cow", help="Compare policy to two baselines")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--episodes", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=100000)
    benchmark.add_argument("--max-steps", type=int, default=96)
    benchmark.add_argument("--scenario", choices=("forage", "threat", "nomad_herd", "cow_signal"), default="forage")
    benchmark.add_argument("--report", type=Path)
    farmer_scenarios = ("agriculture", "manure", "refuge", "farmer_mixed")
    train_farmer = sub.add_parser("train-farmer", help="Train farmer PPO with crop/manure/refuge curricula")
    train_farmer.add_argument("--steps", type=int, default=16384)
    train_farmer.add_argument("--seed", type=int, default=42)
    train_farmer.add_argument("--worlds", type=int, default=4)
    train_farmer.add_argument("--rollout", type=int, default=512)
    train_farmer.add_argument("--max-steps", type=int, default=96)
    train_farmer.add_argument("--scenario", choices=farmer_scenarios, default="farmer_mixed")
    train_farmer.add_argument("--output", type=Path, default=Path("runs/farmer.pt"))
    for title in ("eval-farmer", "benchmark-farmer"):
        cmd = sub.add_parser(title, help="Evaluate the frozen farmer policy and baselines")
        cmd.add_argument("--checkpoint", type=Path)
        cmd.add_argument("--episodes", type=int, default=20)
        cmd.add_argument("--seed", type=int, default=100000)
        cmd.add_argument("--max-steps", type=int, default=96)
        cmd.add_argument("--scenario", choices=farmer_scenarios, default="agriculture")
        cmd.add_argument("--mode", choices=("policy", "random", "scripted"), default="policy")
        cmd.add_argument("--report", type=Path)
    nomad_scenarios = ("nomad_signal", "nomad_herd", "nomad_defend", "nomad_mixed")
    train_nomad = sub.add_parser("train-nomad", help="Train a nomad with cattle signals and defense")
    train_nomad.add_argument("--steps", type=int, default=16384)
    train_nomad.add_argument("--seed", type=int, default=42)
    train_nomad.add_argument("--worlds", type=int, default=4)
    train_nomad.add_argument("--rollout", type=int, default=256)
    train_nomad.add_argument("--max-steps", type=int, default=96)
    train_nomad.add_argument("--scenario", choices=nomad_scenarios, default="nomad_herd")
    train_nomad.add_argument("--output", type=Path, default=Path("runs/nomad.pt"))
    train_nomad.add_argument("--init-checkpoint", type=Path,
                             help="Warm-start from a compatible trained nomad policy")
    train_nomad.add_argument("--nomad-v2", action="store_true",
                             help="Use derived goal-relative-to-visible-herd features")
    for title in ("eval-nomad", "benchmark-nomad"):
        cmd = sub.add_parser(title, help="Evaluate a frozen nomad policy and call ablations")
        cmd.add_argument("--checkpoint", type=Path)
        cmd.add_argument("--cow-checkpoint", type=Path,
                         help="Evaluate with frozen call-aware learned cows, not scripted cows")
        cmd.add_argument("--episodes", type=int, default=12)
        cmd.add_argument("--seed", type=int, default=100000)
        cmd.add_argument("--max-steps", type=int, default=96)
        cmd.add_argument("--scenario", choices=nomad_scenarios, default="nomad_herd")
        cmd.add_argument("--mode", choices=("policy", "policy_greedy", "policy_nocalls",
                                            "policy_greedy_nocalls", "policy_greedy_flipped",
                                            "random", "scripted", "scripted_nocalls",
                                            "scripted_flipped"), default="policy")
        cmd.add_argument("--report", type=Path)
    joint = sub.add_parser("train-herd", help="M5c: alternate cow/nomad PPO against frozen partners")
    joint.add_argument("--cow-checkpoint", type=Path, required=True)
    joint.add_argument("--nomad-checkpoint", type=Path, required=True)
    joint.add_argument("--output-dir", type=Path, default=Path("runs/joint-herd"))
    joint.add_argument("--phase", choices=("alternate", "cow", "nomad"),
                       default="alternate", help="Train both roles or keep one frozen")
    joint.add_argument("--cow-steps", type=int, default=12288)
    joint.add_argument("--nomad-steps", type=int, default=4096)
    joint.add_argument("--rounds", type=int, default=1)
    joint.add_argument("--worlds", type=int, default=4)
    joint.add_argument("--rollout", type=int, default=384)
    joint.add_argument("--max-steps", type=int, default=96)
    joint.add_argument("--seed", type=int, default=407)
    joint.add_argument("--epochs", type=int, default=3)
    comparison = sub.add_parser("benchmark-herd", help="M5c: paired two-learned-policy call ablations")
    comparison.add_argument("--cow-checkpoint", type=Path, required=True)
    comparison.add_argument("--nomad-checkpoint", type=Path, required=True)
    comparison.add_argument("--episodes", type=int, default=24)
    comparison.add_argument("--seed", type=int, default=340000)
    comparison.add_argument("--max-steps", type=int, default=96)
    comparison.add_argument("--report", type=Path)
    ravager_scenarios = ("ravager_ambush", "ravager_carcass",
                         "ravager_hibernate", "ravager_mixed")
    predator = sub.add_parser("train-ravager", help="M6: train masked PPO ravager policy")
    predator.add_argument("--steps", type=int, default=8192)
    predator.add_argument("--seed", type=int, default=42)
    predator.add_argument("--worlds", type=int, default=4)
    predator.add_argument("--rollout", type=int, default=256)
    predator.add_argument("--max-steps", type=int, default=64)
    predator.add_argument("--scenario", choices=ravager_scenarios, default="ravager_mixed")
    predator.add_argument("--output", type=Path, default=Path("runs/ravager.pt"))
    predator.add_argument("--init-checkpoint", type=Path)
    for title in ("eval-ravager", "benchmark-ravager"):
        cmd = sub.add_parser(title, help="M6: frozen ravager policy and ablations")
        cmd.add_argument("--checkpoint", type=Path)
        cmd.add_argument("--episodes", type=int, default=16)
        cmd.add_argument("--seed", type=int, default=100000)
        cmd.add_argument("--max-steps", type=int, default=64)
        cmd.add_argument("--scenario", choices=ravager_scenarios, default="ravager_ambush")
        cmd.add_argument("--mode", choices=("policy", "policy_greedy", "random",
                                            "scripted", "policy_nocharge", "policy_nosneak",
                                            "policy_nohibernate"), default="policy_greedy")
        cmd.add_argument("--report", type=Path)
    scavenger_scenarios = ("scavenger_waste", "scavenger_carrion",
                           "scavenger_competition", "scavenger_mixed")
    scavenger = sub.add_parser("train-scavenger", help="M7: learn waste/carrion scavenging")
    scavenger.add_argument("--steps", type=int, default=8192)
    scavenger.add_argument("--seed", type=int, default=42)
    scavenger.add_argument("--worlds", type=int, default=4)
    scavenger.add_argument("--rollout", type=int, default=256)
    scavenger.add_argument("--max-steps", type=int, default=48)
    scavenger.add_argument("--scenario", choices=scavenger_scenarios,
                           default="scavenger_mixed")
    scavenger.add_argument("--output", type=Path, default=Path("runs/scavenger.pt"))
    scavenger.add_argument("--init-checkpoint", type=Path)
    for title in ("eval-scavenger", "benchmark-scavenger"):
        cmd = sub.add_parser(title, help="M7: evaluate scavenger PPO and controls")
        cmd.add_argument("--checkpoint", type=Path)
        cmd.add_argument("--episodes", type=int, default=16)
        cmd.add_argument("--seed", type=int, default=100000)
        cmd.add_argument("--max-steps", type=int, default=48)
        cmd.add_argument("--scenario", choices=scavenger_scenarios,
                         default="scavenger_mixed")
        cmd.add_argument("--mode", choices=("policy", "policy_greedy", "random",
                                            "scripted", "policy_nowaste",
                                            "policy_nocarrion"), default="policy_greedy")
        cmd.add_argument("--report", type=Path)
    festival_scenarios = ("festival_farmer", "festival_nomad",
                          "festival_combined", "festival_ravager")
    festival_demo = sub.add_parser("festival-demo", help="Run scripted M8 festival scenarios")
    festival_demo.add_argument("--scenario", choices=festival_scenarios,
                               default="festival_combined")
    festival_demo.add_argument("--ticks", type=int, default=20)
    festival_demo.add_argument("--seed", type=int, default=42)
    festival_demo.add_argument("--report", type=Path)
    train_festival = sub.add_parser("train-festival", help="Train M8 festival timing PPO")
    train_festival.add_argument("--scenario", choices=festival_scenarios,
                                default="festival_farmer")
    train_festival.add_argument("--steps", type=int, default=8192)
    train_festival.add_argument("--seed", type=int, default=42)
    train_festival.add_argument("--worlds", type=int, default=4)
    train_festival.add_argument("--rollout", type=int, default=256)
    train_festival.add_argument("--max-steps", type=int, default=16)
    train_festival.add_argument("--output", type=Path, default=Path("runs/festival.pt"))
    for title in ("eval-festival", "benchmark-festival"):
        cmd = sub.add_parser(title, help="Evaluate M8 festival completion and baselines")
        cmd.add_argument("--scenario", choices=festival_scenarios,
                         default="festival_farmer")
        cmd.add_argument("--checkpoint", type=Path)
        cmd.add_argument("--episodes", type=int, default=24)
        cmd.add_argument("--seed", type=int, default=100000)
        cmd.add_argument("--max-steps", type=int, default=16)
        cmd.add_argument("--mode", choices=("policy", "policy_greedy", "random",
                                            "scripted", "no_start"),
                         default="policy_greedy")
        cmd.add_argument("--report", type=Path)
    inspect = sub.add_parser("inspect", help="Show one actor's local observation and action mask")
    inspect.add_argument("--seed", type=int, default=42)
    inspect.add_argument("--festivals", action="store_true",
                         help="Opt into M8 festival observation/action contract")
    inspect.add_argument("--cow-calls", action="store_true",
                         help="Opt into v2 cow call features (legacy v1 is default)")
    inspect.add_argument("--species", choices=("farmer", "cow", "nomad", "scavenger", "ravager"),
                         default="cow")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "population-study":
        from .evaluation.population import run_study
        cfg = SimulationConfig.from_toml(args.config) if args.config else SimulationConfig()
        cfg = replace(cfg, population_dynamics_enabled=True,
                      reproduction_enabled=not args.without_births,
                      festival_enabled=cfg.festival_enabled or args.festivals)
        checkpoint_paths = {species: getattr(args, f"{species}_checkpoint")
                            for species in ("farmer", "cow", "nomad", "scavenger", "ravager")}
        result = run_study(cfg, ticks=args.ticks, seeds=args.seeds,
                           initial_seed=args.seed, sample_every=args.sample_every,
                           mode=args.mode, checkpoints=checkpoint_paths,
                           festival_checkpoint=args.festival_checkpoint)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"mode": result["mode"], "mean_final_population":
                          result["mean_final_population"],
                          "mean_births": result["mean_births"],
                          "mean_deaths": result["mean_deaths"],
                          "mean_festival_successes": result["mean_festival_successes"],
                          "report": str(args.report)}, indent=2))
        return 0
    if args.command == "festival-demo":
        from .agents import scripted_actions
        from .envs.core import QoxCoreEnv
        env = QoxCoreEnv(scenario=args.scenario, max_steps=args.ticks)
        env.reset(seed=args.seed)
        for _ in range(args.ticks):
            env.world.step(scripted_actions(env.world))
        report = env.world.summary()
        report["festival_results"] = [
            {"kind": r.kind.value, "succeeded": r.succeeded,
             "tick": r.tick, "reason": r.reason}
            for r in env.world.festival_results]
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.command in ("train-festival", "eval-festival", "benchmark-festival"):
        try:
            from .learning.festival_ppo import (FestivalTrainOptions, train_festival,
                                                evaluate_festival)
        except ImportError as exc:
            raise SystemExit("Festival learning requires PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-festival":
            report = train_festival(FestivalTrainOptions(
                scenario=args.scenario, steps=args.steps, seed=args.seed,
                worlds=args.worlds, rollout=args.rollout,
                max_steps=args.max_steps), args.output)
            print(json.dumps({k: v for k, v in report.items() if k not in
                              ("history", "recent_returns")}, indent=2))
            return 0
        modes = ("policy_greedy", "policy", "random", "scripted", "no_start") if (
            args.command == "benchmark-festival") else (args.mode,)
        reports = {mode: evaluate_festival(args.checkpoint, scenario=args.scenario,
                                          episodes=args.episodes, max_steps=args.max_steps,
                                          seed=args.seed, mode=mode) for mode in modes}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(json.dumps({mode: {k: v for k, v in result.items() if k != "results"}
                          for mode, result in reports.items()}, indent=2))
        return 0
    if args.command in ("train-scavenger", "eval-scavenger", "benchmark-scavenger"):
        try:
            from .learning.scavenger_ppo import (ScavengerTrainOptions, train_scavenger,
                                                 evaluate_scavenger)
        except ImportError as exc:
            raise SystemExit("Scavenger learning requires PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-scavenger":
            result = train_scavenger(ScavengerTrainOptions(
                steps=args.steps, seed=args.seed, worlds=args.worlds,
                rollout=args.rollout, max_steps=args.max_steps, scenario=args.scenario),
                args.output, init_checkpoint=args.init_checkpoint)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in ("history", "recent_returns")}, indent=2))
            return 0
        modes = (("policy_greedy", "policy", "random", "scripted",
                  "policy_nowaste", "policy_nocarrion")
                 if args.command == "benchmark-scavenger" else (args.mode,))
        result = {mode: evaluate_scavenger(
            args.checkpoint, episodes=args.episodes, seed=args.seed,
            max_steps=args.max_steps, scenario=args.scenario, mode=mode)
                  for mode in modes}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({mode: {k: v for k, v in results.items() if k != "results"}
                          for mode, results in result.items()}, indent=2))
        return 0
    if args.command in ("train-ravager", "eval-ravager", "benchmark-ravager"):
        try:
            from .learning.ravager_ppo import (RavagerTrainOptions, train_ravager,
                                               evaluate_ravager)
        except ImportError as exc:
            raise SystemExit("Ravager training requires PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-ravager":
            result = train_ravager(RavagerTrainOptions(
                steps=args.steps, seed=args.seed, worlds=args.worlds,
                rollout=args.rollout, max_steps=args.max_steps, scenario=args.scenario),
                args.output, init_checkpoint=args.init_checkpoint)
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("history", "recent_returns")}, indent=2))
            return 0
        modes = (("policy_greedy", "policy", "random", "scripted",
                  "policy_nocharge", "policy_nosneak", "policy_nohibernate")
                 if args.command == "benchmark-ravager" else (args.mode,))
        results = {mode: evaluate_ravager(
            args.checkpoint, episodes=args.episodes, seed=args.seed,
            max_steps=args.max_steps, scenario=args.scenario, mode=mode)
                   for mode in modes}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({k: {key: value for key, value in report.items()
                              if key != "results"} for k, report in results.items()}, indent=2))
        return 0
    if args.command in ("train-herd", "benchmark-herd"):
        try:
            from .learning.joint_ppo import (JointTrainOptions, benchmark_joint,
                                             train_joint)
        except ImportError as exc:
            raise SystemExit("Herd training requires PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-herd":
            report = train_joint(JointTrainOptions(
                cow_steps=args.cow_steps, nomad_steps=args.nomad_steps,
                seed=args.seed, rounds=args.rounds, worlds=args.worlds,
                rollout=args.rollout, max_steps=args.max_steps,
                epochs=args.epochs), args.cow_checkpoint,
                args.nomad_checkpoint, args.output_dir, phase=args.phase)
            print(json.dumps({k: v for k, v in report.items() if k != "rounds"},
                             indent=2))
            return 0
        report = benchmark_joint(args.cow_checkpoint, args.nomad_checkpoint,
                                 episodes=args.episodes, seed=args.seed,
                                 max_steps=args.max_steps)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "paired_results"},
                         indent=2))
        return 0
    if args.command in ("train-nomad", "eval-nomad", "benchmark-nomad"):
        try:
            from .learning.nomad_ppo import (NomadTrainOptions, train_nomad,
                                             evaluate_nomad)
        except ImportError as exc:
            raise SystemExit("Nomad training requires PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-nomad":
            result = train_nomad(NomadTrainOptions(
                steps=args.steps, seed=args.seed, worlds=args.worlds,
                rollout=args.rollout, max_steps=args.max_steps,
                scenario=args.scenario,
                nomad_v2=args.nomad_v2 or args.scenario == "nomad_signal"),
                args.output, init_checkpoint=args.init_checkpoint)
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("history", "recent_returns")}, indent=2))
            return 0
        modes = (("policy_greedy", "policy_greedy_nocalls", "policy_greedy_flipped",
                  "random", "scripted", "scripted_nocalls", "scripted_flipped")
                 if args.command == "benchmark-nomad" and args.scenario == "nomad_signal"
                 else ("policy", "policy_greedy", "policy_nocalls", "random", "scripted",
                       "scripted_nocalls")
                 if args.command == "benchmark-nomad" else (args.mode,))
        report = {mode: evaluate_nomad(args.checkpoint, episodes=args.episodes,
                                       seed=args.seed, max_steps=args.max_steps,
                                       scenario=args.scenario, mode=mode,
                                       cow_checkpoint=args.cow_checkpoint)
                  for mode in modes}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({mode: {k: v for k, v in data.items() if k != "results"}
                          for mode, data in report.items()}, indent=2))
        return 0
    if args.command in ("train-farmer", "eval-farmer", "benchmark-farmer"):
        try:
            from .learning.farmer_ppo import (FarmerTrainOptions, train_farmer,
                                              evaluate_farmer)
        except ImportError as exc:
            raise SystemExit("Farmer learning requires PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-farmer":
            report = train_farmer(FarmerTrainOptions(
                steps=args.steps, seed=args.seed, worlds=args.worlds,
                rollout=args.rollout, max_steps=args.max_steps,
                scenario=args.scenario), args.output)
            print(json.dumps({k: v for k, v in report.items()
                              if k not in ("history", "recent_returns")}, indent=2))
            return 0
        modes = (("policy", "random", "scripted") if args.command == "benchmark-farmer"
                 else (args.mode,))
        results = {mode: evaluate_farmer(checkpoint=args.checkpoint,
                                         episodes=args.episodes, seed=args.seed,
                                         max_steps=args.max_steps, scenario=args.scenario,
                                         mode=mode) for mode in modes}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({mode: {k: v for k, v in result.items() if k != "results"}
                          for mode, result in results.items()}, indent=2))
        return 0
    if args.command in ("train-cow", "eval-cow", "benchmark-cow"):
        try:
            from .learning.cow_ppo import TrainOptions, evaluate_cow, train_cow
        except ImportError as exc:
            raise SystemExit("Cow training needs PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-cow":
            result = train_cow(TrainOptions(steps=args.steps, seed=args.seed,
                                          worlds=args.worlds, rollout=args.rollout,
                                          max_steps=args.max_steps,
                                          scenario=args.scenario,
                                          cow_calls=args.cow_calls), args.output)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in ("history", "recent_returns")}, indent=2))
            return 0
        modes = (("policy", "policy_nocalls", "policy_flipped", "random", "scripted",
                  "scripted_nocalls", "scripted_flipped")
                 if args.command == "benchmark-cow" and args.scenario == "cow_signal"
                 else ("policy", "policy_nocalls", "random", "scripted", "scripted_nocalls")
                 if args.command == "benchmark-cow" and args.scenario == "nomad_herd"
                 else ("policy", "random", "scripted") if args.command == "benchmark-cow"
                 else (args.mode,))
        report = {mode: evaluate_cow(checkpoint=args.checkpoint,
                                     episodes=args.episodes, seed=args.seed,
                                     scenario=args.scenario, max_steps=args.max_steps,
                                     mode=mode) for mode in modes}
        compact = {mode: {k: value for k, value in data.items() if k != "results"}
                   for mode, data in report.items()}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(compact, indent=2))
        return 0
    if args.command == "inspect":
        from .entities import Species
        from .perception import action_catalog, features_for, observe
        from .world import World
        world = World(SimulationConfig(seed=args.seed,
                                       festival_enabled=args.festivals))
        actor = next(a for a in world.actors.values() if a.species == Species(args.species))
        state = observe(world, actor, cow_calls=args.cow_calls,
                        festivals=args.festivals)
        print(json.dumps({"actor": actor.id, "species": actor.species.value,
                          "features": dict(zip(features_for(actor.species, cow_calls=args.cow_calls,
                                                                   festivals=args.festivals),
                                               state["observation"].astype(float).tolist(), strict=True)),
                          "legal_actions": [action.to_dict()
                                            for action, is_legal in zip(action_catalog(actor.species,
                                                                             festivals=args.festivals),
                                                                        state["action_mask"], strict=True)
                                            if is_legal]}, indent=2))
        return 0
    if args.command == "replay":
        world = replay(Trace.load(args.trace))
        print(f"Replay OK: {world.tick} ticks; hash={world.state_hash()[:16]}")
        return 0
    if args.command == "replay-viewer":
        from .viewer import display
        trace = Trace.load(args.trace)
        display(SimulationConfig(**trace.config), replay_trace=trace)
        return 0
    config = SimulationConfig.from_toml(args.config) if args.config else SimulationConfig()
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.festivals or args.mode == "learned-festival":
        config = replace(config, festival_enabled=True)
    if getattr(args, "reproduction", False):
        config = replace(config, population_dynamics_enabled=True,
                         reproduction_enabled=True)
    if args.command == "viewer":
        from .viewer import display
        display(config, mode=args.mode, checkpoint=args.checkpoint,
                cow_checkpoint=args.cow_checkpoint,
                scenario=args.festival_scenario,
                record_trace=args.record_trace)
        return 0
    world, trace = run(config, ticks=args.ticks, mode=args.mode,
                       record=args.trace is not None, checkpoint=args.checkpoint,
                       cow_checkpoint=args.cow_checkpoint)
    report = world.summary()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if trace is not None:
        trace.save(args.trace)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
