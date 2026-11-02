"""Command line entry point: run, replay, and optional interactive viewer."""

from __future__ import annotations

import argparse
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
    simulate.add_argument("--mode", choices=("scripted", "random", "idle", "learned-cow"), default="scripted")
    simulate.add_argument("--report", type=Path)
    simulate.add_argument("--trace", type=Path)
    simulate.add_argument("--checkpoint", type=Path)
    check = sub.add_parser("replay", help="Verify deterministic action trace")
    check.add_argument("--trace", type=Path, required=True)
    view = sub.add_parser("viewer", help="View the world in Pygame (optional extra)")
    view.add_argument("--seed", type=int, default=42)
    view.add_argument("--config", type=Path)
    view.add_argument("--mode", choices=("scripted", "random", "idle", "learned-cow"), default="scripted")
    view.add_argument("--checkpoint", type=Path)
    train = sub.add_parser("train-cow", help="Train a shared cow policy with masked PPO")
    train.add_argument("--steps", type=int, default=8192)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--worlds", type=int, default=4)
    train.add_argument("--rollout", type=int, default=512)
    train.add_argument("--max-steps", type=int, default=96)
    train.add_argument("--scenario", choices=("forage", "threat"), default="forage")
    train.add_argument("--output", type=Path, default=Path("runs/cow.pt"))
    evaluate = sub.add_parser("eval-cow", help="Evaluate a frozen cow policy or baseline")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--mode", choices=("policy", "random", "scripted"),
                          default="policy")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=100000)
    evaluate.add_argument("--max-steps", type=int, default=96)
    evaluate.add_argument("--scenario", choices=("forage", "threat"), default="forage")
    evaluate.add_argument("--report", type=Path)
    benchmark = sub.add_parser("benchmark-cow", help="Compare policy to two baselines")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--episodes", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=100000)
    benchmark.add_argument("--max-steps", type=int, default=96)
    benchmark.add_argument("--scenario", choices=("forage", "threat"), default="forage")
    benchmark.add_argument("--report", type=Path)
    inspect = sub.add_parser("inspect", help="Show one actor's local observation and action mask")
    inspect.add_argument("--seed", type=int, default=42)
    inspect.add_argument("--species", choices=("farmer", "cow", "nomad", "scavenger", "ravager"),
                         default="cow")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command in ("train-cow", "eval-cow", "benchmark-cow"):
        try:
            from .learning.cow_ppo import TrainOptions, evaluate_cow, train_cow
        except ImportError as exc:
            raise SystemExit("Cow training needs PyTorch: pip install -e '.[ml]'") from exc
        if args.command == "train-cow":
            result = train_cow(TrainOptions(steps=args.steps, seed=args.seed,
                                          worlds=args.worlds, rollout=args.rollout,
                                          max_steps=args.max_steps,
                                          scenario=args.scenario), args.output)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in ("history", "recent_returns")}, indent=2))
            return 0
        modes = ("policy", "random", "scripted") if args.command == "benchmark-cow" else (args.mode,)
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
        from .perception import ACTIONS, FEATURES, observe
        from .world import World
        world = World(SimulationConfig(seed=args.seed))
        actor = next(a for a in world.actors.values() if a.species == Species(args.species))
        state = observe(world, actor)
        print(json.dumps({"actor": actor.id, "species": actor.species.value,
                          "features": dict(zip(FEATURES,
                                               state["observation"].astype(float).tolist())),
                          "legal_actions": [action.to_dict()
                                            for action, is_legal in zip(ACTIONS[actor.species],
                                                                        state["action_mask"])
                                            if is_legal]}, indent=2))
        return 0
    if args.command == "replay":
        world = replay(Trace.load(args.trace))
        print(f"Replay OK: {world.tick} ticks; hash={world.state_hash()[:16]}")
        return 0
    config = SimulationConfig.from_toml(args.config) if args.config else SimulationConfig()
    if args.seed is not None:
        config = SimulationConfig(**{**config.as_dict(), "seed": args.seed})
    if args.command == "viewer":
        from .viewer import display
        display(config, mode=args.mode, checkpoint=args.checkpoint)
        return 0
    world, trace = run(config, ticks=args.ticks, mode=args.mode,
                       record=args.trace is not None, checkpoint=args.checkpoint)
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
