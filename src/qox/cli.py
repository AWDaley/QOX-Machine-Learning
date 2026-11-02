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
    simulate.add_argument("--mode", choices=("scripted", "random", "idle"), default="scripted")
    simulate.add_argument("--report", type=Path)
    simulate.add_argument("--trace", type=Path)
    check = sub.add_parser("replay", help="Verify deterministic action trace")
    check.add_argument("--trace", type=Path, required=True)
    view = sub.add_parser("viewer", help="View the world in Pygame (optional extra)")
    view.add_argument("--seed", type=int, default=42)
    view.add_argument("--config", type=Path)
    view.add_argument("--mode", choices=("scripted", "random", "idle"), default="scripted")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "replay":
        world = replay(Trace.load(args.trace))
        print(f"Replay OK: {world.tick} ticks; hash={world.state_hash()[:16]}")
        return 0
    config = SimulationConfig.from_toml(args.config) if args.config else SimulationConfig()
    if args.seed is not None:
        config = SimulationConfig(**{**config.as_dict(), "seed": args.seed})
    if args.command == "viewer":
        from .viewer import display
        display(config, mode=args.mode)
        return 0
    world, trace = run(config, ticks=args.ticks, mode=args.mode,
                       record=args.trace is not None)
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
