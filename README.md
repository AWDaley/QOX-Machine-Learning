# QOX Machine Learning

**Status: planning phase — no runnable simulation has been implemented yet.**

QOX Machine Learning is a planned 2D, multi-agent artificial-life simulator. Five populations—**farmers, cows, nomads, scavengers, and ravagers**—will learn through local observations and neural-network policies while competing and cooperating in a shared ecosystem.

The main research question is whether local, resource-based incentives can lead to herding, refuge use, agriculture, ecological recycling, hunting tactics, and successful community festivals without scripting those strategies.

## Design highlights

| Actor | Role | Distinctive mechanics |
|---|---|---|
| Farmers | Settlement-centered food producers | Crops, manure collection/fertilization, storage, refuge |
| Cows | Herbivores and herd resource | Grazing, crop damage, milk, waste, danger avoidance |
| Nomads | Fast mobile herders and defenders | Friendly/scary cattle calls, milking, cattle protection |
| Scavengers | Waste and carrion consumers | Scent-based resource search, competition for manure |
| Ravagers | Rare apex predators | Sneak, expensive high-speed charge, drag prey, feed, hibernate |

Farmer-only, nomad-only, and combined festivals require real surplus, explicit contributions and attendance. Festivals create sound/smell cues that can attract existing ravagers; they do not spawn predators.

## Implementation roadmap

See **[the complete implementation plan](docs/IMPLEMENTATION_PLAN.md)** for the full world/actor specification, observation and action design, learning approach, reproducibility rules, risks, and testable milestones.

| Phase | Focus |
|---|---|
| M0 | Python package, configuration, CI, CLI foundation |
| M1 | Deterministic world, ecology, resources, simple viewer |
| M2 | Perception and PettingZoo multi-agent interface |
| M3 | First learned cow policy |
| M4 | Farmer agriculture, manure and settlement refuge |
| M5 | Nomad herding signals and defense |
| M6 | Ravager stealth, charge, prey handling and hibernation |
| M7 | Scavengers and nutrient cycling |
| M8 | Solo and combined festivals; ravager attraction |
| M9 | Reproduction, integrated learning and held-out evaluation |

## Engineering principles

- **Train headlessly; render separately.** Viewer framerate must not change simulation outcomes.
- **Physics and resources are explicit; strategies are learned.**
- **Reproducibility first:** fixed seeds, action traces, event logs and deterministic world updates.
- **CPU-first:** Python, PyTorch, compact policies, Pygame viewer; no Docker or database required.
- **Test against baselines:** do not treat plausible-looking behavior as proof of learning.

## Next implementation step

Start with **M0** followed by a minimal **M1** world running scripted/random actors. Confirm deterministic replay and resource conservation before adding neural-network training.
