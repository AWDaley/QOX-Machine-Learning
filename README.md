# QOX Machine Learning

**Current status: M0–M3 implemented (ecosystem foundation, local perception, ParallelEnv adapter, first cow neural-network policy).** QOX is an experimental artificial-life simulation with farmers, cows, nomads, scavengers and ravagers. World rules are deterministic and auditable; emergent strategies must be learned, not scripted as outcomes.

## Quick start

Python 3.11+; no Docker, database or GPU required. Core simulation and perception use NumPy. PettingZoo and PyTorch are optional extras.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ml,viewer]"
python -m pytest -q
qox run --ticks 200 --mode scripted --seed 42
qox viewer --seed 42 --mode scripted
```

On macOS/Linux, activate with `source .venv/bin/activate`. To install only basic simulation requirements, use `pip install -e .`. To install the PettingZoo adapter without PyTorch, use `pip install -e ".[marl]"`. The `dev` extra includes PettingZoo for conformance tests.

## Learn with real neural networks (M3)

```powershell
# Local CPU-first training, starting with 8,192 cow decisions
qox train-cow --steps 8192 --seed 42 --scenario forage --output runs/cow.pt

# Held-out evaluation: frozen learned policy vs random and scripted cows
qox benchmark-cow --checkpoint runs/cow.pt --episodes 30 --seed 100000 --report runs/benchmark.json

# Run learned cows in the full ecosystem; others still scripted
qox run --ticks 200 --mode learned-cow --checkpoint runs/cow.pt --trace runs/cow-trace.json
qox replay --trace runs/cow-trace.json
qox viewer --mode learned-cow --checkpoint runs/cow.pt
```

The training scenario is **deliberately simplified**, with patchy grass, one cow and stronger metabolism. It is not the same as the full 40×40 world. In the combined world the policy may transfer imperfectly. For a more substantial experiment, try `--steps 65536` and compare several training seeds. The viewer supports **Space** pause, **N** single-step, **+/-** speed, **click** to inspect an actor, and **Esc** quit. Learned cow mode displays an actor's two most probable actions.

To evaluate an individual mode or a threat scenario:

```powershell
qox eval-cow --checkpoint runs/cow.pt --mode policy --episodes 20 --seed 100000
qox eval-cow --mode random --episodes 20 --seed 100000
qox inspect --species cow --seed 42
```

The `threat` learning scenario includes a minimal scripted ravager contact attack; advanced hunting, stealth and charge mechanics will be added in M6. Do not interpret M3 as demonstrating learned herding or tactical predator evasion yet.

## Implemented

- Seeded 2D world with vegetation, fertile soil, crops, cattle waste, milk, carcasses, storage and energy expenditure.
- Simultaneous cardinal movement with deterministic tie resolution, immutable action traces, replay and invariant tests.
- Five species with scripted/random baseline actions and finite resources.
- **M2:** 60 named, bounded local observation features; vision with cover occlusion, waste scent gradients, recent sound cues, actor-relative targets and role-specific legal-action masks.
- **M2:** Core simultaneous-action environment plus optional PettingZoo ParallelEnv adapter exposing Gymnasium Dict observations containing `observation` and `action_mask`.
- **M3:** Compact PyTorch actor-critic, masked PPO, configurable curricula, saved/checkable policies, frozen-policy evaluation against two baselines, and a learned-cow viewer mode.
- Simple non-stealth ravager contact attack to create a real early survival threat.
- CLI, ANSI and Pygame visualization, structured events, automated Python CI.

## Not yet implemented

The other four species **do not have learned policies yet**. M4–M9 will add farmer refuge and advanced agriculture, nomad push/pull calls and defense, ravager stealth/charge/drag/hibernate, scavenger nutrient competition, solo/combined festivals and reproduction. Features described in the design plan are not claims of completed code. QOX currently demonstrates foraging-policy training, not a validated stable ecology.

## Documents

- [Complete implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [M2–M3 observations, training and baseline results](docs/M2_M3_LEARNING.md)

## Scientific reproducibility

Training checkpoints record policy weights, hyperparameters, action/observation schemas and training steps. Evaluation uses separate seed ranges, frozen model weights, identical world seeds for the policy and baselines, and raw results in JSON. Action traces can be replayed against hash-checked world state. The learning process is stochastic and results depend on seed, training budget and scenario. All local artifacts under `runs/` are intentionally ignored by Git.
