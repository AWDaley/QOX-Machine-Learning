# QOX Machine Learning

**Current status: M0–M5c implemented (deterministic ecosystem, local perception, cow/farmer PPO, nomad calls/mobile defense, and alternating learned-herd training).** QOX is an experimental artificial-life simulation with farmers, cows, nomads, scavengers and ravagers. World rules are deterministic and auditable; emergent strategies must be learned, not scripted as outcomes.

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

## Train farmer networks and test refuge (M4)

```powershell
qox train-farmer --steps 16384 --scenario farmer_mixed --seed 47 --output runs/farmer.pt
qox benchmark-farmer --checkpoint runs/farmer.pt --scenario agriculture --episodes 20
qox benchmark-farmer --checkpoint runs/farmer.pt --scenario refuge --episodes 20
qox run --ticks 200 --mode learned-farmer --checkpoint runs/farmer.pt --trace runs/farmer-trace.json
qox replay --trace runs/farmer-trace.json
qox viewer --mode learned-farmer --checkpoint runs/farmer.pt
```

Farmers can cultivate/harvest, collect and apply cattle manure, transfer crops to their home store, enter finite-capacity shelters, eat from stored food, and leave refuge by moving back outside. Refuge blocks physical attacks but consumes finite stored food. The farmer uses **76 local features** and its own PyTorch model, while the original **60-feature cow model stays compatible**.

The short farmer curricula contain one trainable farmer and scripted neighbors. On held-out local tests, the learned farmer beat random legal actions at food deposition and survival, but trailed the scripted agriculture baseline. **Do not interpret this as mastery of long-term farming, manure management, or stable ecosystem cooperation.** See [M4 learning results and caveats](docs/M4_FARMER_LEARNING.md).

## Nomad calls and learned-herd experiments (M5)

M5 implements **friendly (pull)** and **scary (push)** calls as separate local sound signals, physically fast nomad dashes, ravager-only defensive melee, optional 66-input call-aware cow checkpoints, 72-input nomad PPO, scripted baselines, and no-call ablations. Nomad calls alter **cow sensory inputs only**, never cattle positions directly.

```powershell
# Learned nomad, with scripted responsive cows and predators
qox train-nomad --scenario nomad_herd --steps 16384 --seed 61 --output runs/nomad.pt
qox benchmark-nomad --checkpoint runs/nomad.pt --scenario nomad_herd --episodes 24

# Train a new call-aware cow v2 model (M3 60-input models still work)
qox train-cow --scenario nomad_herd --cow-calls --steps 8192 --seed 67 --output runs/cow-v2.pt
qox benchmark-cow --checkpoint runs/cow-v2.pt --scenario nomad_herd --episodes 16

# Run both learned policies together; remaining species are scripted
qox viewer --mode learned-herd --checkpoint runs/nomad.pt --cow-checkpoint runs/cow-v2.pt
```

**Full-herd learning is not yet reliable:** the first nomad PPO performed similarly to random legal actions in held-out pasture tests and still trails the scripted herder; the original M5 cow-v2 model showed no clear causal benefit from calls in the full migration scenario. See [M5 mechanics, benchmarks, and limitations](docs/M5_NOMADS_COMMUNICATION.md). Do not treat scripted call responses as trained behavior.

### Controlled signal-learning follow-up

The newer `cow_signal` curriculum independently randomizes the nomad's position and whether **friendly means approach** or **scary means retreat**, then rewards actual movement to an unseen training waypoint. This isolates call interpretation from visible pasture. A 65,536-decision cow PPO run reached its target on **57/96 (59.4%)** held-out episodes, versus **38/96 (39.6%)** with no calls, **0/96 with reversed calls**, and **17/96 (17.7%)** for random legal actions. The scripted responder achieved 75/96. This is a limited but encouraging learned signal-response result—not yet coordinated herding.

```powershell
qox train-cow --scenario cow_signal --cow-calls --steps 65536 --worlds 4 --rollout 256 --max-steps 24 --seed 513 --output runs/cow-signal.pt
qox benchmark-cow --checkpoint runs/cow-signal.pt --scenario cow_signal --episodes 96 --seed 205000 --max-steps 24 --report runs/cow-signal-results.json
```

See [the controlled call curriculum and paired counterfactual results](docs/M5_SIGNAL_CURRICULUM.md).

## Implemented

- Seeded 2D world with vegetation, fertile soil, crops, cattle waste, milk, carcasses, storage and energy expenditure.
- Simultaneous cardinal movement with deterministic tie resolution, immutable action traces, replay and invariant tests.
- Five species with scripted/random baseline actions and finite resources.
- **M2:** 60 named, bounded local observation features; vision with cover occlusion, waste scent gradients, recent sound cues, actor-relative targets and role-specific legal-action masks.
- **M2:** Core simultaneous-action environment plus optional PettingZoo ParallelEnv adapter exposing Gymnasium Dict observations containing `observation` and `action_mask`.
- **M3:** Compact PyTorch actor-critic, masked PPO, configurable curricula, saved/checkable policies, frozen-policy evaluation against two baselines, and a learned-cow viewer mode.
- **M4:** Farmer-only action/observation extension, finite-capacity refuge and food ration accounting, cow crop damage and manure/fertilizer, farmer PPO and agriculture/manure/refuge benchmarks, learned-farmer viewer and trace replay.
- **M5:** Nomad two-tile dash, melee attacks against ravagers, distinct friendly/scary sound calls with finite costs/cooldowns, opt-in 66-input neural cow observations, 72-input nomad policy and PPO curricula, no-call evaluation ablations, combined learned-herd viewer and trace replay.
- Simple non-stealth ravager contact attack to create a real early survival threat.
- CLI, ANSI and Pygame visualization, structured events, automated Python CI.

## Not yet implemented

Scavengers and ravagers **do not have learned policies yet**. M6–M9 will add ravager stealth/charge/drag/hibernate, advanced scavenger nutrient competition, solo/combined festivals and reproduction. M5's neural cattle have demonstrated **limited call sensitivity in a controlled task**, but not reliable full-herd communication; and neural nomads have not mastered herding or defense; the effective scripted counterparts are reference policies, not scientific evidence of emergence. The simulation has not demonstrated long-run stable ecology.

## Documents

- [Complete implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [M2–M3 observations, training and baseline results](docs/M2_M3_LEARNING.md)
- [M4 farmer agriculture/refuge and training benchmarks](docs/M4_FARMER_LEARNING.md)
- [M5 nomad communication, cow-v2 policies and honest benchmarks](docs/M5_NOMADS_COMMUNICATION.md)
- [M5 controlled two-signal curriculum and counterfactual tests](docs/M5_SIGNAL_CURRICULUM.md)

## Scientific reproducibility

Training checkpoints record policy weights, hyperparameters, action/observation schemas and training steps. Evaluation uses separate seed ranges, frozen model weights, identical world seeds for the policy and baselines, and raw results in JSON. Action traces can be replayed against hash-checked world state. The learning process is stochastic and results depend on seed, training budget and scenario. All local artifacts under `runs/` are intentionally ignored by Git.


## M5b: learned nomad signal selection and herd transfer

The new [M5b curriculum and benchmark](docs/M5B_NOMAD_CURRICULUM.md) trains a
version-2 nomad policy to choose between friendly and scary calls in a controlled
one-cow task. Frozen-policy counterfactuals (no calls and reversed calls) verify
that the signal is causally responsible for movement. In the isolated task, a
32,768-decision policy reached its waypoint in 96/96 held-out episodes; with
**three scripted cows** it reached 2.96/3 on 24 held-out migration worlds.
This **does not yet** transfer reliably to three independently learned cows.

Run the new training task:

```powershell
qox train-nomad --scenario nomad_signal --steps 32768 --max-steps 8 --output runs/nomad-signal.pt
qox benchmark-nomad --scenario nomad_signal --checkpoint runs/nomad-signal.pt --episodes 96 --max-steps 8
```

Existing v1 nomad checkpoints and legacy cow checkpoints still load. Use
`--nomad-v2` when continuing the new model on the normal `nomad_herd` scenario;
use `--init-checkpoint` to warm-start from the signal-learning checkpoint.

## M5c: alternating training with frozen learned partners

M5c trains **all three cows** in one shared PPO policy against a frozen learned nomad, then optionally swaps which species learns. The cow network sees no global pasture destination and the calls never physically move cattle. Nomad v1/v2 and cow v2 checkpoint formats remain compatible.

```powershell
qox train-herd --phase cow --cow-checkpoint checkpoints/cow_signal_65k.pt --nomad-checkpoint checkpoints/nomad_signal_v2_32k.pt --cow-steps 24576 --seed 409 --output-dir runs/m5c
qox benchmark-herd --cow-checkpoint runs/m5c/cow_round1.pt --nomad-checkpoint checkpoints/nomad_signal_v2_32k.pt --episodes 48 --seed 460000 --report runs/m5c-eval.json
qox viewer --mode learned-herd --checkpoint checkpoints/nomad_signal_v2_32k.pt --cow-checkpoint runs/m5c/cow_round1.pt
```

In a separately held-out **48-episode** test, the selected learned cow + unchanged nomad reached **1.96/3** cows to pasture, up from **0.65/3** for the original learned pair; removing the calls reduced arrivals to **1.50/3**, and reversing them to **0.25/3**. Scripted herding remains stronger at **2.77/3**, and the learned-vs-random nomad comparison has wide uncertainty. See the [M5c experiment methodology and limitations](docs/M5C_JOINT_HERD_LEARNING.md). These scores measure short migration scenarios, not demonstrated stable ecology or learned predator defense.
