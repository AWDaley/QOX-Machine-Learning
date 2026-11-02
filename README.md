# QOX Machine Learning

**Current status: M0–M9 implemented (deterministic ecosystem, species-specific PPO, learned-herd experiments, ravager intelligence, scavenger nutrient competition, and three festival types).** QOX is an experimental artificial-life simulation with farmers, cows, nomads, scavengers and ravagers. World rules are deterministic and auditable; emergent strategies must be learned, not scripted as outcomes.

## M12 — Readable, confirmed ecosystem events

The enlarged pixel-art viewer now shows **confirmed** action effects rather than assuming every requested action succeeded. It adds visible charge wind-ups, impacts and misses; distinct nomad cattle-call signals; harvest and fertilizer sparks; carcass handling, births, deaths, and festival outcomes. A bounded, clickable recent-event feed appears over the map. The selected actor inspector distinguishes a confirmed action from **No confirmed outcome**. These overlays never alter simulation physics, neural policy weights, or seeded replay.

New controls: **E** events panel, **T** major/all event filter, **V** event effects, **H** actor corner status badges. Click a feed row to focus the location. Read [M12 visual events implementation](docs/M12_VISUAL_EVENTS.md) for rules, tests, and limitations.

## M8 — Festivals and prosperity

The new M8 implementation adds **farmer-only, nomad-only and combined festivals**:
explicit grain/milk contributions, separate survival and celebration stores,
healthy community attendance, multi-tick celebrations, limited cooldowns,
local predator-attracting sound/scent, and tracked successes/failures.
Festivals are off in old baseline runs unless enabled by `--festivals` or a
festival scenario, preserving existing neural-network checkpoint contracts.

```powershell
qox festival-demo --scenario festival_combined --ticks 20
qox viewer --festival-scenario festival_combined
qox run --festivals --ticks 250 --mode scripted
qox train-festival --scenario festival_farmer --steps 8192 --output runs/festival.pt
qox benchmark-festival --scenario festival_farmer --checkpoint runs/festival.pt --episodes 24
qox viewer --festival-scenario festival_farmer --mode learned-festival --checkpoint runs/festival.pt
```

See [M8 Festival Economy research report](docs/M8_FESTIVAL_ECONOMY.md) for
mechanical invariants, learning conditions, held-out results, limitations, and
future ecological experiments.

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

The `threat` curriculum was designed with an early contact-bite predator; the broader world now also supports the M6 charge and stealth mechanics. Do not interpret M3 as demonstrating learned herding or tactical predator evasion yet.

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
- **M6:** Ravager sneak, costly wind-up charge, finite carcass relocation, satiety/hibernation, 82-input ravager PPO, learning/evaluation benchmarks and learned-ravager viewer mode.
- CLI, ANSI and Pygame visualization, structured events, automated Python CI.

## Not yet implemented

Scavengers now have learned PPO policies and nutrient-recycling rules (M7); M8–M9 will address solo/combined festivals and reproduction. M6 ravager learning is **partial**: charge hunting and satiated hibernation show measurable progress in isolated tasks, but safe carcass relocation and stealth are not yet reliably learned. M5's neural cattle have demonstrated **limited call sensitivity in a controlled task**, but not reliable full-herd communication; and neural nomads have not mastered herding or defense; the effective scripted counterparts are reference policies, not scientific evidence of emergence. The simulation has not demonstrated long-run stable ecology.

## Documents

- [Complete implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [M2–M3 observations, training and baseline results](docs/M2_M3_LEARNING.md)
- [M4 farmer agriculture/refuge and training benchmarks](docs/M4_FARMER_LEARNING.md)
- [M5 nomad communication, cow-v2 policies and honest benchmarks](docs/M5_NOMADS_COMMUNICATION.md)
- [M5 controlled two-signal curriculum and counterfactual tests](docs/M5_SIGNAL_CURRICULUM.md)
- [M6 ravager physics, PPO, test results and remaining gaps](docs/M6_RAVAGER_INTELLIGENCE.md)
- [M7 scavenger nutrient recycling and held-out benchmarks](docs/M7_SCAVENGER_ECOLOGY.md)


## M7: Learned scavenging and manure competition

Scavengers now receive **71 local neural inputs**, including carrion scent, visible waste and nearby competitors. They consume finite waste and carcass meat, return only a fraction of consumed organic matter to soil through digestion, and compete with farmers who need manure to fertilize. Passive carcass decay also returns nutrients. No rule teleports food or forces scavenger movement. Existing species checkpoints are unchanged.

```powershell
qox train-scavenger --scenario scavenger_mixed --steps 8192 --output runs/scavenger.pt
qox benchmark-scavenger --scenario scavenger_competition --checkpoint runs/scavenger.pt --episodes 48 --report runs/scavenger-benchmark.json
qox viewer --mode learned-scavenger --checkpoint runs/scavenger.pt
```

In a held-out 48-episode evaluation, the selected two-stage policy consumed **5.608 waste units** versus 0.169 for random movement in the waste scenario, **8.247 carrion units** versus 0.333 for random in the carcass scenario, and **7.784 waste units** versus 0.068 for random during farmer competition (greedy evaluation). Farmers collected less manure when the trained scavenger consumed more, demonstrating a real tradeoff. A scripted scavenger remained stronger on reliable carrion consumption. These are short scenario results, not a proven stable long-term ecosystem. See [M7 research report](docs/M7_SCAVENGER_ECOLOGY.md).

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

## M6: Ravager hunting and energy management

Ravagers can sneak at half movement frequency, wind up and launch a high-speed
straight-line charge (with stamina and calorie costs), grab/drag/drop finite
carcasses, eat to gain satiety, and hibernate to reduce their metabolism.
Stealth requires cover and distance rather than granting invisibility. Training
and ablations use a new 82-feature policy without changing old cow, farmer, or
nomad model formats.

```powershell
qox train-ravager --scenario ravager_ambush --steps 32768 --output runs/ravager.pt
qox benchmark-ravager --scenario ravager_ambush --checkpoint runs/ravager.pt --episodes 48 --report runs/m6-benchmark.json
qox viewer --mode learned-ravager --checkpoint runs/ravager.pt
```

In held-out short scenarios, the learned greedy ravager killed prey in
**33.3%** of ambush trials versus **4.2%** for a random legal-action baseline;
a no-charge ablation produced **0%** kills. A separately trained satiated-rest
policy saved energy through hibernation. The learned policies have **not** yet
mastered safe carcass relocation or the complete multi-stage hunting cycle.
See [M6 experiment details and caveats](docs/M6_RAVAGER_INTELLIGENCE.md).

## M9: Population dynamics and long-horizon evaluation

**Opt-in, implemented:** mature adults can reproduce at an energy cost, creating juveniles with inherited group/home identity and permanent, unique IDs. Age-related mortality, species caps, and finite birth budgets prevent runaway simulation size; ravagers mature and reproduce more slowly. Pre-M9 checkpoint observation/action schemas are unchanged.

```bash
qox run --reproduction --ticks 800 --seed 42 --mode scripted --trace runs/m9.json
qox replay --trace runs/m9.json
qox population-study --ticks 1200 --seeds 5 --seed 260000 --mode scripted --festivals --report runs/m9.json
qox population-study --ticks 1200 --seeds 5 --seed 260000 --mode scripted --festivals --without-births --report runs/m9_without_births.json
```

The counterfactual keeps adult maturity and aging enabled in **both** conditions. `population-study --mode learned` loads frozen checkpoint(s) for selected species and keeps omitted species scripted; consult [`docs/M9_POPULATION_DYNAMICS.md`](docs/M9_POPULATION_DYNAMICS.md) for exact options and limitations. M9 has **not** demonstrated a permanently stable or independently learned ecosystem.


## Windows desktop app (M10)

QOX now has a **launcher with buttons**, built as a portable Windows `QOX.exe` using PyInstaller in GitHub Actions. It supports five **fresh untrained** neural networks, training one species using PPO, importing existing actor checkpoints, saving all five models to a `.qox` file, loading them back, and viewing all five policies in the 2D world.

- [Download a Windows build from GitHub Actions](https://github.com/AWDaley/QOX-Machine-Learning/actions/workflows/desktop-windows.yml) (choose a successful run and its `QOX-Windows-Desktop` artifact)
- [Windows setup, menu guide, and learning/save limitations](docs/DESKTOP_APP.md)
- On an installed Python source checkout: `python -m pip install -e ".[viewer,ml]"` then `qox-desktop`

**Important:** Fresh networks start with random weights. The viewer does not automatically train them; use **Train selected species** before saving the trained models. Saves contain neural weights, not the world state. Festival initiation uses a clearly labelled scripted helper until the standard five networks support festival decisions.


## M13 — Live analytics, neural inspection, and trace replay

The M13 observer adds **three viewer sidebar tabs**: `1` actors, `2` neural inspection,
`3` ecosystem graphs. Press `I` or `G` for quick neural/chart toggles, and use
`J/K` to scroll the selected actor's *pre-action* local observation features.
The neural panel shows the highest-probability **legal** actions, the requested
move, and whether the world confirmed an outcome. It does not reveal inaccessible
information or infer a neural model's intentions.

Press `B` to cycle species-specific event feeds or `Y` to restrict events to
one selected actor. Press `X` in the viewer to export bounded population and
resource telemetry as JSON/CSV and the confirmed event history as JSON under
`~/Documents/QOX/`.

```powershell
# Record deterministic world actions for later exploration
qox run --ticks 250 --seed 42 --mode scripted --reproduction --festivals --trace runs/explore.json
qox replay-viewer --trace runs/explore.json

# Alternatively record directly inside the viewer
qox viewer --seed 42 --mode scripted --record-trace runs/viewer-trace.json
```

Replay supports pause, stepping, 25-tick jumps, major-event navigation, and
clickable timeline seeking; each processed tick is verified against the recorded
hash. The desktop **File** menu adds trace recording and reopening without a
command line. Recorded traces are not world saves; `.qox` still stores neural
networks only. See [M13 observability and replay guide](docs/M13_OBSERVABILITY_AND_REPLAY.md).


## M15: Save and resume an entire ecosystem

In the Windows executable, press **F5** while watching a live world to quick-save it,
or **Ctrl+S** to choose a filename. World files have the `.qoxworld` extension.
From the desktop launcher select **File > Open saved world...** to load one.
Saved neural worlds must use the exact same five network weights, so keep a
separate `.qox` model set via **Save model set**. Resumed worlds start paused.

Headless scripted study:

```powershell
qox run --ticks 200 --mode scripted --reproduction --festivals --save-world runs/day1.qoxworld
qox resume-world --input runs/day1.qoxworld --ticks 200 --output runs/day2.qoxworld
qox viewer --load-world runs/day2.qoxworld
```

World saves include physical state, all events, births/juveniles, active calls/festivals,
carcasses, counters and both PRNG states. They are compressed, validated JSON ZIPs,
not model weights or pickle files. See [M15 World Save and Resume](docs/M15_WORLD_SAVES.md).

## M16a: learned cow–nomad herding under predator pressure

M16a extends the existing shared PPO training pipeline with migration, defense,
and mixed curricula, and permits a stochastic frozen learning partner. A new
paired `compare-herd` validation command checks **both** migration with call
ablations and cow health under a scripted predator. The command does **not**
replace or promote a trained checkpoint.

```powershell
qox train-herd --scenario nomad_mixed --partner-mode sample --phase alternate `
  --cow-checkpoint checkpoints/cow.pt --nomad-checkpoint checkpoints/nomad.pt `
  --cow-steps 6144 --nomad-steps 3072 --worlds 4 --rollout 384 `
  --max-steps 64 --output-dir runs/m16a

qox compare-herd --baseline-cow checkpoints/cow.pt --baseline-nomad checkpoints/nomad.pt `
  --candidate-cow runs/m16a/cow_round1.pt --candidate-nomad runs/m16a/nomad_round1.pt `
  --episodes 24 --seed 660000 --max-steps 64 --report runs/m16a-validation.json
```

Initial validation showed a tradeoff: the mixed trained pair improved mean cow
health in predator encounters, but decreased pasture arrivals and weakened call
sensitivity. A separate cow-only update did not convincingly improve arrivals.
Neither candidate passed the no-automatic-promotion diagnostic, so the original
M5c/M5b policy pair remains preferred. Details, controls, and research caveats:
[**M16a Joint Herd Reliability**](docs/M16A_JOINT_HERD_RELIABILITY.md).
