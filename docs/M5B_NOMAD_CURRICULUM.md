# M5b — Learned nomad cattle calls, transfer, and counterfactuals

**Status:** M5b research implementation and exploratory evaluation. The nomad now solves an *isolated* push/pull task with a small neural policy, but full cooperation between independently learned nomads and cows remains an open problem.

## What changed

- Added the **`nomad_signal`** curriculum: one stationary nomad, one scripted but **locally responsive** cow, an assigned destination, and a balanced friendly-versus-scary signal choice across four compass orientations.
- A nomad may make **one physically emitted call** per short training episode; it can also wait/rest. Movement, milking and combat are masked for this *curriculum only*, and become available again in the normal migration world. The call is a world sound with energy costs and range/decay; it **never directly pushes or pulls** the cow.
- A new **version-2 nomad observation** adds three goal-relative-to-*visible*-herd features: desired displacement X/Y and signed alignment between caller position and intended travel direction. These are derived from the assigned goal and locally visible cows, rather than a global population map or a hidden correct-action label. The alignment is an **engineered geometric clue**: this experiment evaluates learned call selection, not whether the network independently discovers the geometry of shepherding.
- Added optional **PPO checkpoint warm starts** so a one-cow signaling policy can be fine-tuned in existing moving three-cow worlds without changing its network/action vocabulary.
- Added **paired no-call and reversed-call evaluations**, per-call-type metrics, and a way to benchmark nomads against frozen, call-aware **neural-network cows**, not just scripted responders.
- Version-1 nomad observation/checkpoints (72 features), legacy cow checkpoints (60/66 features), and ordinary game-world action rules remain compatible. The v2 nomad has 75 inputs; the v1 nomad still has 72.

## Experiment A — isolated learned call selection

On each episode, the world chooses a cardinal caller direction and a target on the caller's side (requiring **friendly**) or on the opposite side (requiring **scary**). The cow cannot see a destination marker; the nomad knows the local assigned goal. Only the call-sensing scripted cow moves in response; rewards depend on **actual displacement and target arrival**, not on choosing an explicitly marked "correct" action.

Training: PPO, seed **71**, **32,768 nomad decisions**, 4 parallel worlds, 8 ticks per episode, 2×64 MLP plus separate actor/value heads; checkpoint `nomad-signal-v2-32k-seed71.pt`.

Evaluation: **96 held-out episodes** with seeds **150000–150095**, frozen greedy nomad actions, paired starting worlds.

| Caller policy / ablation | Cow arrival rate | Friendly tasks | Scary tasks |
|---|---:|---:|---:|
| Learned v2, correct calls | **100% (96/96)** | 100% | 100% |
| Same policy, calls disabled | **0%** | 0% | 0% |
| Same policy, emitted calls reversed | **0%** | 0% | 0% |
| Random allowed actions | 57.3% | 54.0% | 60.9% |
| Scripted nomad baseline | 100% | 100% | 100% |
| Scripted, no calls or reversed calls | 0% | 0% | 0% |

Two additional **16,384-decision** training runs (seeds **72** and **73**) each achieved **24/24** on a separate held-out 24-episode check. These are supportive small replications, not a confidence interval or evidence of reliability across arbitrary world designs.

Ablation is **causal within this controlled scenario**: no calls and reversed calls use the same map and goal, with only the signal changed. The goal-alignment input and simple one-shot environment deliberately make this easier than realistic multi-actor cooperation.

## Experiment B — transfer to moving three-cow herding

Evaluate the *unchanged* one-cow v2 nomad policy in `nomad_herd`. Movement/dash and repeated signaling are restored; three scripted cows respond to actual local sound and a pasture target is varied across east/west and nomad-leading/trailing layouts.

Evaluation: **24 held-out episodes**, seeds **220000–220023**, 96 ticks each; mean cows ever reaching pasture, out of 3 per episode.

| Nomad controller | Mean cows reaching pasture | Mean emitted calls |
|---|---:|---:|
| V2 policy transferred directly (greedy) | **2.96 / 3** | 21.83 |
| Same policy, calls disabled | **0 / 3** | 0 |
| Scripted herder | 2.83 / 3 | 13.38 |
| Scripted herder, calls disabled | 0 / 3 | 0 |
| Random nomad | 1.71 / 3 | 12.46 |
| V2 policy fine-tuned another 16,384 decisions (seed 172) | 2.46 / 3 | 6.63 |

**Interpretation:** the original signal policy transfers surprisingly well to three *scripted* cattle without further training. The 16k moving-herd fine-tune cut call use but also decreased pasture arrivals; it is **not** the preferred checkpoint. A stable migration strategy might need a multi-objective reward that values arrival and signaling efficiency, plus more varied situations.

## Experiment C — replacing scripted cows with learned cows

Use the **frozen M5 `cow_signal_65k.pt` 66-feature cow policy**, with deterministic argmax actions, rather than scripted cow behavior. Keep the same 12 migration seeds (230000–230011) and three-cow task for each nomad comparison.

| Nomad | Mean cows arriving (out of 3) |
|---|---:|
| V2 learned nomad | **0.58** |
| Scripted nomad | 0.75 |
| Random nomad | 0.25 |

**Main limitation:** two independently trained neural policies do **not yet** exhibit effective full-herd cooperation. The cow was trained on a very different hidden-waypoint signal task, and it appears not to generalize reliably to this pasture/migration environment. Repeating with more seeds, training together or with alternating frozen counterparts, and training on pasture-based goals is the next substantive research work. These results are *not* evidence of predator-defense improvement.

## Reproduce

Install with `python -m pip install -e ".[dev,ml]"` on a Python 3.11+ CPU machine. M5b does not require Docker or a database.

```powershell
# Stage 1: train a nomad to choose a call for one cow
qox train-nomad --scenario nomad_signal --steps 32768 --worlds 4 --rollout 256 --max-steps 8 --seed 71 --output runs/nomad-signal.pt

# Three-way counterfactual benchmark (plus random and scripted controls)
qox benchmark-nomad --scenario nomad_signal --checkpoint runs/nomad-signal.pt --episodes 96 --seed 150000 --max-steps 8 --report runs/signal-eval.json

# Stage 2: continue the v2 network with movement and three cows
qox train-nomad --scenario nomad_herd --nomad-v2 --init-checkpoint runs/nomad-signal.pt --steps 16384 --worlds 4 --rollout 256 --seed 172 --output runs/nomad-migrating.pt

# Generalization of original checkpoint to moving pasture, without re-training
qox benchmark-nomad --scenario nomad_herd --checkpoint runs/nomad-signal.pt --episodes 24 --seed 220000 --max-steps 96

# Compare with learned cattle rather than transparent scripted responders
qox eval-nomad --scenario nomad_herd --checkpoint runs/nomad-signal.pt --cow-checkpoint checkpoints/cow_signal_65k.pt --mode policy_greedy --episodes 12 --seed 230000
```

The included `runs/m5b_research_results.json` in the downloadable research package records **per-episode outputs** (seed, call counts, success, returns), and the generated checkpoint metadata records each training configuration. Intermediate training checkpoints and research binaries are deliberately **not** committed to GitHub source control.

## Next research gates

1. Train cows on actual pastures/grass **with** frozen learned nomads, retaining no-call/reversed-call ablations.
2. Alternate which species policy is trainable, keeping the other fixed; test unseen relative positions and other pasture layouts.
3. Reduce call spam while maintaining arrival performance; measure **successful moves per calorie/stamina spent**.
4. Only after stable learned-cow herding, introduce ravager attacks and evaluate the tradeoff between herd migration and defense.

The test suite covers curriculum action masking, physical call propagation, deterministic evaluation, versioned checkpoint loading, safe warm starts, and legacy compatibility.
