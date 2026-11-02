# M5 — Nomad calls, mobile combat, and learned herding experiments

**Implementation status:** mechanics and end-to-end training are implemented and tested. **Research result:** learning is still weak. The experiments below do *not* establish that learned nomads reliably herd cattle, that learned cows understand calls, or that learned nomads consistently defend the herd.

## World mechanics

- **Friendly cattle call** emits a spatial blue sound signal; **scary cattle call** emits a separate orange signal. Sound radius, lifetime, stamina cost, and cooldown are configurable. Calls do not reposition cattle, change cattle rewards, or force any movement.
- Sound sources are local: cattle can observe **two distinct relative directions and strengths** for the loudest friendly and scary calls they hear. Signals attenuate by distance and age. Two overlapping signal types remain distinguishable. No perfect map knowledge is transmitted.
- **Nomad dash** moves two cardinal tiles in one tick if the intermediate tile is clear, passable, and unoccupied. It consumes more stamina and energy than ordinary movement. The destination is still resolved in the simultaneous movement competition.
- **Nomad attack** can damage an adjacent ravager for configurable damage/stamina costs. Nomads cannot attack cows through this action. Ravager attacks remain stronger and dangerous to lone nomads.
- A hand-written **reference cow policy** can react to local friendly/scary sounds. This reference is explicitly scripted and never presented as learned cow intelligence. It is used to verify the communication physics before training.
- Calls and dash/attack events are included in world event records and hashed deterministic action traces. A viewer shows call radii as blue/orange rings.

## Observation and checkpoint compatibility

| Role | Input schema | Notes |
|---|---:|---|
| Cow v1 (M3) | 60 | Unchanged; older cow checkpoints still load |
| Cow v2 (M5 opt-in) | 66 | Adds friendly `dx,dy,strength` and scary `dx,dy,strength` |
| Farmer (M4) | 76 | Unchanged; older farmer checkpoints still load |
| Nomad (M5) | 72 | Adds nearby herd summary, known pasture objective, nearby threat, call readiness |

Nomads only know the *assigned pasture marker* in herding benchmark worlds; cows do not receive its coordinates. Perception of enemies, other cows and auditory calls remains local. A checkpoint stores its exact feature names/version and action vocabulary, and incompatible checkpoints fail at load time. The default `qox inspect --species cow` retains the original 60 inputs; use `--cow-calls` for M5's 66-input version.

## Training and evaluation

M5 includes two independent masked-PPO paths:

1. **Nomad policy:** one learned nomad, other actors scripted, environments `nomad_herd`, `nomad_defend` or `nomad_mixed`. The herd curriculum includes left/right pastures, a nomad initially ahead/behind the cows, and small vertical offsets. Group reward comes from actual cow displacement toward pasture, first arrival, avoiding cow damage/deaths and final cow condition—not from pressing call buttons.
2. **Cow v2 policy:** one learned cow, a scripted nomad and other scripted cattle. Extra audio features are enabled only for the v2 checkpoint. A cow's environmental shaping uses actual travel to pasture in addition to usable grazing and survival rewards. The no-call ablation blocks scripted nomad calls.

The trainer is on-policy PPO with shared species weights, legal-action masking and seeded headless training. **A nomad's frozen PPO distribution is stochastic**, so evaluation includes `policy` (seeded sample) and `policy_greedy` (argmax); a trained neural network does not have to act greedily. Its sampled actions use a random stream separate from world physics, which preserves deterministic trace replay.

### Local baseline results, held-out seeds

**Nomad migration** — 16,384 training decisions, scenario `nomad_herd`, 80 ticks per episode, 24 held-out seeds starting at 170000:

| Nomad choice | Mean cows reaching pasture (out of 3) | Mean calls per episode |
|---|---:|---:|
| Learned PPO, stochastic | **1.71** | 9.83 |
| Learned PPO, greedy | 0.00 | 0.00 |
| Same learned policy, calls disabled | 0.00 | 0.00 |
| Random legal actions | 1.71 | 11.21 |
| Scripted herder | 2.79 | 10.67 |
| Scripted herder, calls disabled | 0.00 | 0.00 |

Interpretation: communication is physically useful in this scenario, and the sampled learned nomad benefits from the ability to call, **but is not better than random actions**. Deterministic action selection fails outright.

**Cow v2 call inputs** — 8,192 training decisions, scenario `nomad_herd`, 80 ticks, 16 held-out seeds starting at 175000:

| Cow choice | Pasture-arrival rate | Mean grazing biomass |
|---|---:|---:|
| Learned call-aware cow | **50%** | 4.78 |
| Same policy, calls disabled | **50%** | 4.44 |
| Random actions | 0% | 3.48 |
| Scripted cow listening to calls | 100% | 50.84 |
| Scripted cow, calls disabled | 0% | 0.62 |

Interpretation: the neural cow improves on random navigation in this curriculum, **but the unchanged arrival rate without calls does not demonstrate learned understanding of nomad signals**. Direction memorization is a plausible explanation. More varied terrain, changed starting placements and stronger causal tests are needed.

**Nomad defense** — 8,192 decisions training on `nomad_defend`, 64 ticks, 12 held-out seeds starting at 180000:

| Nomad choice | Mean surviving cows (out of 3) |
|---|---:|
| Learned PPO, stochastic | 0.83 |
| Random legal actions | 0.58 |
| Scripted defender | 2.50 |

The learned defender is still well below the scripted baseline. These are small exploratory experiments with few maps, not population-wide scientific claims. No joint co-training or stable ecology has been demonstrated.

## Run the experiment

Install: `python -m pip install -e ".[dev,ml,viewer]"`

```powershell
# Train and benchmark the nomad policy
qox train-nomad --scenario nomad_herd --steps 16384 --seed 61 --output runs/nomad.pt
qox benchmark-nomad --checkpoint runs/nomad.pt --scenario nomad_herd --episodes 24 --seed 170000 --report runs/nomad-benchmark.json
qox benchmark-nomad --checkpoint runs/nomad.pt --scenario nomad_defend --episodes 12 --seed 180000

# Train the opt-in v2 cow policy. The original v1 M3 cow trainer is unchanged.
qox train-cow --cow-calls --scenario nomad_herd --steps 8192 --seed 67 --output runs/cow-v2.pt
qox benchmark-cow --checkpoint runs/cow-v2.pt --scenario nomad_herd --episodes 16 --seed 175000

# Observe two learned species sharing one larger world (others are scripted).
qox viewer --mode learned-herd --checkpoint runs/nomad.pt --cow-checkpoint runs/cow-v2.pt
qox run --ticks 200 --mode learned-herd --checkpoint runs/nomad.pt --cow-checkpoint runs/cow-v2.pt --trace runs/joint.json
qox replay --trace runs/joint.json

# See the M5 local sound features and actions.
qox inspect --species nomad
qox inspect --species cow --cow-calls
```

## Validation / remaining work

The new physics is covered by tests for call radius/decay, type separation, cooldown and resource expenditure, no forced cow movement, dash path collision and stamina use, nomad-vs-ravager attacks, herd seed variability, no-call ablations, checkpoint schemas, two learned policies running together, and replay. The optional PettingZoo API contract is run in GitHub CI where dependencies are installed.

See also [`M5_SIGNAL_CURRICULUM.md`](M5_SIGNAL_CURRICULUM.md) for a follow-up controlled two-signal learning experiment, held-out counterfactual evaluation and its limitations.

**Next research step (before treating full herding as learned):** expand training seeds and cover/obstacle layouts, randomize pasture and caller assignments more aggressively, compare v2 cows to v1 in identical worlds, test both-signal confusion, vary nomad/cow population and defend scenarios, improve reward credit assignment, evaluate multi-species co-training against unseen opponent checkpoints, and require significant results across multiple train seeds. None of these research milestones should be inferred from the current scripted results.

M6 will implement ravager sneak/charge/drag/hibernate and is outside the M5 scope.
