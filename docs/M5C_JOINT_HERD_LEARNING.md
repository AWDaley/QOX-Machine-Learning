# M5c — alternating shared-policy training with learned herd partners

**Status:** Implemented and evaluated. This is a constrained research result, not a claim that complete herding or multi-agent cooperation is solved.

## Motivation

M5b's learned nomad selected push/pull calls reliably with **scripted cattle**, but only 0.58 of 3 independently learned cows reached pasture in a 12-seed check. The learned M5 cow was trained to follow a hidden one-cow waypoint signal, not to migrate across a larger pasture map. M5c changes **what experience cows are trained on**, not their 66-feature observation schema or the underlying meaning of calls.

## Implementation

`src/qox/learning/joint_ppo.py` adds:

- A `train-herd` CLI mode that loads the existing cow v2 (66-feature) and nomad v1/v2 (72/75-feature) checkpoints. It rejects the 60-feature legacy cow for joint herding, without altering legacy checkpoint support elsewhere.
- A **cow training phase** in which **all three cows** act from one shared trainable PPO policy. The nomad acts from its **frozen neural-network policy**, not a scripted cattle call routine. Cows do not see the global pasture goal in their observations.
- A **nomad training phase** in which one nomad is trained while the three cows use their **frozen newly trained policy**. The engine still decides simultaneous movement, sound propagation, food and reward accounting.
- Training `--phase alternate`, `--phase cow`, or `--phase nomad`. A cow-only phase is useful when a known competent nomad should remain fixed. Each stage saves an individually loadable checkpoint with the existing model schema, raw PPO settings and training-decision count.
- Paired `benchmark-herd` with a learned pair, disabled calls, reversed calls, scripted nomad with learned cows, random nomad with learned cows, and a learned nomad with scripted cows. It records raw outcomes for each seed.

Training uses a no-predator `nomad_herd` curriculum. Every cow action is an independent neural policy choice using only its own local information. The physics still implements calls as sound events, not enforced cow movement. This milestone does **not** train combat or ravager defense.

## PPO and accounting

Multiple cattle share parameters, **not hidden state**. Their three trajectories are rolled into the PPO loss. Episodes reset as a whole, on a deterministic seed schedule, to avoid crossing episode boundaries with stale partner state. Model parameters are updated **only** for the active species; the partner's weights are frozen. Reward comes from true progress to pasture and finite world outcomes, not from simply issuing calls.

Training steps count individual **species decisions**: 24,576 cow decisions across 3 cows are not 24,576 world ticks. Stages save separate checkpoints so an unsuccessful nomad fine-tune does not overwrite a successful cow policy.

## Reproducible experiments

All data below come from local frozen-policy evaluation. Training seeds differ from validation and held-out test seeds; choose candidate based on validation only, then measure on the separate final test range. All cows start from the M5 call-aware `cow_signal_65k.pt` checkpoint, and nomads start from the M5b `nomad_signal` v2 policy.

### Training candidates

- **Seed 407**: 24,576 cow decisions (then 8,448 nomad decisions). Validation seed range 390000–390023: with the *original frozen nomad*, **1.58 of 3** cows reach pasture. With calls disabled **1.17 of 3** arrive. Training the nomad afterward reduced the paired policy's result on the first 12 validation seeds from **1.67 to 1.33**.
- **Seed 409**: independent **24,576 cow decisions**, keeping original M5b nomad policy frozen. Same 24 validation seeds: **2.08 of 3** cows reach pasture; no-call **1.50**; reversed calls **0.25**. This cow checkpoint was selected for final evaluation.
- **Nomad seed 511**, 8,448 decisions, warm-starting original M5b nomad while using seed-409 cows: validation **2.04 of 3** versus **2.08 of 3** for unmodified nomad. Fine-tuning reduced mean calls from **21.25 to 15.58** per episode, at slightly worse arrivals. Keep original nomad for best arrivals.

### Final held-out benchmark (not used to select the candidate)

**48 episodes**; seeds **460000–460047**; 96 ticks each; greedy frozen policies. Outcome is mean cows **ever reaching** the pasture, maximum 3. All alternatives use the same seeded worlds.

| Controller | Mean cows arriving / 3 |
|---|---:|
| Original M5b learned nomad + original M5 cow | **0.65** |
| M5b nomad + M5c seed-409 cow (selected) | **1.96** |
| Same learned pair, nomad calls disabled | **1.50** |
| Same learned pair, friendly/scary calls reversed before emission | **0.25** |
| Scripted nomad + M5c learned cows | **2.77** |
| Random nomad + M5c learned cows | **1.54** |

Paired effect on 48 worlds: selected pair minus original pair **+1.31 cows/episode** (95% percentile bootstrap interval approximately **+0.81 to +1.77**). Correct calls minus no calls **+0.46** (bootstrap interval approximately **+0.10 to +0.81**). Selected learned nomad minus random nomad **+0.42** (bootstrap interval approximately **−0.04 to +0.88**, includes zero). Bootstrap samples are resampled *worlds*, not ticks or cattle, with replacement. These are exploratory uncertainty summaries and are not a claim of global ecosystem reliability.

**Interpretation:** cow learning with an actual learned nomad produces a substantial improvement over independently trained M5 models. But the no-call control retains 1.50/3 arrivals, so navigation and the nomad's positioning can contribute without signal understanding. Reversed calls sharply degrade arrivals, indicating sensitivity to call identity/direction. The scripted herder still wins and the difference versus a random nomad is uncertain with 48 worlds. Independent learned herding is **partially working**, not solved.

## Reproduce with CPU

Install `python -m pip install -e ".[dev,ml]"` and locate the two input checkpoints used for the experiments (not committed to the repository because of their binary size). Their paths below are examples from the downloadable M5b package.

```powershell
# Train cows against a frozen learned nomad; preserve the original nomad weights
qox train-herd --phase cow --cow-checkpoint checkpoints/cow_signal_65k.pt --nomad-checkpoint checkpoints/nomad_signal_v2_32k.pt --cow-steps 24576 --nomad-steps 8192 --worlds 4 --rollout 384 --max-steps 96 --epochs 3 --seed 409 --output-dir runs/m5c-cows

# Optional sequential fine tuning, train both species but keep every stage checkpoint
qox train-herd --phase alternate --cow-checkpoint checkpoints/cow_signal_65k.pt --nomad-checkpoint checkpoints/nomad_signal_v2_32k.pt --cow-steps 24576 --nomad-steps 8192 --seed 407 --output-dir runs/m5c-alternating

# Frozen-policy controls on different seeds
qox benchmark-herd --cow-checkpoint runs/m5c-cows/cow_round1.pt --nomad-checkpoint checkpoints/nomad_signal_v2_32k.pt --episodes 48 --seed 460000 --report runs/m5c-held-out.json
```

**Checkpoint compatibility:** Both outputs remain loadable by existing `load_cow` / `load_nomad` and the learned-herd viewer; existing 60-feature cow networks and 72-feature nomad networks continue to load for their respective older tasks.

## Caveats and next work

- No learned herding guarantee: repeat across more map distributions and more random training seeds before promoting a policy as broadly robust.
- The stationary one-cow signal task and three-cow migration task have different reward geometry; a real long-run multi-agent environment may need memory, visual occlusion, social incentives and a curriculum with more varied herd placements.
- No-call progress is nonzero, so isolate *signal-specific* behavior from general navigation and proximity to nomad; continue no-call and reversed-call ablations on each iteration.
- Predator defense, replenishment, long-range herd coordination, festivals and reproduction remain future milestones.
- Neither actor is given a secret 'correct call' action label. Feature engineering gives the nomad a relative goal; cow observations do not include global destination coordinates.
