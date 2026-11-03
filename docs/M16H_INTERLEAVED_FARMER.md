# M16h — Interleaved on-policy farmer curricula

**Status:** Implemented, tested and evaluated against the archived M4 farmer.
The matched-budget study produced **0/6 development qualifiers**, so the
independent audit stayed closed. No model was installed automatically.

## Motivation

M16f found that staged agriculture → manure → refuge → mixed-farmstead PPO
lost food production relative to the frozen M4 farmer. M16g's offline
teacher KL reduced divergence on stored M4 states but failed to retain
farming in new worlds. The next hypothesis is that **frequent on-policy
revisits to the real farming worlds** will preserve useful behaviors more
effectively than forcing resemblance to a fixed set of old observations.

## Training method

Both arms start from the same immutable, trained M4 checkpoint.

| Setting | Sequential control (1 cycle) | Interleaved (4 cycles) |
| --- | --- | --- |
| Scenario schedule | agriculture → manure → refuge → mixed farmstead | (agriculture → manure → refuge → mixed farmstead) repeated four times |
| PPO decisions per scenario | 4,096 in one stage | 4 × 1,024 decisions |
| Total farmer decisions | 16,384 per training seed | 16,384 per training seed |
| PPO optimizer | Adam, carried across all stages | Same Adam, carried across all 16 segments |
| PPO epochs per update | 2 | 2 |
| Survival penalty | 1× farming, 3× refuge/mixed | Identical task-specific multiplier in every cycle |
| Training partners | Scripted nonfarmer agents | Same scripted partners |
| M4 teacher KL | Disabled (0) | Disabled (0) |

Every segment collects **fresh on-policy PPO trajectories** from its own
training worlds. Changing tasks never copies or forces an action into the
student policy. Model weights and Adam optimizer state continue from the
previous segment. Historical one-cycle training seed generation and stage
file names remain unchanged; the new four-cycle option has a separate
deterministic 3B+ seed namespace and descriptive `cycle_N_stage_N_*.pt`
checkpoints.

The per-cycle decision budget must be an exact multiple of the PPO rollout
and must cover at least one **complete episode per environment worker** for
the longest configured scenario. This prevents apparent rehearsal that
only sees the first few ticks of a 192-tick farmstead. A per-task stage
budget of 4,096 with four cycles, four workers and rollout 256 meets
this constraint.

## Predeclared controlled study

- Original checkpoint: M4 farmer trained for 32,768 decisions, archived
  SHA-256 `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd`.
- Two schedules: 1 cycle (sequential control) vs. 4 cycles.
- Matching seeds: 17, 37, 59; survival weight 3; 4,096 decisions per task;
  epochs 2; workers 4; rollout 256.
- Evaluation: 48 matched safe and 48 threatened farmsteads per candidate,
  192 ticks per episode. All six candidates use the same new development
  seed blocks 2,000,000 and 2,100,000.
- Blind audit reserved at 3,000,000 and 3,100,000. Only the **single**
  preselected development qualifier, if any, can read those seeds. At
  least three training seeds and 48 evaluation episodes per scenario are
  required for manual review. Audit cannot be used for tuning or retrying.
- Report paired seed-level food/manure, crop damage, refuge, survival,
  health and uncertainty. The unchanged M16e safety and food-gain gate
  applies to both schedules, and no candidate replaces a model set.

The new development and audit seed blocks are different from M16f's
1.6M/2.6M and M16g's 1.8M/2.8M blocks. The 3B+ on-policy training seeds
remain separate from all evaluation blocks and the older 2B+ control seeds.

### Command line

Supply your own archived M4 checkpoint; it is not in the Git repository.
Estimate cost without opening that file or creating output directories:

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16h_001 --weights 3 --stage-budgets 4096 --interleave-cycles 1 4 --epochs 2 --training-seeds 17 37 59 --development-seed 2000000 --audit-seed 3000000 --dry-run
```

Run the same predeclared comparison:

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16h_001 --weights 3 --stage-budgets 4096 --interleave-cycles 1 4 --epochs 2 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 2000000 --audit-seed 3000000
```

This estimates 60 separate stage/segment trainings, 98,304 total PPO
decisions and 1,728 development world episodes (including the two
action-ablated comparisons per candidate). Only if a development winner
qualifies are 480 audit/reference world episodes estimated.

For a cheap *nonqualifying* functional smoke test, use `--stage-budgets
256 --worlds 2 --rollout 64 --epochs 1 --episode-cap 18
--training-seeds 17 --development-episodes 2 --audit-episodes 2`.

## Interpretation

Even a positive screening result does not prove broad generalization or
demonstrate independent multi-species coordination: other species remain
scripted. Increasing the training budget relative to M16f and M16g can
itself improve learning; therefore the meaningful causal contrast in this
study is **four cycles versus one cycle at exactly the same larger budget**.
With only three training seeds, report individual results as well as means.
Do not open the blind audit for a failed development study, or silently
promote an underperforming network.

## First M4-baseline experiment — completed

The run used the verified M4 checkpoint SHA-256 above, all predeclared
training and evaluation parameters, 60 stages/segments and 98,304 PPO
decisions. The M4 baseline averaged **3.3314 deposited food** and **2.2815
manure applied** in the 48 safe development farmsteads. Differences below
are candidate *minus the same M4 baseline*, with threatened survival in
percentage points and threatened final-health changes in health units.

| Cycles | Train seed | Safe food Δ | Safe manure Δ | Threat survival Δ | Threat health Δ | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 17 | −0.3309 | −2.2815 | −4.17 pp | −27.2292 | Rejected |
| 1 | 37 | −0.4242 | −2.2815 | 0 pp | −3.5000 | Rejected |
| 1 | 59 | −1.2746 | −1.7213 | 0 pp | −1.2500 | Rejected |
| 4 | 17 | −0.3560 | −2.2815 | 0 pp | +0.7500 | Rejected |
| 4 | 37 | −0.3560 | −2.2815 | 0 pp | −2.0000 | Rejected |
| 4 | 59 | −0.3560 | −2.2815 | 0 pp | +0.7500 | Rejected |

At these seeds, four-cycle training was more consistent than sequential
training on food deposition and avoided the severe threatened-health
failure of the sequential seed-17 candidate. Mean differences across
the three *training seeds* were:

| Schedule | Mean safe food Δ vs. M4 | Mean safe manure Δ vs. M4 | Mean threatened health Δ vs. M4 |
| --- | ---: | ---: | ---: |
| Sequential (1 cycle) | −0.6766 | −2.0948 | −10.6597 |
| Interleaved (4 cycles) | −0.3560 | −2.2815 | −0.1667 |

These are descriptive summaries of only three independently trained
networks, not statistical proof that interleaving improves survival. Even
the better food/health averages remained below the untouched M4 reference.
For example, the four-cycle seed-17 policy's paired 95% bootstrap interval
for safe food change was **[−0.7071, +0.0093]**, while its manure change
was **[−2.5413, −2.0328]**. Its mean safe manure application was zero.

The manure failure across all three interleaved policies is particularly
informative. In the current environment reward implementation, fertilizer
events earn only **0.025 per unit**, compared with **0.55 per unit** for
food deposits; collecting waste itself receives no immediate event bonus.
This is an *inference*, not a proven cause, but repeated exposure cannot
guarantee learning of a weakly rewarded multi-step behavior. An appropriate
next controlled test would strengthen manure **training-only** credit
assignment and independently evaluate useful food, manure and survival,
without modifying finite world resources or relaxing the acceptance gate.

**Decision: 0/6 development qualifiers, no selected candidate, no audit,
no promotion.** All candidate checkpoints and per-seed comparisons were
retained outside Git; audit seed blocks 3.0M/3.1M remain unused. The study
provides useful negative evidence, not an independently validated learned
farmer improvement.
