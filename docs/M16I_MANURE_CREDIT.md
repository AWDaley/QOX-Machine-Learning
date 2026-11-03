# M16i — Finite-manure training credit assignment

**Status:** Implemented and evaluated against the archived M4 baseline.
**Zero of nine** development candidates passed the unchanged farming/safety
gate. The blind audit was never opened; no policy was installed or promoted.

## Question and causal intervention

M16h's three four-cycle farmer policies applied **zero manure** in held-out
safe farmsteads, although the original M4 farmer applied 2.2815 units per
episode. In the unchanged environment reward function, useful fertilization
receives 0.025 reward/unit, deposits earn 0.55 reward/unit, and collection
earns no event bonus. This suggests, but does not prove, a credit-assignment
problem across the collection → carrying → application sequence.

M16i adds one optional PPO reward parameter, `manure_credit`, initially zero.
Only executed `collect_waste` and `fertilize` events *owned by the controlled
farmer* can earn an additional reward:

```
training_bonus = manure_credit * (usefully_applied + 0.2 * collected)
```

Physical collection consumes finite waste from a tile; a fertilizer event
consumes carried manure only when it actually increases depleted soil's
fertility. The small 0.2× collection contribution helps assign credit to the
first step while emphasizing productive application. Wasted attempts, acting
on full-fertility soil, other actors' actions, and repeated collection from
an emptied tile receive no credit. Parameters are limited to 0–2 inclusive;
the 0 default preserves legacy PPO behavior. The trainer logs actual collected
and applied quantities and the total added bonus for each stage.

**Nothing in world physics, evaluation reward, action masks, observation
features, installed policies, or checkpoint action schema is modified.** All
previous farmstead gates remain intact. More rewarded collection without
fertilization or food production is not considered success.

## Predeclared M4-controlled study

| Parameter | Fixed plan |
| --- | --- |
| Baseline | Original M4 32,768-step farmer checkpoint, SHA-256 `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd` |
| Hypothesis arms | `manure_credit = 0` (M16h four-cycle control), `0.3`, `1.0` |
| Initial weights | Exact same read-only M4 checkpoint in every run |
| Training seeds | 17, 37, 59 per arm; identical stage/world seeds |
| Curriculum | Four interleaved cycles × agriculture, manure, refuge, farmstead_mixed |
| Training budget | 4,096 decisions **per task**, 16,384 per seed and arm; four worlds; rollout 256; 2 PPO epochs |
| Safety weighting | 1× agriculture/manure, 3× refuge/mixed |
| Teacher rehearsal | Disabled (`rehearsal_kl = 0`) |
| Development cases | 48 matched safe farmsteads at seed 2,200,000; 48 threatened at 2,300,000, each 192 ticks |
| Audit cases | Reserved 48 safe at 3,200,000 and 48 threatened at 3,300,000; unopened unless the unchanged development gate passes |
| Candidate gate | Existing M16e/M16h food, manure, survival, health and crop-damage checks; ≥3 training seeds and 48 episodes per scenario |

Three arms × three seeds × 16 training segments = **144 stage/segment
trainings**, **147,456 PPO decisions**, **2,592 development world episodes**
(including paired no-fertilize and no-refuge ablations). Selection may open
the audit **once** for one development-qualified candidate; the audit is not
reused for tuning. Audit and development seeds do not overlap prior M16h's
2.0M/3.0M blocks, M16g's 1.8M/2.8M blocks or the separate 3B+ training
namespace.

The 0-credit control should match archived M16h four-cycle training policy
weights, notwithstanding new metadata in its checkpoint. Where available,
compare the actual parameter tensors to detect unintended control changes.

### Run with the archived M4 checkpoint

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16i_001 --weights 3 --stage-budgets 4096 --interleave-cycles 4 --manure-credits 0 0.3 1.0 --epochs 2 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 2200000 --audit-seed 3200000 --dry-run
```

Remove `--dry-run` to train. This does not overwrite earlier outputs or any
model set. For a quick nonqualifying smoke test, use `--stage-budgets 256
--worlds 2 --rollout 64 --epochs 1 --episode-cap 18 --training-seeds 17
--development-episodes 2 --audit-episodes 2` with distinct output directory.

## Interpretation rules

- Compare *actual evaluation manure application*, food deposition, safe and
  threatened health/survival, crop damage, plus ablated action performance;
  do not interpret higher PPO return from shaping as ecological improvement.
- Look for true `fertilize` events rather than just collected manure. An actor
  that accumulates manure without applying it has not solved the task.
- Report every seed, including zero effects and negative results. Three
  independently trained seeds are a small descriptive sample, not proof.
- If no candidate clears all development checks, keep the blind audit closed,
  publish the negative result and do not install a policy.

## First M4-baseline study

The study completed all **144 stage/segment trainings**, **147,456 PPO
decisions** and **2,592 development world episodes** with the exact
predeclared settings. The same original M4 baseline, SHA-256
`5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd`,
was read-only throughout. For the new safe development seed block, it
averaged **3.6064 food deposited** and **2.2817 useful manure applied**.
All nine candidates and their ablations used the same 48 safe and 48
threatened development seeds.

Candidate changes below are relative to that same M4 baseline. Threatened
survival was 100% for all nine candidates and the baseline; threatened-health
changes are in health units. The gate is the *full* original gate, not just
food or manure.

| Bonus | Seed | Safe food Δ | Safe manure Δ | Threat health Δ | Development gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 17 | −0.6311 | −2.2817 | +0.1042 | Rejected |
| 0 | 37 | −0.6311 | −2.2817 | −2.2917 | Rejected |
| 0 | 59 | −0.6311 | −2.2817 | +0.1042 | Rejected |
| 0.3 | 17 | −0.6731 | **+0.9158** | −0.5000 | Rejected |
| 0.3 | 37 | −1.6898 | −2.2817 | −0.6458 | Rejected |
| 0.3 | 59 | −0.6731 | −1.1630 | +1.2500 | Rejected |
| 1.0 | 17 | −1.6898 | −2.2817 | +0.7500 | Rejected |
| 1.0 | 37 | −1.6898 | −2.2817 | −0.3958 | Rejected |
| 1.0 | 59 | −1.4018 | −1.5972 | +0.7500 | Rejected |

Across the three independently trained seeds in each arm:

| Bonus | Mean safe food | Mean safe manure | Mean manure applied **during training** per seed |
| ---: | ---: | ---: | ---: |
| 0 | 2.9753 | 0.0000 | 139.5147 |
| 0.3 | 2.5944 | 1.4387 | 216.8113 |
| 1.0 | 2.0127 | 0.2282 | 258.6830 |

These are descriptive three-network averages; the training totals aggregate
all 16 segments, whereas the development metrics are per held-out episode.
Larger reward strengths generated more real fertilizer events in training,
but **did not transfer reliably** to novel farmsteads. Stronger shaping also
coincided with lower held-out food deposits. These observations do not prove
the mechanism of failure; they rule out the simplest hypothesis that just
raising finite-manure event rewards fixes the M16h result.

The seed-17 `0.3` candidate achieved the only positive safe manure gain:
**+0.9158 units/episode**, with a paired 95% bootstrap interval of
**[+0.1333, +1.7750]**. Its food-storage difference was **−0.6731**,
with paired interval **[−1.0565, −0.3620]**. Another `0.3` seed still applied
zero manure. This is exactly why single-seed success cannot determine
model admission.

### Reproducibility and decision

- The 0-credit policy parameters from **all three** training seeds were
  bit-identical to the corresponding archived M16h four-cycle control
  tensors; their new checkpoint metadata and development seeds differ.
- Verified baseline hash was unchanged before/after the sweep; 144 stage
  checkpoints and all nine seed-level development reports were retained.
- `development_qualifiers = 0`; `selected_before_audit = null`;
  `audit = null`. The reserved 3.2M/3.3M audit cases were not evaluated,
  and no recommendation or model-set promotion occurred.
- The next justified investigation is **joint credit and retention**, e.g.
  actor-verified task completion with constrained food-production
  preservation, or a memory/action-context intervention. Keep the existing
  holdout, safety and non-promotion rules; do not simply sweep more reward
  strengths on the same development seed block.

**Decision: 0/9 development qualifiers; independent audit remains closed;
original M4 model retained.**
