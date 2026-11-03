# M16i — Finite-manure training credit assignment

**Status:** Implemented. The predeclared M4-baseline study below measures
whether stronger *training-only* credit can recover physical manure use without
degrading food storage, survival, health or crop damage. No automatic promotion.

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

*Results and reproducibility checks will be recorded here once the controlled
run completes.*
