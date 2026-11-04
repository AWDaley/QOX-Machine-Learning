# M16j — On-policy food-skill retention during manure learning

**Status:** Implemented; controlled nine-network M4 study completed.
**0/9 development qualifiers.** The reserved blind audit was not opened;
no policy was promoted, and the original M4 farmer remains the default.

## Motivation and intervention

M16i's verified finite-manure reward increased fertilizer usage *during
training* as the reward weight rose, yet food deposition deteriorated and
manure use did not transfer reliably to new farmsteads. The 0.3-credit arm
was the least damaging compromise: one of three seeds surpassed M4 on
manure use, but all seeds underperformed M4 on food.

M16g previously tried KL rehearsal against a **fixed**, limited set of M4
training observations. Its strong replay teacher-match did not translate
into stronger farmstead behavior. M16j isolates a different hypothesis:
**would a food-task safeguard computed on the student's *current on-policy
states* retain the crucial planting/harvesting/deposit routine?**

For each PPO minibatch, load the original frozen M4 farmer as a teacher and
query it on the exact learner observations and legal-action masks already
collected from QOX. Apply teacher-to-student masked KL only if the teacher's
most probable legal action is:

1. `plant`, `harvest`, or `deposit`; or
2. `move` while the learner's local observation shows it is carrying at
   least 2.5 food units (normalized feature `carried_food >= 0.25`).

This also safeguards some food-delivery navigation, not just final deposit
actions. The KL term is averaged **over the entire PPO minibatch** (non-food
states contribute zero), so stronger state coverage gives a stronger regularizer.
M4 is a frozen training reference, not an action controller. The safeguard
cannot command an actor, change its observation, teleport resources, or
alter the environment's evaluation reward. It also does not change the
M16i reward or relax any admission check. A KL weight of zero follows the
previous training path exactly, without loading a teacher.

The trainer records the mean KL penalty and fraction of sampled states
covered by this safeguard; real manure collection/application counts remain
separately recorded. These diagnostic metrics are *not* acceptance criteria.

## Predeclared matched-budget study

| Setting | Protocol |
| --- | --- |
| Baseline | Unchanged M4 farmer, 32,768 steps; SHA-256 `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd` |
| Arms | Food-anchor KL `0` (M16i control), `0.5`, `2.0` |
| Manure reward | `manure_credit = 0.3` in **all** arms; collection earns one-fifth of useful application credit |
| Training seeds | 17, 37, 59 per arm |
| Curriculum | Four on-policy cycles through agriculture, manure, refuge, farmstead_mixed |
| Budget | 4,096 decisions per curriculum task, 16,384 per seed/arm; 4 worlds, rollout 256, 2 PPO epochs |
| Injury/death penalty | 1× agriculture/manure; 3× refuge/mixed |
| Offline M4 rehearsal | Disabled (0), so only the new on-policy safeguard differs |
| Development | 48 matched safe cases seeded 2,400,000 plus 48 threatened cases seeded 2,500,000; horizon 192 |
| Reserved blind audit | 48 safe at 3,400,000 and 48 threatened at 3,500,000; consulted at most once, for one candidate that first passes all development gates |
| Selection | Exact existing M16e–M16i food/manure, health, crop-damage and survival gate; ≥3 training seeds, ≥48 episodes per scenario; never auto-install |

Three arms × 3 seeds × 16 segments = **144 stage trainings** and
**147,456 PPO decisions**, followed by **2,592 development world episodes**
including the established ablated-action controls. Training uses the
separate 3B+ deterministic seed namespace; no held-out seed block is
used for optimizing teacher targets. Development and audit blocks are
distinct from M16h's 2.0M/3.0M, M16g's 1.8M/2.8M and M16i's 2.2M/3.2M.

The zero-KL arm should follow archived M16i's manure-0.3 *control path*:
the new KL operation is skipped entirely. Cross-run parameter equality is
examined below; bit-for-bit equality across separate runtime setups cannot
be assumed.

### Reproduction

The archived M4 baseline is not checked into Git.

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16j_001 --weights 3 --stage-budgets 4096 --interleave-cycles 4 --manure-credits 0.3 --food-anchor-kls 0 0.5 2 --epochs 2 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 2400000 --audit-seed 3400000 --dry-run
```

Remove `--dry-run` to train. For a small *nonqualifying* CLI check, override
with `--stage-budgets 256 --worlds 2 --rollout 64 --epochs 1 --episode-cap
18 --training-seeds 17 --development-episodes 2 --audit-episodes 2` and a
new destination. Existing experiment paths are intentionally protected.

## Interpretation and safety

- Compare true held-out **food deposits and manure application**, not PPO
  return or training-only anchor-KL reduction.
- Count actual sampled food-anchor states; if coverage is near zero, the
  intervention lacked opportunities to influence the policy.
- A candidate with high manure use but poor food delivery still fails.
- Do not retune on audit seeds or promote a failed network; the control and
  previous M4 model remain available for direct comparison.
- One positive candidate among nine is not broad ecological generalization.
  M4's skillful movement may involve contexts the selective safeguard misses.

## Completed first M4-baseline study

All **144 curriculum-stage trainings** and **2,592 development world
episodes** completed with the budgets and seed blocks declared above. All
nine candidates were evaluated against the same immutable M4 reference.
The following are actual safe-world outcomes, not shaped PPO training rewards.

| Food-anchor KL | Training seed | Safe food deposited | Food Δ vs M4 | Safe manure applied | Manure Δ vs M4 | Threat survival |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (control) | 17 | 2.933 | −0.882 | 3.254 | +1.107 | 0.979 |
| 0 (control) | 37 | 1.917 | −1.899 | 0.000 | −2.147 | 1.000 |
| 0 (control) | 59 | 2.933 | −0.882 | 1.169 | −0.977 | 0.979 |
| 0.5 | 17 | 2.045 | −1.770 | 0.697 | −1.450 | 1.000 |
| 0.5 | 37 | 2.933 | −0.882 | 1.466 | −0.681 | 0.979 |
| 0.5 | 59 | 2.955 | −0.861 | 2.554 | +0.407 | 1.000 |
| 2.0 | 17 | 2.975 | −0.840 | 3.887 | +1.740 | 1.000 |
| 2.0 | 37 | 2.975 | −0.840 | 4.222 | +2.076 | 1.000 |
| 2.0 | 59 | 2.766 | −1.050 | 3.199 | +1.053 | 1.000 |

M4's matched safe-world reference was **3.815 food deposited** and
**2.147 manure applied**; its threat-world survival was **1.000**.
Average outcome across the three independent training seeds per arm:

| Food-anchor KL | Mean safe food | Mean safe manure | Mean threat survival | Mean last-stage anchor coverage |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.594 | 1.474 | 0.986 | 0% |
| 0.5 | 2.644 | 1.572 | 0.993 | 12.4% |
| 2.0 | **2.905** | **3.770** | **1.000** | 12.2% |

**Gate result: 0/9 qualified.** With KL 2.0, all three independently
trained candidates exceeded M4 on held-out manure application, and every
one passed every *other* food/manure/survival/health/crop-damage gate.
Each nevertheless deposited less food than M4, failing the required
`food_storage_improves` check. KL 0.5 had mixed manure outcomes; the control
also had mixed manure and generally less food. The stronger safeguard
increased mean held-out food by about **0.311** and manure by **2.295**
relative to the matched zero-anchor arm, but mean food still lagged M4
by approximately **0.910**. This is descriptive evidence from only three
training seeds, not a claim of ecological generalization.

For KL 2.0 seed 37, the matched-case bootstrap 95% intervals were
**[+1.477, +2.665]** for manure difference and **[−1.236, −0.474]**
for food difference relative to M4: the gain and the shortfall coexist.
Last-rollout anchor coverage was only approximately 12% of sampled PPO
states; this may be insufficient to retain the complete multi-step food
routine, although coverage alone does not establish causality.

### Control and audit integrity

- SHA-256 of M4 checkpoint was reverified:
  `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd`.
- Zero-anchor training skips teacher loading and the new loss completely.
  Against archived M16i manure-0.3 seed 17/37/59 tensors, the maximum
  absolute differences were **4.47e−8, 5.96e−8, 5.96e−8** respectively;
  these are tiny cross-run numerical differences, **not bit equality**.
  Two repeated zero-anchor runs within the current runtime produced
  bit-identical parameters. No material PPO drift was found.
- The development report records `development_qualifiers: 0`,
  `selected_before_audit: null`, `audit: null`, and
  `eligible_for_manual_review: false`. The audit seed ranges were not
  evaluated; no checkpoint was installed or promoted.

### Research interpretation and next experiment

On-policy teacher anchoring appears to have helped **transfer actual manure
use** while modestly retaining more food than an unanchored manure learner.
The remaining failure is food delivery, not a justification to weaken the
gate. M16k should diagnose the full farmer work chain—resource navigation,
manure collection/application, planting, harvesting, carrying, deposit—on
paired held-out trajectories, then test an explicit short-horizon task-phase
or goal-memory representation *without* touching world physics or using the
closed audit for tuning. Keep M4 in service until that complete chain clears
the original development and single-use audit procedure.
