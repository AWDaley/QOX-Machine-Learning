# M16d — Farmer planning and safe resource production

**Status:** Implemented farmstead curricula and evaluation; experimental policy **not promoted**.

## Question

Can farmers learn to produce and store food over longer periods, recycle finite
manure, and enter a real shelter under predator pressure—without sacrificing
survival? Earlier farmer tests separated these abilities into short episodes.

## New scenarios

- **farmstead:** 12×12 world, one learned farmer and one scripted cow;
  multiple planted fields of varying fertility, finite dung, crop damage,
  carried seed, and a settlement food reserve.
- **farmstead_threat:** identical agricultural mechanics with one physical
  ravager that approaches the fields. Farmer refuge still consumes food and
  prevents outside work.
- **farmstead_mixed:** alternating safe/threat worlds selected at reset, not
  an inserted predator during an episode. This dynamic-population curriculum
  is supported by native PPO, not the fixed-population PettingZoo adapter.

No new omniscient observations, scripted actions, world physics changes, or
checkpoint feature/action shapes. The 76-input farmer policy remains compatible.

## Training and comparison

The Windows Training Lab exposes the three farmstead scenarios with 192-tick
episodes. Candidate policies do not replace existing networks until accepted.

Command line:

    qox train-farmer --scenario farmstead_mixed --steps 4096 --max-steps 192 --seed 31027 --output runs/m16d_candidate.pt
    qox compare-farmers --baseline runs/baseline_farmer.pt --candidate runs/m16d_candidate.pt --episodes 48 --max-steps 192 --report runs/m16d_review.json

The comparison uses matching seeds and durations for baseline/candidate across
safe and threatened worlds, plus the candidate with fertilizing or entering
refuge disabled. Reports include per-world results and paired bootstrap
intervals. Comparisons read and hash policy checkpoints but never modify them.

## Exploratory validation (24 matched seeds, 160 ticks)

Fine-tuned the previously archived **M4 32k farmer** for 4,096 decisions on
the mixed farmstead, then evaluated 24 development seeds starting at 1,620,000.

| Measurement | M4 baseline | Candidate |
| --- | ---: | ---: |
| Safe farmstead: food deposited | 3.168 | **4.813** |
| Safe farmstead: manure applied | **2.324** | 1.885 |
| Safe farmstead: survival | 100% | 100% |
| Threat: food deposited | 1.635 | **3.051** |
| Threat: survival | **100%** | 91.7% |
| Threat: final health | **97.5** | 57.9 |

Removing the candidate's fertilize action reduced safe food deposits to 3.746.
Removing its refuge action reduced threatened-world survival to 16.7%.
These are controlled action ablations, not proof of a generalizable strategy.

**Decision: reject candidate for promotion.** More food production is not
adequate to compensate for worse manure application and predator survival.
The 24-seed sample is a development comparison; 48+ held-out seeds and a
separate blind range are required for candidate review.

## Guardrails and limitations

- The simulation still has only one learned farmer in these tasks; cow and
  predator opponents are scripted, not independently learned.
- All resources remain finite and all actions obey ordinary QOX mechanics.
- The rewards are unchanged; we measure actual food deposited, harvest,
  fertilizer applied, crop damage, shelter use, health, and survival.
- Reproducibility tests check action-stream world hashes and immutable models.
- If future experiments improve output while reducing survival, report
  both outcomes and do not silently replace the better-safe policy.

## Follow-up

Try staged agriculture → manure → refuge training, multiple training seeds,
longer predator encounters, and a multi-objective selection gate. A candidate
must maintain farmer health and survival while improving food production.
