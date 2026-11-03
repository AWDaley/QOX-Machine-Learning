# M16g — Frozen-policy rehearsal against catastrophic forgetting

**Status:** Implemented in the experimental farmer PPO and sweep runner.
No policy is automatically promoted or installed.

## Research question

The first M16f experiment continued the original M4 farmer for four staged
curricula, with 1× or 3× injury/death penalties, 1,024 decisions per stage,
and **two PPO epochs**. All six candidates reduced safe-world food storage
relative to M4.
Did sequential PPO updates forget the previously learned agriculture and
manure behaviors? Can explicit rehearsal preserve them while training for
survival?

## Method: masked KL on frozen M4 farming demonstrations

1. Load a compatible **trained** frozen farmer baseline (the historical M4
   policy is recommended). Its checkpoint bytes and architecture never change.
2. Before PPO, run that teacher **greedily** in separate agriculture, manure
   and farmstead training worlds (four episodes of 96 ticks each by default).
   Store only the farmer's local observations, legal-action masks and full
   teacher action probability vectors. These are training examples, never
   evaluation outcomes or privileged state.
3. Preserve rare skills during replay sampling: observations where the
   teacher chose planting, harvesting, depositing or manure work receive 8×
   sampling weight; moving receives 2× and other actions 1×. The weighting
   is fixed in advance and disclosed in the replay metadata. The teacher's
   action probabilities, not hard labels, are the distillation targets.
4. Continue ordinary PPO in four stages. On each PPO minibatch, sample an
   equal-sized independent replay batch and add a **teacher-to-student KL**
   loss, measured using the stored legal action masks:

   `total_loss = PPO_loss + rehearsal_kl * KL(M4_teacher || current_farmer)`

   `rehearsal_kl=0` recovers the original staged PPO control. Replay sampling
   uses a separate seeded torch generator, so it does not consume rollout
   or PPO-shuffle random numbers. The replay is shared across all four
   training stages and independent training seeds.
5. Compare all final candidates to the same frozen M4 baseline on matching
   development worlds using the M16e **safety and production gate**. Choose
   at most one development qualifier globally across the complete sweep.
   Only that selected candidate may be evaluated on the separate audit
   worlds. Neither the audit nor the scripted/random references can be
   used to choose another candidate.

This is offline policy distillation used as a *regularizer*, not on-screen
online learning or an override of the farmer's selected actions. Checkpoints,
per-stage replay KL values, teacher action counts, replay sample hash,
training seeds and comparisons are recorded in each experiment directory.

## Reproducible CLI

The original 32,768-decision M4 checkpoint must be supplied separately;
it is **not** committed to Git.

First estimate cost without loading that file or creating directories:

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16g_001 --weights 3 --stage-budgets 1024 --epochs 2 --rehearsal-kls 0 1 5 --training-seeds 17 37 59 --development-seed 1800000 --audit-seed 2800000 --dry-run
```

Then launch the controlled A/B/C experiment with identical training budgets:

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16g_001 --weights 3 --stage-budgets 1024 --epochs 2 --rehearsal-kls 0 1 5 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 1800000 --audit-seed 2800000
```

The two-epoch setting matches the archived M16f 3×-safety run and allows
an exact no-rehearsal policy-state control check. The run requires 36
stage trainings, about 36,864 PPO decisions, up to
2,592 development world episodes, and 24 predeclared teacher rehearsal
episodes (three worlds × four episodes × two nonzero KL configurations).
The 1.8M/1.9M development and 2.8M/2.9M audit seed blocks do not overlap
the 1.2M/1.3M/1.4M teacher-demo blocks, earlier M16f development data, or
2B+ train-stage seeds. Audit remains blind unless a candidate qualifies.

Small smoke runs can use `--stage-budgets 64 --worlds 2 --rollout 64
--epochs 1 --episode-cap 18 --training-seeds 17 --rehearsal-episodes 1
--rehearsal-max-steps 18 --development-episodes 2 --audit-episodes 2`,
but these cannot pass the sample-size gate for manual review.

## What to measure

- **Retention:** safe-world deposited food and applied manure versus the
  original M4 baseline, in raw differences with paired confidence intervals.
- **Survival:** threatened-world survival and final health, along with crop
  damage and food retained under threat.
- **Distillation strength:** trained-stage rehearsal KL and representative
  teacher action counts; low KL alone does *not* prove ecological competence.
- **Honest control:** 0× replay KL trained for the same number of decisions
  on the same seeds as nonzero-KL variants.

The M16e promotion gate still requires **at least +0.5 safe-world food
deposited** relative to M4, near-preserved manure, safe and threatened
survival, health and crop metrics. Merely matching M4 is evidence of
retention, not an acceptable improvement. No trained policy is installed
automatically. If all candidates fail development, audit remains untouched.

## Limitations

Replay observations come from the greedy teacher's visited states, not the
whole state distribution. Increasing KL strength can preserve undesirable
old behavior and suppress useful adaptations. Only the farmer is trained;
the other species remain scripted during these tests. Short-budget tests
and finite paired worlds do not establish broad generalization.
