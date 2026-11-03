# M16b — Nomad defense training, attack ablation, and herding retention

**Status:** Experimental curriculum and evaluation infrastructure. **No learned network is promoted.**

## Hypothesis

QOX's M5b nomad learns which herding signal to use but often does not engage ravagers in defense scenarios. We test whether a **shared learned nomad policy** can acquire meaningful combat while retaining its existing ability to herd independently learned cattle. In particular, we must distinguish actually stopping a predator from merely acting as a stationary shield.

## Implementation

New opt-in training/evaluation scenarios, using unchanged world combat and unchanged 72-input nomad-v2 checkpoint schema:

- **`nomad_guard`** — One nomad starts adjacent to a 24-health ravager, between the predator and three cows. The encounter is rotated by seed. An attack only succeeds when within melee distance and with stamina available. Normal physics resolves movement, damage, death, and energy.
- **`nomad_intercept`** — The nomad starts out of attack range and must approach a 36-health ravager to intervene. Rotation, cows and their positions vary by seed.
- **`nomad_defense_mix`** — Alternate Guard and Intercept situations, without modifying action rewards to pay for attacking.
- **`nomad_guard_mixed`** — Half genuine Guard episodes and half original pasture migration episodes. Both pasture directions occur. This is for rehearsal of herding skills under simultaneous defense training.

The learning pipeline also supports an optional **KL penalty** (`--anchor-kl`) toward the starting learned policy, with frozen reference weights. The anchor does not override actions or supply scripted combat labels. Original checkpoint schema and defaults remain compatible.

Added a **`compare-defenders`** CLI evaluator. It evaluates the baseline and candidate with a frozen call-aware learned-cow checkpoint using identical held-out seeds on Guard, Intercept, the original `nomad_defend` setting, and `nomad_herd`. It measures the number of surviving cows, cow health, attacks, ravager survival, nomad survival/health, pasture arrivals, and call counts. It includes an **attack-disabled counterfactual**, a scripted reference, and no-call migration ablation. Paired bootstrap intervals convey approximate uncertainty, not definitive statistical proof. Candidate promotion requires gains in cattle outcomes and defender survival while retaining pasture performance; **the command never overwrites a checkpoint**.

## Initial experimental results

From the original frozen M5b nomad and M5c cow checkpoints, an **8,192-decision Guard PPO** candidate was evaluated on 24 validation worlds per scenario (seed series starting at 910,000, with non-overlapping scenario-specific offsets; 40 ticks per world).

| Metric | Original learned nomad | Guard-trained candidate |
|---|---:|---:|
| Guard: cows surviving / 3 | 2.375 | **3.000** |
| Guard: average cow health (deaths count as zero) | 50.12 | **65.00** |
| Guard: nomad attacks per world | 1.21 | **3.00** |
| Guard: ravager survival | 62.5% | **0%** |
| Intercept: cows surviving / 3 | 2.250 | **2.500** |
| Intercept: average cow health | 43.25 | **53.50** |
| Intercept: ravager survival | 87.5% | **50%** |
| Original defense: cow health | 31.17 | 35.33 |
| Migration: cows arriving / 3 | **1.875** | 1.458 |

**Crucial counterfactual:** When the candidate's attack actions are disabled on Guard worlds, the nomad's survival falls from 100% to 0% and all ravagers remain alive. **Cattle survival and measured mean cow health stay the same over these short Guard episodes**, so the positive result is not proof that attacking *caused* the higher cattle survival; moving into an intercepting position is itself protective. In Intercept worlds, disabling attacks similarly prevents predator elimination. Longer trials with different enemy behaviors are needed for causal claims about protecting cattle through combat.

The candidate fails the migration-retention gate (about **0.42 fewer cows** to pasture per world). It is **not** adopted. An early attempt at mixed training with stronger anchoring preserved migration on a small validation set but did not learn useful attacks. A later two-stage experiment learned more attacks in Intercept but also degraded migration. These exploratory variants are *not* promoted.

No 24-world tuning score should be reported as a blind final test; reserve new unseen seeds for future testing of a candidate that actually clears the validation criteria. The sample size, prepared cattle health, altered predator health and explicit geometry all limit generalization. None of these data establish long-run survival in the full ecosystem.

## Run the experiments

With a local Python/PyTorch installation, starting from compatible nomad-v2 and cow-v2 checkpoints:

```powershell
qox train-nomad --scenario nomad_guard --nomad-v2 --steps 8192 --worlds 4 --rollout 256 --max-steps 36 --seed 16610 --init-checkpoint checkpoints/nomad_m5b_selected.pt --output runs/guard_candidate.pt

qox train-nomad --scenario nomad_guard_mixed --nomad-v2 --steps 8192 --worlds 4 --rollout 256 --max-steps 40 --anchor-kl 0.03 --init-checkpoint checkpoints/nomad_m5b_selected.pt --output runs/mixed_candidate.pt

qox compare-defenders --baseline-nomad checkpoints/nomad_m5b_selected.pt --candidate-nomad runs/guard_candidate.pt --cow-checkpoint checkpoints/cow_m5c_selected.pt --episodes 24 --seed 910000 --max-steps 40 --report runs/m16b_validation.json
```

Do not use the validation seeds again as an unbiased test set once they have influenced candidate selection.

## Next research work

1. Train a mixed policy that performs positional interception **and** preserves both friendly/scary herding signals; consider prioritized experience replay of migration *scenarios*, not canned action labels.
2. Evaluate multiple independent training seeds and harder held-out threat geometries (including an uninjured ravager), then test on larger herds with more than one nomad.
3. Add counterfactuals where the ravager targets cows rather than nomads and where the defender can safely flee, to separate true herd protection from sacrificial blocking.
4. Test longer integrated worlds and report cattle survival, food production, and festivals without assuming that controlled-scenario gains generalize.

## Integrity and compatibility

- Same physics, actor attributes, action masks, and observation features as M16a.
- Scenario randomization does not expose a global threat coordinate or a scripted action to the policy.
- Existing M5b/M5c checkpoints still load; defaults unchanged.
- All candidate comparisons are frozen-policy and non-destructive.
- `nomad_guard` and `nomad_intercept` are training fixtures, not rebalanced normal-world combat rules.
