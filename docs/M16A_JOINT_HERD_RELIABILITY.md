# M16a — Joint Herd Reliability and Predator-Pressure Curriculum

**Status:** Engineering extension plus exploratory validation. No policy promoted.  
**Scope:** Learned cows + learned nomad; does not implement M16 farmer planning, ravager-complete hunting, or fully learned festivals.

## The research problem

In M5c, independently trained cows and nomads reached approximately 1.96 of 3 cattle at a target pasture on 48 held-out worlds, compared with 2.77 for the scripted reference. Some movement occurred when calls were disabled, so the result cannot be attributed entirely to learned communication. Neither learned policy handled herd defense especially well.

M16a extends this existing pipeline rather than introducing new observation or action schemas:

- **Three training scenarios:** `nomad_herd` (predator-free migration), `nomad_defend` (a scripted ravager attacks), and `nomad_mixed` (seed-balanced migration or defense).
- **Frozen-partner action mode:** `greedy` (original behavior) or `sample` (a more variable learned partner; *not* a scripted cow responder).
- **Correct early-reset handling:** If any member of a training world dies, the whole world's rollouts end. Remaining actors' advantage estimates are also terminated so PPO cannot mix reward sequences across independent episodes.
- **Paired evaluation and rejection:** The `compare-herd` command runs *both learned species together*, under the same migration and defense seeds, with a no-calls counterfactual; it reports paired bootstrap intervals. It **never** changes the active checkpoint and makes no claim of population-scale ecological stability.
- **Compatibility:** Existing 66-feature cow v2 and 72/76-feature nomad v1/v2 policy checkpoints are unchanged. The first two are the intended M16a initial policies. No world state, rendering or festival behavior is modified.

The cow network cannot see the hidden pasture waypoint. The nomad can receive the pasture goal through its pre-existing task-specific observation. No call teleports cattle or forces a response.

## Commands

From a source checkout with `.[ml]` installed, using a compatible pre-trained cow v2 and nomad v2 checkpoint:

```powershell
qox train-herd --scenario nomad_mixed --partner-mode sample `
  --phase alternate --cow-checkpoint checkpoints/cow.pt `
  --nomad-checkpoint checkpoints/nomad.pt --cow-steps 6144 `
  --nomad-steps 3072 --worlds 4 --rollout 384 `
  --max-steps 64 --seed 7720 --output-dir runs/m16-mixed

qox compare-herd --baseline-cow checkpoints/cow.pt `
  --baseline-nomad checkpoints/nomad.pt `
  --candidate-cow runs/m16-mixed/cow_round1.pt `
  --candidate-nomad runs/m16-mixed/nomad_round1.pt `
  --episodes 24 --seed 660000 --max-steps 64 `
  --report runs/m16-comparison.json

qox benchmark-herd --scenario nomad_defend `
  --cow-checkpoint checkpoints/cow.pt --nomad-checkpoint checkpoints/nomad.pt `
  --episodes 24 --seed 860000 --max-steps 64
```

**Policy-deployment rules:** Training creates new files. `compare-herd` reports diagnostic checks but does not deploy, overwrite, merge, or re-train models. Use distinct, untouched *test seeds* after candidate selection; validation comparisons used to choose a model are not a blind scientific test.

## Initial validation results (24 common seeds, 64 ticks)

Starting pair: previously selected M5c cow policy and M5b nomad policy. These weights are *not bundled with the source repository*; use valid `.pt` models from an existing local QOX training package. Two experimental policies were trained independently:

| Measurement | Original pair | Mixed-training candidate | Migration-only cow update |
|---|---:|---:|---:|
| Cow training decisions | — | 6,144 | 12,288 |
| Nomad training decisions | — | 3,072 | none (unchanged) |
| Mean cows at pasture (out of 3) | **1.875** | 1.500 | 1.917 |
| Mean cows at pasture with calls disabled | 1.250 | 1.333 | 1.667 |
| Correct-call lift vs no-call | **0.625** | 0.167 | 0.250 |
| Mean cow health in defense (100-point scale) | 33.17 | **36.67** | 31.83 |
| Mean cows alive in defense (out of 3) | 2.00 | 2.00 | 2.00 |

The mixed policy's paired change in pasture arrivals was **−0.375 cows/world** (paired bootstrap 95% interval approximately **−1.19 to +0.40**). Its cow-health improvement under predator pressure was **+3.50 points**, but this did not improve how many cows survived. The migration-only update gave **+0.042 cows/world** with an interval spanning negative and positive outcomes, and showed weaker call sensitivity. **Neither candidate passes the policy-selection diagnostic; the baseline checkpoint remains preferred.** These validation seeds were reused for comparing candidates, so do not treat them as a final untouched test.

The current nomad pair registered **zero nomad attacks per defense episode** under the evaluated greedy policies, so effective learned protection has not been demonstrated. A stronger defense curriculum is needed before claiming M16 cooperative protection.

## Next experiments

1. Train a nomad against **frozen learned cattle**, adding early reward for preserving cow health only through real defensive outcomes (not +points for attack action selection). Compare nomad positioning and actual attacks; do not let defense training destroy existing call-selection capability.
2. Alternate short migration and defense training phases; select checkpoints via independent validation and evaluate once on 100+ untouched test seeds.
3. Add multiple fixed predator opponents or saved ravager checkpoints. Current defender-pressure tests use a scripted predator, which cannot establish learned-versus-learned predator combat.
4. Extend from pasture-arrival episodes to genuine grazing and cattle survival over longer resource-constrained worlds; continue to log no-call and reversed-call counterfactuals.
5. Only then integrate the resulting validated candidate pairs into the Windows training laboratory as an explicit joint-herding scenario.

## Implementation tests

- Strict checkpoint schema loading for cows and nomads, with identical legacy action vectors.
- Old `nomad_herd` default and M5c tests still pass.
- Training in mixed predator/no-predator worlds handles early deaths and maintains shared policy across three cattle.
- Fixed-seed comparisons are deterministic, include no-call controls and report raw per-seed results.
- Candidate evaluation does not modify any original checkpoint; an identical candidate is never auto-accepted.

**Interpretation:** M16a adds an experimentally sound, predator-aware learning path and provides concrete evidence that training improvements are not guaranteed to transfer. This milestone is successful as instrumentation and methodology; it is not a claim that the neural ecosystem has solved herding or defense.
