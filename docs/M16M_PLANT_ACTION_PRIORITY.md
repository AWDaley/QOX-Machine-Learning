# M16m — Frozen-M4 planting-action preference study

**Status:** Predeclared matched protocol. Frozen-policy action competition was
diagnosed on a disjoint exploratory block before selecting this intervention.
Results and promotion decisions are appended only after the full run.

## What M16l missed

M16l successfully increased verified planting during stochastic training,
but not one of six final deterministic farmers planted in the independent
24-episode diagnostic. To understand the ranking failure, M16m added a
**read-only, post-selection** action-kind histogram when PLANT is legal but
the greedy policy chooses something else. It never affects inference.

An additional, non-audit 24-episode farmstead block at seeds
**1,850,000–1,850,023** (192-step horizon) gave:

| Frozen model | Legal PLANT steps per episode | Chosen PLANT per episode | What beat PLANT (total missed steps) |
| --- | ---: | ---: | --- |
| Original M4 | 8.0417 | 3.125 | DEPOSIT 59, REST 59 |
| M16l control, seed 17 | 1.6667 | 0 | DEPOSIT 40 |
| M16l control, seed 37 | 1.6667 | 0 | DEPOSIT 40 |
| M16l control, seed 59 | 3.9583 | 0 | DEPOSIT 38, FERTILIZE 27, COLLECT_WASTE 18, MOVE 12 |
| M16l credit, seed 17 | 1 | 0 | DEPOSIT 24 |
| M16l credit, seed 37 | 1.6667 | 0 | DEPOSIT 40 |
| M16l credit, seed 59 | 1.6667 | 0 | DEPOSIT 40 |

Across the six trained candidates, **222 of 279 missed planting
opportunities (79.6%)** were displaced by DEPOSIT. This is descriptive,
not a matched causal estimate: different policies visit different states.

## Hypothesis and scope

**Hypothesis:** the learner needs a stronger *relative action-preference*
constraint in planting states, not another immediate crop reward. M16m
regularizes the student's distribution when the **frozen original M4**
teacher chooses PLANT from the **same current legal-action mask and
actor-local observations**. On each selected PPO minibatch observation,
add a hinge loss:

\[
  L_{\rm plant}=\max\{0,\;0.25-(\ell_{\rm plant}-
                    \max_{a\ne {\rm plant}}\ell_a)\}.
\]

It is averaged over only teacher-selected **legal PLANT** contexts;
minibatches with none add zero gradient. The teacher is inference-only,
has no optimizer and sees only its unchanged original 76 features.
The 0.25 logit gap and treatment weight `1` are fixed before running.
The baseline food-anchor KL remains unchanged at `2`, so this is a
separate preference term, not a different teacher or reward.

No scripted actions are substituted, legal masks are not modified,
inference remains ordinary policy argmax, the neural architecture is the
same v1 M4 network, and world physics and evaluation rewards are unchanged.
This experiment probes preference transfer; it cannot claim learning a
genuinely improved farming strategy without the unchanged outcomes gate.

## Locked study design

| Setting | Predeclared value |
| --- | --- |
| Immutable M4 farmer | Original 32,768-step checkpoint SHA-256 `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd` |
| Arms | PLANT-priority weight `0` same-code control vs `1` treatment |
| Shared settings | Manure credit `0.3`, food-anchor KL `2.0`, crop-cycle credit `0`, v1 history `none` |
| Training | Seeds `17, 37, 59`; four cycles of agriculture → manure → refuge → mixed farmstead |
| Budgets | 4,096 decisions/stage, four worlds, 256 rollout, 2 PPO epochs; survival scale 3 on last two stage types |
| Development | 48 paired safe episodes from 6,600,000; 48 threatened from 6,700,000; 192 steps; no earlier seed range reused |
| Reserved *blind audit* | 48 safe episodes from 7,600,000 and 48 threatened from 7,700,000; accessed at most once and only if a development-eligible candidate exists |
| Gate | **Unchanged** M16e–M16l food, manure, safety, health, crop-damage and threatened-food criteria; select maximum food gain among *eligible* development candidates |
| Promotion | No automatic promotion; manual review only if one candidate also passes its single audit |

Expected work: **96 segments; 98,304 PPO decisions; 1,728 development
world episodes** counting all three paired evaluation modes. The primary
outcome is real held-out deposited food with the **entire original gate**,
not shaped return, training planting count or post-hoc behavior. Measure
physical harvested food, manure, survival, health, crop damage; report
every seed and failed gate. A second independent 24-episode frozen-policy
diagnostic at 1,850,000 is explanatory only, never used for selection.

Separate lightweight validation tests use tiny synthetic checkpoints,
one seed and short horizon and cannot qualify. All six full models start
from identical M4 bytes. We stop before any audit unless a full development
gate passes; the original checkpoint and installed policies remain untouched.

### Reproduction

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16m_001 --weights 3 --stage-budgets 4096 --interleave-cycles 4 --manure-credits 0.3 --food-anchor-kls 2 --crop-cycle-credits 0 --history-modes none --plant-priorities 0 1 --epochs 2 --worlds 4 --rollout 256 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 6600000 --audit-seed 7600000 --dry-run
```

Remove `--dry-run` to execute. Reserve a new output directory each time.

## Full study results

Pending full locked run.
