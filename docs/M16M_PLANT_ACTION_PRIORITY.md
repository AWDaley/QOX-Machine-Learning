# M16m — Frozen-M4 planting-action preference study

**Status:** Full preregistered two-arm study complete. **0/6 development
qualifiers.** The single-use blind audit was never opened, no candidate was
promoted, and the original M4 checkpoint is unchanged.

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

All **96 PPO segments** were completed, totaling **98,304 farmer action
decisions** and **1,728 development-world episodes**. On the predeclared
48-world **safe** development set (seeds 6,600,000–6,600,047), the original
M4 deposited **3.1880**, harvested **3.7910**, and applied **2.4669**
units of manure. In the threatened block, original M4 final health averaged
**96.7083**. Below, health comes from the threatened block; other quantities
come from the safe block. All are physical outcomes, **not PPO shaping**.

| Model | Seed | Food deposited | Harvested | Manure applied | Threatened final health | Failed gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M4 (reference) | — | **3.1880** | **3.7910** | 2.4669 | 96.7083 | Reference |
| Priority `0` | 17 | 2.9753 | 2.0753 | 3.9494 | 96.4583 | Food |
| Priority `0` | 37 | 2.9753 | 2.0753 | 3.7163 | 96.4583 | Food |
| Priority `0` | 59 | 2.3702 | 1.4702 | 3.5434 | 95.7083 | Food |
| Priority `1` | 17 | 2.7333 | 2.0333 | 1.7321 | 94.9583 | Food, manure, threatened health |
| Priority `1` | 37 | 3.1097 | 2.2097 | 0.2533 | 96.9583 | Food, manure |
| Priority `1` | 59 | 2.9753 | 2.0753 | 0.8241 | 96.7083 | Food, manure |

Across the three matched seeds, control means were **2.7736** food,
**1.8736** harvested and **3.7364** manure; treatment means were
**2.9394** food, **2.1061** harvested and **0.9365** manure. The targeted
margin yielded a small descriptive food improvement of **+0.1658** over
the control mean, but remained below original M4 food delivery and
substantially impaired manure use. The **unchanged full gate failed all
six candidates**, so **no audit candidates were selected** and the
reserved 7,600,000/7,700,000 seed blocks were never evaluated. These
sample means are descriptive, not proof of generalization.

### Real training versus deterministic policy

Training used identical 16,384-decision budgets per seed. Successful
plantings were as follows:

| Seed | Priority `0`: planted | Priority `1`: planted | Share of treated PPO states where M4 preferred legal PLANT |
| --- | ---: | ---: | ---: |
| 17 | 130 | 231 | 4.61% |
| 37 | 147 | 187 | 4.93% |
| 59 | 155 | 186 | 3.98% |
| **Total** | **432** | **604** | — |

Targeting M4-preferred planting states increased **successful stochastic
training plantings by 39.8%**. Unlike M16l, one treatment seed also
recovered *some* greedy planting in the separate non-audit 24-case
diagnostic at seed 1,850,000. Still, two of three trained treatment
policies planted zero, and all three were below the original M4.

| Frozen policy | Legal planting opportunities / episode | Greedy PLANT / episode | PLANT probability when legal | Dominant missed action |
| --- | ---: | ---: | ---: | --- |
| Original M4 | 8.0417 | **3.1250** | 0.1673 | DEPOSIT and REST, 59 each |
| Priority `0`, seed 17 | 1.6667 | 0 | 0.0838 | DEPOSIT |
| Priority `0`, seed 37 | 1.6667 | 0 | 0.0836 | DEPOSIT |
| Priority `0`, seed 59 | 3.9583 | 0 | 0.1276 | DEPOSIT, with manure actions |
| Priority `1`, seed 17 | 2.3333 | **0.6667** | 0.2000 | DEPOSIT |
| Priority `1`, seed 37 | 1.6667 | 0 | 0.1559 | DEPOSIT |
| Priority `1`, seed 59 | 1.6667 | 0 | 0.1093 | DEPOSIT |

The M16m mechanism is therefore **partly effective at teaching action
ranking** in one seed but does not produce a farmer worthy of replacing
M4. The treatment may shift attention from collecting and applying
manure into planting without completing sufficient productive cycles.
The action histogram also shows DEPOSIT still displacing PLANT in most
missed treated-policy opportunities; the intervention did not eliminate
the action-ranking failure. The identical control-arm training planting
counts (130/147/155) reproduce M16l's zero-crop-credit controls and
support budget and seed consistency.

### Next hypothesis (not tested here)

Rather than increasing this loss weight or cherry-picking the one seed,
M16n should test **balanced crop/manure skill retention**: a fixed,
training-only rehearsal set with higher coverage of M4 planting contexts
*and* a matched preserve-manure constraint. Predeclare the held-out gate,
single audit, total training budget and scripted/random descriptive
controls again. The original M4 is still the safest production choice.
