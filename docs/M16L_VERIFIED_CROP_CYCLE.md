# M16l — Verified farmer crop-cycle credit

**Status:** Implementation and full predeclared matched study complete;
**0/6 development qualifiers**. The reserved blind audit was never opened.
No model was promoted and the original M4 farmer remains installed.

## Motivation and hypothesis

[M16k's completed study](M16K_FARMER_TASK_MEMORY.md) found that all six
trained neural farmers chose **zero planting actions** in a 24-episode
independent farmstead diagnostic despite legal opportunities; original M4
planted 1.875 times per episode. The environment's verified planting action
costs 0.3 units of carried food and has **no immediate reward**. Harvest
and deposit only pay later, so the sequence has difficult credit assignment.

**M16l hypothesis:** training-only credit for the observed
plant→harvest→deposit sequence restores genuine crop production without
reducing manure use or safety. We compare it with an otherwise identical
zero-credit control, both starting from the same immutable M4 checkpoint.

## Intervention: finite event-provenance accounting

A new per-episode `FarmerCropCycleLedger` consumes only successful
world-emitted events attributable to that farmer:

| Verified event | Training credit at strength 1 |
| --- | ---: |
| Plant a real crop at an event-provided coordinate | +0.6, **first two** positive planting events per episode only |
| Harvest food from a coordinate the same farmer planted in this episode | +0.4 per physical unit harvested |
| Deposit food attributable to that owned harvest | +0.6 per physical unit delivered |

The ledger tracks a conservatively bounded quantity of the farmer's own
harvested food. Seed expenditure, eating carried food, and festival
contributions subtract from that quantity; depositing consumes it as well,
so repeated deposits or depositing initial seed grain cannot earn a bonus.
Missing event position, zero-amount actions, harvesting another plot,
and another actor's actions never generate crop-chain credit. Valid
positive planting events are tracked even after the two-plant bonus cap,
so genuine later harvest and delivery still receive credit.

The training strength multiplies all these bonuses, with `0` forming the
same-code control and `1` forming the proposed treatment. Both arms use
**the original 76-feature M4 architecture** (`history_mode=none`), the
same initial weights/optimizer state, the same legal-action mask, and the
same frozen M4 food-anchor KL regularizer. No scripted goals, hidden crop
queries, privileged policy observations, world physics, or evaluation
rewards are changed. We report each verified event count separately from
shaped returns, so a higher training return alone cannot constitute
evidence of better farming.

## Predeclared, held-out comparison

| Setting | Fixed protocol |
| --- | --- |
| Immutable M4 checkpoint | Original 32,768-step model; SHA-256 `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd` |
| Arms | Crop-cycle credit `0` (control) and `1` (treatment); same v1 architecture |
| Shared manure/food regularizers | Manure credit `0.3`; frozen M4 on-policy food KL `2.0` |
| Seeds and stages | Training seeds `17`, `37`, `59`; four interleaved cycles of agriculture, manure, refuge, mixed farmstead |
| Budget | 4,096 actions per stage = 16,384 per seed; four worlds, rollout 256, two PPO epochs |
| Survival scaling | `1` on agriculture/manure, `3` on refuge/mixed stages |
| Development | 48 paired safe at 4,600,000; 48 paired threatened at 4,700,000, 192 steps |
| Reserved single-use blind audit | 48 paired safe at 5,600,000; threatened at 5,700,000; opened only if an eligible candidate exists |
| Selection | Unchanged M16e–M16k food/manure/survival/health/crop-damage gates; global top development-eligible candidate, then one audit |

Total: **96 training segments, 98,304 PPO decisions, 1,728 development
world episodes** (including paired benchmarks and ablations). The pilot
configuration uses a separate output directory, fewer seeds and shorter
episodes, and is **nonqualifying**. For descriptive behavior analysis we
also compare the original M4 and six frozen candidates on a shared
24-episode non-audit block (seed 1,750,000) with legal/chosen planting,
harvesting, deposit, and visible ripe crop opportunities. Those traces
do not set gates or trigger an audit.

### Reproduction

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16l_001 --weights 3 --stage-budgets 4096 --interleave-cycles 4 --manure-credits 0.3 --food-anchor-kls 2 --history-modes none --crop-cycle-credits 0 1 --epochs 2 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 4600000 --audit-seed 5600000 --dry-run
```

Remove `--dry-run` to train and evaluate. For an inexpensive but **not
statistically qualifying** local smoke test, use new output directories,
`--stage-budgets 256 --worlds 2 --rollout 64 --epochs 1 --episode-cap 18
--training-seeds 17 --development-episodes 2 --audit-episodes 2`.

## Development and audit results

All **96 stage checkpoints** were trained as planned. Results below are
means on the paired 48-world *safe* development block, except threatened
health from the paired 48-world threatened block. Figures report **real
physical outcomes**, with no training-only bonus included.

| Policy | Seed | Food deposited | Food harvested | Manure applied | Threatened final health | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original M4 (reference) | — | **3.2936** | **4.3924** | 2.2437 | 98.5000 | Reference |
| No crop credit | 17 | 2.9753 | 2.0753 | 3.5781 | 98.7500 | Fail: food |
| No crop credit | 37 | 2.9753 | 2.0753 | 4.0634 | 98.5000 | Fail: food |
| No crop credit | 59 | 2.6778 | 1.7777 | 3.2838 | 98.7500 | Fail: food |
| Crop credit 1 | 17 | 1.9167 | 1.0167 | 1.4185 | 98.7500 | Fail: food, manure, safe crop damage |
| Crop credit 1 | 37 | 2.9837 | 2.0837 | 0.9983 | 99.0000 | Fail: food, manure |
| Crop credit 1 | 59 | 2.9753 | 2.0753 | 0.2555 | 98.2500 | Fail: food, manure |

Across seeds, the zero-credit control averaged **2.8761** deposited food,
**1.9761** harvested food and **3.6418** manure; the crop-credit treatment
averaged **2.6252** food, **1.7252** harvest and **0.8908** manure. The
original M4 on the same safe seeds averaged **3.2936** deposited and
**4.3924** harvested. Threatened survival remained **100% for all six**
candidates. The credit treatment generally **worsened** food delivery and
especially manure application, and no candidate passed the food gate.

The paired bootstrap for the superficially closest food result,
crop-credit **seed 37**, showed −0.3099 deposited food relative to M4
(95% CI −0.5827 to −0.0410), −2.3087 harvested food (CI −3.5661 to
−1.2208), and −1.2454 manure (CI −1.5776 to −0.8559). These comparisons
are descriptive; the full predeclared multi-metric gate decides eligibility.

### Training event credit versus frozen-policy behavior

Training-metric files recorded successful planting events aggregated across
the same 16,384 PPO decisions per seed:

| Training seed | No crop credit: planted | Crop credit 1: planted | Credit awarded in treatment |
| --- | ---: | ---: | ---: |
| 17 | 130 | 150 | 67.7759 |
| 37 | 147 | 190 | 82.8839 |
| 59 | 155 | 183 | 87.4577 |
| **Total** | **432** | **523** | **238.1175** |

The credit arm elicited about **21.1% more successful planting events
during stochastic PPO rollouts**. But that did **not** generalize into the
preferred deterministic actions when each frozen network was evaluated.
The trained networks still substantially favor actions other than planting.

All seven policies were compared independently on the same **24 non-audit
farmstead episodes**, seeds 1,750,000–1,750,023 (horizon 192). Legal/chosen
planting and visible-ripe counts were predeclared. The probability of a
planting action *conditional on its legality* and its top-two rank were
added afterward as exploratory explanatory diagnostics; neither affects
the gates, action selection or audit decision.

| Policy | Plant legal steps / episode | Plant chosen / episode | Ripe-visible steps / episode | Plant probability when legal |
| --- | ---: | ---: | ---: | ---: |
| Original M4 | 5.17 | **2.08** | 114.38 | 0.158 |
| No credit 17 | 1.67 | **0** | 10.50 | 0.084 |
| No credit 37 | 1.67 | **0** | 10.63 | 0.084 |
| No credit 59 | 4.58 | **0** | 97.58 | 0.129 |
| Crop credit 17 | 1.00 | **0** | 52.79 | 0.151 |
| Crop credit 37 | 1.67 | **0** | 16.08 | 0.111 |
| Crop credit 59 | 1.67 | **0** | 78.92 | 0.105 |

Every candidate had legal opportunities and **zero chosen planting**.
The treatment seed-17 model even ranked planting in its top two actions
at every legally available instance while still never ranking it first.
In other words, positive plant-event credit during stochastic training
was insufficient to restore **deterministic planting preference**.
Exploration counts alone cannot demonstrate a capable agricultural policy.

### Gate integrity and recommendation

The aggregate sweep report records **`development_qualifiers: 0`**,
`selected_before_audit: null`, `audit: null`, and
`recommended_checkpoint: null`. None of the reserved 5,600,000/5,700,000
audit episodes was opened. The immutable M4 SHA-256 was rechecked by
the sweep and has not changed. All raw stage checkpoints, per-seed
development comparisons, stage training metrics and seven chain traces
are retained with the reproducibility archive.

**Next hypothesis (M16m):** investigate *relative action preference* on
legal planting states, particularly the competing top-ranked action,
and test a dedicated planting/harvesting curriculum or targeted
advantage/regularizer. Hold the fixed M4 model, deterministic evaluation,
and same full food/manure/safety gates. Avoid further undirected crop-credit
weight sweeps without evidence of restored preferred planting and full
plant→harvest→deposit behavior.
