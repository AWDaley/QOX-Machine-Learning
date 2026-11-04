# M16k — Actor-local farming task memory and action-chain diagnostics

**Status: Implementation and six-network, predeclared matched study complete;
0/6 development qualifiers.** No blind audit was opened and **no model was
promoted**. The original M4 farmer remains the production baseline.

## Why this experiment

M16j's KL 2.0 farmer improved held-out manure application across all three
training seeds but deposited less food than M4. Closer examination of the
48-case held-out development reports reveals that the gap was principally
**harvesting**, not delivery: KL-2.0 candidates harvested about 1.87–2.08
units compared with M4's 5.14, yet they deposited essentially all harvested
food plus their initial 0.9 units.

An independent, non-audit **24-episode diagnostic** on farmstead seeds
1,650,000–1,650,023 gave the following descriptive results:

| Diagnostic per episode | Original M4 | M16j KL-2.0 seed 37 |
| --- | ---: | ---: |
| Food harvested | 3.374 | 2.075 |
| Food deposited | 3.393 | 2.975 |
| Crop planting actions | 1.875 | **0** |
| Steps with ripe crops visible | 106.9 | **11.6** |
| Steps where harvest was legally available | 2.54 | 1.67 |
| Legally available harvests not chosen | 0 | 0.33 |

This suggests a breakdown in planting and subsequent navigation toward
harvestable crops. Those descriptive diagnostic seeds are not part of M16k's
new development or blind-audit blocks. The diagnostics count legal choices
and visibly ripe patches; they do not feed task goals to the actor.

## Intervention

The M16k neural farmer has a **version-2, explicitly opt-in input** which
appends 14 small, normalized actor-local history features to its original
observation. It remembers *only physically verified events performed by
that farmer*: plant, harvest, collect waste, fertilize and deposit. The
features encode recent counts over a 32-step window, age of the most recent
harvest/fertilization/deposit, the last successful event, and its age.

Two matched v2 arms have **exactly the same network architecture**:

- `masked`: all 14 history inputs stay zero (negative control).
- `events`: history inputs reflect verified farmer-owned events.

Each is initialized from the same immutable M4 checkpoint. The new first
layer's history columns start at **zero**, and the original M4 weights are
copied unchanged, so the initial v2 predictions match M4 before training.
The initial incompatible v1 optimizer state is deliberately not loaded for
*either* arm; all subsequent v2 stage continuations preserve v2 optimizer
state. This isolates the memory signal from extra network capacity and
optimizer migration.

The neural policy still chooses every action using a legal-action mask.
No code supplies a scripted goal, forces planting, moves actors, alters
world physics, changes public observations for legacy models, or consults
unobserved crops or other actors' events. Existing version-1 checkpoints
remain loadable and usable. Version-2 policies can also be evaluated, resumed,
and loaded through the desktop `.qox` model-set workflow. A resumed desktop
simulation reconstructs each farmer's bounded history from its own recorded
events. M16j's frozen M4 KL teacher sees only its original version-1
observation channels.

## Predeclared matched-budget experiment

| Setting | Fixed protocol |
| --- | --- |
| Immutable original M4 | 32,768-step checkpoint; SHA-256 `5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd` |
| Experimental arms | `masked`, `events` (shared v2 model shape) |
| Reward / teacher | Verified manure credit `0.3`, M16j food-anchor KL `2.0` for both arms |
| Training seeds | 17, 37, 59 per arm |
| Training schedule | 4 interleaved cycles × agriculture, manure, refuge, farmstead_mixed |
| Budget | 4,096 decisions per task, 16,384 per seed; 4 training worlds, rollout 256, 2 PPO epochs |
| Safety weight | 1× in agriculture/manure; 3× in refuge/mixed |
| Development | 48 paired safe cases at seed 2,600,000; 48 paired threatened at 2,700,000, horizon 192 |
| Blind audit | 48 paired safe at seed 3,600,000 and threatened 3,700,000; **never opened absent an eligible candidate** |
| Gate and selection | Original M16e–M16j food, manure, health, survival and crop-damage checks; at most one audit for the globally selected development-eligible candidate |

Two modes × three seeds × four tasks × four cycles = **96 segment
trainings**, **98,304 PPO decisions**, and **1,728 development world
episodes** including the preexisting diagnostic ablations. No threshold is
relaxed. Developer traces at 1,650,000 may illustrate failure modes, but
the 2,600,000 development block decides eligibility, and the 3,600,000
audit block is inaccessible until that decision is made.

### Reproduction

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/m16k_001 --weights 3 --stage-budgets 4096 --interleave-cycles 4 --manure-credits 0.3 --food-anchor-kls 2 --history-modes masked events --epochs 2 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48 --development-seed 2600000 --audit-seed 3600000 --dry-run
```

Remove `--dry-run` for the full experiment. For an inexpensive, **nonqualifying**
smoke test, use a new output directory with `--stage-budgets 256 --worlds 2
--rollout 64 --epochs 1 --episode-cap 18 --training-seeds 17
--development-episodes 2 --audit-episodes 2`.

Action-chain diagnostics can be requested for any frozen v1/v2 farmer:

```powershell
qox eval-farmer --checkpoint checkpoints/baseline_farmer_M4_32k.pt --scenario farmstead --episodes 24 --max-steps 192 --seed 1650000 --chain-diagnostics
```

## Development and audit results

All **96 training segments** completed at the specified budget. The following
values are means on the paired, 48-world *safe* development block except
threatened health, which uses the 48-world threatened block. Values are
rounded to four decimal places; food, harvest and manure refer to physical
quantities actually deposited, harvested or applied per episode.

| Policy | Seed | Food deposited | Food harvested | Manure applied | Threatened final health | Development gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original M4 (reference) | — | **3.4251** | **4.0084** | **2.1959** | **98.7500** | Reference |
| Masked history | 17 | 2.9502 | 2.0502 | 2.3783 | 97.2500 | Fail: food, threatened health |
| Masked history | 37 | 2.9753 | 2.0753 | 1.2333 | 94.1042 | Fail: food, manure, threatened health |
| Masked history | 59 | 1.9167 | 1.0167 | 0.0000 | 96.1042 | Fail: food, manure, threatened health, safe crop damage |
| Event history | 17 | 2.9753 | 2.0753 | 0.0781 | 98.7500 | Fail: food, manure |
| Event history | 37 | 2.9753 | 2.0753 | 3.1746 | 98.7500 | Fail: food only |
| Event history | 59 | 2.9753 | 2.0753 | 1.1712 | 98.2500 | Fail: food, manure |

Across the three training seeds, the **masked** arm averaged 2.6141 food,
1.7141 harvest, 1.2039 manure and 95.8195 threatened final health; the
**event-history** arm averaged 2.9753 food, 2.0753 harvest, 1.4746 manure
and 98.5833 threatened final health. Both arms maintained **100% threatened
survival** on these development episodes. The event arm's +0.3612 mean
food and harvest relative to the identical-architecture masked control is
**descriptive**, not a qualifying food recovery or a proven causal effect.
Every event arm seeded run reached exactly 2.9753 mean safe food and 2.0753
mean harvest, below M4: local event history did **not** recover the full
crop-production cycle.

For the promising-looking event-history **seed 37**, paired M4 comparisons
on the safe development cases had 95% bootstrap confidence intervals of
**−0.4497** deposited food (CI −0.7867 to −0.1384), **−1.9331** harvested
food (CI −3.1074 to −0.8934), and **+0.9787** applied manure (CI +0.5769
to +1.4077). Better manure application did not offset lost food.

### Observed action-chain bottleneck

Separately, all six frozen candidates and M4 were inspected on the same
**24 non-audit farmstead episodes**, seeds 1,650,000–1,650,023, horizon
192. These traces are exploratory diagnostics; the development block above
alone decided qualification.

| Policy | Plant legal steps / episode | Plant chosen / episode | Ripe-visible steps / episode | Harvest legal / episode | Harvest chosen / episode | Deposit chosen / episode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 | 4.33 | **1.875** | 106.88 | 2.54 | 2.54 | 2.46 |
| Masked 17 | 5.79 | **0** | 125.88 | 1.33 | 1.33 | 1.67 |
| Masked 37 | 1.67 | **0** | 10.00 | 2.00 | 1.33 | 2.33 |
| Masked 59 | 1.00 | **0** | 139.21 | 0.67 | 0.67 | 1.00 |
| Events 17 | 1.67 | **0** | 117.25 | 2.00 | 1.33 | 2.33 |
| Events 37 | 1.67 | **0** | 14.79 | 1.83 | 1.33 | 2.17 |
| Events 59 | 1.67 | **0** | 16.50 | 1.67 | 1.33 | 2.00 |

**All six candidates chose zero plant actions despite legal opportunities.**
Some also saw ripe crops without navigating to, or choosing, the available
harvest. M4 plants an average of 1.875 times per episode. The candidates
can deposit what they harvest, so the principal observed failure is
**planting and crop-cycle navigation, not deposit execution**. The event
history vector may help retain previously completed activity, but it does
not create incentives or planning for the next necessary action by itself.

### Gate integrity and next experiment

The completed sweep recorded **`development_qualifiers: 0`**, no globally
selected candidate, **`audit: null`**, and no recommendation. No blind-audit
episodes were run, and M4 was neither replaced nor modified. Raw training
checkpoints, six development reports, the aggregate `summary.json`, and all
seven independent chain-diagnostic JSON files are preserved with the
experiment archive.

**Next: M16l crop-cycle learning.** Test a targeted, *verified*
plant→harvest→deposit credit or crop-cycle curriculum, optionally with
learned subgoals, while keeping environment physics, legal actions and
the production model unchanged. Retain the immutable M4 and a matched
architecture control, and require exactly the same held-out food/manure/
safety gate before opening the blind audit. Avoid further untargeted
history-memory sweeps until planting and navigation are demonstrably
restored.
