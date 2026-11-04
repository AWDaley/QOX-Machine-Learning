# M16k — Actor-local farming task memory and action-chain diagnostics

**Status:** New opt-in policy/checkpoint contract and controlled study;
**no automatic model promotion**. The original M4 farmer stays installed
unless the predeclared full food/manure/safety development and blind-audit
gates are met.

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
remain loadable and usable. M16j's frozen M4 KL teacher sees only its
original version-1 observation channels.

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

*To be filled only after both predeclared arms finish. Report all six seeds,
task-chain diagnostics, gate status, and whether any audit was opened.*
