# M16l — Verified farmer crop-cycle credit

**Status:** Predeclared matched study; no automatic promotion. The original
M4 farmer remains installed unless a candidate passes unchanged food,
manure, survival, health, and crop-damage development and blind-audit gates.

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

*To be recorded only after the predeclared arms complete. Report every seed,
control vs treatment, physical farming metrics, diagnostics, all development
gate failures, and whether the reserved audit was ever opened. Never
replace M4 on training reward alone.*
