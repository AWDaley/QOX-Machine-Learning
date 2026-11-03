# M16f — Farmer safety-weight and curriculum-budget experiments

**Status:** Sweep runner and paired evaluation implemented; the first six
M4-baseline candidates failed the development gate. **No model promoted.**

## Research question

M16d found a food-production versus survival tradeoff. M16e introduced four
staged curricula and safety-weighted PPO. M16f tests whether the safety
multiplier or training budget changes that tradeoff without reusing the audit
seeds to pick among many trials.

## Historical baseline

The archived M4 farmer policy trained for **32,768 decisions** is available
in the earlier QOX M4 and M16d experiment packages. Those independent
archives contain byte-for-byte identical checkpoints (SHA-256):

`5036afd395239049b5b99cef4aec7b99277c118eec705cb185fe3124a77938dd`

The repository does not include trained checkpoints. Restore this checkpoint
into your own `checkpoints/` or `runs/` directory before running an experiment.
Do not substitute an untrained random policy and call it the M4 baseline.

## Experiment design

1. Choose safety weights (e.g. 1×, 3×, 6× injury/death penalties) and stage
   budgets (e.g. 1,024 vs. 4,096 PPO decisions). Every combination starts
   from the **same unchanged baseline** and runs independent continuations
   for the requested training seeds.
2. Train agriculture, manure, refuge, mixed farmstead stages in sequence. The
   injury/death multiplier applies only to the last two training stages;
   physical mechanics and evaluation rewards remain unchanged.
3. Compare each final candidate with the baseline on matching development
   worlds: 48 safe + 48 threatened episodes, with candidate no-fertilizer
   and no-refuge action ablations. Record food, manure, crop damage, health,
   survival, confidence intervals, and per-seed outcomes.
4. Apply the M16e safety/production point-estimate gates. Among passing
   development candidates, pick the one with greatest combined food gain.
   Ties are settled by predetermined configuration and training-seed order.
5. If and only if a candidate passes, evaluate that **single preselected**
   policy on separate audit seeds, with scripted and random references added
   for interpretation. Never choose a different candidate in response to
   audit failure. At least three distinct training seeds and 48 episodes per
   evaluation scenario are required before manual review can be flagged.

All candidate checkpoint files and comparison reports remain separate from
the baseline and from the active Windows model set. There is no automatic
acceptance or online weight update in the visible world. The `audit.json`
file and `audit_references.json` only appear when development yields a
qualifier. The full sweep records paths, hashes, parameter grid, costs and
selection in `summary.json`.

## Command line

Estimate work before starting a large CPU run (no file creation):

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/sweep_full --dry-run
```

Run two survival settings using three matched training seeds:

```powershell
qox sweep-farmers --baseline checkpoints/baseline_farmer_M4_32k.pt --output-dir runs/sweep_1024 --weights 1 3 --stage-budgets 1024 --training-seeds 17 37 59 --development-episodes 48 --audit-episodes 48
```

The default 3×2 hyperparameter grid creates 72 stage-training runs and
roughly 184,320 PPO decisions before any evaluations. For inexpensive CI
or functional checks, use `--training-seeds 17 --stage-budgets 64
--worlds 2 --rollout 64 --epochs 1 --episode-cap 18
--development-episodes 2 --audit-episodes 2`; these tiny runs **cannot**
qualify for audit or manual promotion.

## First controlled M4-baseline sweep

The first run uses the verified M4 32k checkpoint, safety multipliers 1× and
3×, stage budget 1,024 decisions, three training seeds 17/37/59, 48
development worlds per scenario, and 192-tick evaluation horizons. This is a
controlled **short-budget** training study with full-horizon development,
not a claim that 1,024 decisions are sufficient for optimal farming.

### Development results (completed)

The immutable M4 checkpoint averaged **3.2849 food deposited** and **2.3790
manure applied** in safe farmsteads. In threatened farmsteads its survival
was **47/48 (97.9%)**, with mean final health **97.17**.

Every delta in the table is **candidate minus the same M4 baseline** across
48 matched development worlds per scenario; survival is in percentage points.

| Safety weight | Train seed | Safe food Δ | Safe manure Δ | Threat survival Δ | Development gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 1× | 17 | −1.3682 | −2.3234 | +2.08 pp | Rejected |
| 1× | 37 | −1.3682 | −2.3790 | +2.08 pp | Rejected |
| 1× | 59 | −0.3515 | −0.6280 | +2.08 pp | Rejected |
| 3× | 17 | −0.3095 | −2.3729 | 0 pp | Rejected |
| 3× | 37 | −0.3515 | +0.5125 | +2.08 pp | Rejected |
| 3× | 59 | −0.3515 | −0.7959 | +2.08 pp | Rejected |

All six candidates decreased safe-world food output. For example, the 3×,
seed-37 candidate had a paired 95% bootstrap interval of **[−0.6716,
−0.0562]** on safe food deposition. It was closest to the multi-objective
gate: its manure application, survival, threatened health and crop-damage
checks passed, but it deposited about 0.35 less safe-world food rather than
achieving the required increase of 0.5.

**Decision: 0/6 development qualifiers. No candidate selected, no audit
conducted, no weights installed.** The 2,600,000 audit seed range remains
unexamined for a future predeclared policy. Reports include each candidate's
per-seed outcomes, confidence intervals, checkpoint hashes and action
ablations; raw comparisons and checkpoints are archived separately from Git.

The evidence suggests that simple staged PPO with a larger injury/death
penalty is not sufficient to preserve baseline farming productivity at the
tested **1,024 decisions per stage**. We should next test interleaved rehearsal
of agriculture/manure tasks, a regularization anchor toward the M4 policy,
or stagewise early stopping—while keeping the audit range closed. A longer
training budget is another hypothesis, not an established improvement.

## Interpretation limits

- The selection gates use paired point estimates and seed-level bootstrap
  intervals; they are screening rules, not a proof of generalization.
- More configurations increase selection pressure on development seeds, but
  the one-time audit remains withheld until selection is complete.
- Farmers remain the only trained species in these worlds; cattle and
  ravagers are scripted environmental partners.
- Short training budgets, a small seed grid, and imperfect simulation
  environments can yield misleading negative or positive outcomes. Report
  results, including failures, rather than silently promoting a checkpoint.
