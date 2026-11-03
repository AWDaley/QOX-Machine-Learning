# M16e — Staged farmer training with a survival-first policy gate

**Status:** Training and evaluation workflow implemented. No experimental
farmer weights promoted; a passing research outcome has not been established.

## Motivation

M16d increased food deposition but worsened manure use and predator survival.
This stage tests whether separate skills, followed by a mixed curriculum with
stronger penalties for *verified* injury and death, can preserve both health
and useful production.

## Method

The `train-farmer-staged` command warm-starts from a compatible 76-feature
farmer checkpoint and trains four sequential policies **per training seed**:

| Stage | Scenario | Default episode cap | Safety multiplier |
| --- | --- | ---: | ---: |
| 1 | `agriculture` | 96 ticks | 1× |
| 2 | `manure` | 96 ticks | 1× |
| 3 | `refuge` | 96 ticks | 3× |
| 4 | `farmstead_mixed` | 192 ticks | 3× |

The safety multiplier acts **only inside the PPO trainer**. Specifically,
the recorded environment reward is adjusted by
`(weight - 1) × (injury_penalty + death_penalty)`; these penalties come from
actual world outcomes. There is no reward for choosing refuge or any other
button, and no change to world physics, observations, action indices, old
checkpoint shape or normal evaluations. All other rewards are unchanged.
The `--survival-weight` option controls the multiplier; the regular
`train-farmer` command defaults to 1× to preserve legacy behavior.

Three distinct training seeds are used by default, and each stage's physical
worlds are seeded in a separate high-numbered training namespace (2B+).
Development comparisons start at 1,600,000 and independent audit comparisons
at 2,600,000; both cover safe and threatened farmsteads on matched seeds. The
command rejects overlapping development/audit ranges and reports the model
hashes, exact seeds, episodes and per-seed outcomes.

## Usage

Begin with an existing farmer policy, such as a trained M4 checkpoint:

```powershell
qox train-farmer-staged --baseline runs/baseline_farmer.pt --output-dir runs/m16e_001 --training-seeds 17 37 59 --stage-steps 4096 --development-episodes 48 --audit-episodes 48
```

Small CPU-friendly **smoke check only** (never eligible for promotion):

```powershell
qox train-farmer-staged --baseline runs/baseline_farmer.pt --output-dir runs/m16e_smoke --training-seeds 17 --stage-steps 64 --worlds 2 --rollout 64 --epochs 1 --episode-cap 18 --development-episodes 2 --audit-episodes 2
```

Use a **new** output directory for each run. The experiment keeps a checkpoint
and metrics file after every stage, plus `seed_*/development.json`. If a
development candidate passes all checks, only the strongest passing
candidate is evaluated in `audit.json`. `summary.json` gives the decision,
input/checkpoint paths and SHA-256 hashes; no baseline checkpoint is modified.
Failures leave earlier checkpoints for inspection rather than deleting them.

## Conservative selection and audit

1. Every training seed starts from the same baseline and gets distinct PPO
   action/world random seeds. Development evaluates **all** final candidates
   against that frozen baseline on the same safe and threatened worlds.
2. The M16d screen requires at least 48 episodes per scenario, higher safe
   food deposition, retained manure use, unchanged survival in both settings,
   and no material decline in threatened health.
3. M16e additionally requires preserved safe-world health, no material
   increase in cow crop damage, and retained food production under threat.
   These are preliminary *point-estimate* checks, not statistical proof.
4. Among passing development candidates only, sum the safe and threatened
   improvements in food deposited; select the best. No candidate passes means
   no audit takes place. We do **not** choose among candidates based on audit
   outcomes or retry a different candidate after an audit failure.
5. Run the same gates on the separate audit seed range. At least three
   distinct training seeds are required before labeling a successful audit
   `eligible_for_manual_review`. This is a **review flag**, not an automatic
   promotion. A larger independent replication is still necessary.

There is no automatic overwrite of the visible world's five trained network
weights or `.qox` model sets. The active world still uses frozen networks,
not online learning.

## What is and is not established

- The command now provides an end-to-end staged PPO workflow with multiple
  training seeds, immutable baseline, candidate checkpoints, matched tests,
  transparent rejection and a one-time separate audit.
- Injury/death penalties in PPO are stronger only for the two danger-related
  stages. This is a training hypothesis, **not evidence** that survival will
  improve.
- Default training/validation sizes are CPU-oriented, but 3 × 4 stages plus
  matched evaluations can still take time on a laptop.
- Scripted cows and predators remain the environmental partners. Actual
  independent multi-agent adaptation has not been demonstrated here.
- Passing heuristic thresholds or a limited audit should not be described as
  generalization or deployment readiness without further replication.

**Next experiment:** compare multiple survival multipliers and stage budgets
using new development seeds, then hold the audit seeds untouched for one
predeclared candidate. Preserve negative results and report any food/survival
tradeoff explicitly.
