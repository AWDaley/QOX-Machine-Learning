# M16c — Generalizable Herd Protection

**Status:** Research and evaluation infrastructure; experimental policy candidates NOT promoted.

## Why this experiment exists

M16b demonstrated that a nomad can learn to attack in short static encounters while losing migration performance. In the previous tasks the nomad could sometimes prevent attacks simply by blocking a predator. M16c creates a harder, more realistic objective: **move three cattle toward pasture while a predator approaches**. The controller must balance positioning, herding calls, self-preservation, and combat.

## World and policy contracts

- `nomad_convoy`: 18×16 world; three cows, one nomad, one ravager. The pasture destination, cattle positions, predator approach direction, nearby cover, and predator health vary with seeded initial conditions. The target is only disclosed to the nomad through the existing goal features; cows still rely on local vision and sound.
- `nomad_convoy_mixed`: alternating safe migration and threatened convoy episodes, including varied seeds, for curriculum learning. A predator is **not spawned mid-episode**; mixed episodes start with either zero or one. The mixed scenario is intentionally unsuitable for PettingZoo's stable `possible_agents` contract and is rejected by its adapter; it remains supported through `QoxCoreEnv` and the native PPO runner.
- Three **scripted** predator policies are used for variance: existing reference behavior, direct pursuit of locally visible cattle, or avoidance of a nearby defender. All are limited to ordinary world actions. This is not opponent self-play or learning.
- Cow v2 and nomad v2 checkpoints remain compatible; no physics or observation schema changes were made.

## Training and evaluation methodology

Use `qox train-herd --scenario nomad_convoy_mixed` to alternate PPO updates to all three cows (one shared policy) and the nomad (one shared policy), freezing the other species in each phase. The `--partner-mode sample` option varies the frozen partner's actions; it does not expose hidden opponent states to the learner. A separate `train-nomad` curriculum supports KL regularization (`--anchor-kl`) toward the original checkpoint to reduce forgetting.

Use `qox compare-protectors` to test baseline and candidate on **matched, unseen world seeds** in four conditions: predator-free migration, moving convoy, static Guard, and Intercept. It also tests candidate **without calls**, and **without attacks** where combat is possible. The report contains per-seed results, paired bootstrap intervals, model hashes, both pasture arrivals and individual-survival outcomes, and conservative promotion checks. No automatic checkpoint updates occur.

A 24-world test is an exploratory *validation* comparison. The promotion screen requires at least 48 held-out worlds, plus migration retention and a causal advantage from combat. Any candidate that passes must still be checked on a separate blind test-seed range.

## Initial M16c exploratory experiments

All numbers below are **per scenario, 24 evaluation seeds starting at 1,190,000 with 64 simulation ticks**. These are development/validation results, not a blind final result. Values are means, not percentages.

| Condition | Original pair | Joint PPO candidate (4,096 cow + 3,072 nomad decisions) | Anchored nomad candidate (2,048 decisions, baseline cow unchanged) |
| --- | ---: | ---: | ---: |
| Convoy cows reaching pasture (out of 3) | 0.58 | 0.42 | **0.75** |
| Convoy cows surviving (out of 3) | **1.92** | 1.88 | 1.79 |
| Convoy average cow health (scale 0–100; dead cows contribute zero) | 37.79 | **42.25** | 35.38 |
| Static Guard cows surviving (out of 3) | 2.33 | 2.33 | **3.00** |
| Intercept cows surviving (out of 3) | 2.25 | 2.25 | **2.46** |
| Predator-free pasture arrivals (out of 3) | 1.79 | 1.38 | **1.96** |

The joint PPO candidate improves average health in convoy worlds but harms migration and call-based herding. The anchored nomad is stronger at isolated Guard and Intercept, and moves more cows toward pasture, but has **worse convoy cow survival and health** than the original. Neither meets all gate criteria. Most importantly, the joint candidate's attack/no-attack comparison showed **no measurable cow-health benefit from its attacks** in the evaluated worlds.

**Do not replace the saved preferred cow/nomad checkpoints or interpret higher PPO rewards as proven generalization.** The existing preferred pair remains the baseline. The new candidates and benchmark JSON are retained for further experiments.

## Next research question

The convoys expose a policy tradeoff, not a software defect. Proposed next investigations are longer encounters with learned (rather than scripted) opponent policies, a better value function for long-term herd health, and separate multi-objective model selection. Test a range of training seeds and keep migration and defense metrics separately visible. Do not script cooperative defensive maneuvers as rewards or guaranteed effects.

## Acceptance and regression safeguards

1. New scenario worlds keep finite actors/resources, distinct IDs and deterministic seeded action-stream outcomes. The generic `qox replay` format does not serialize specialized scenario fixtures, so use the seeded environment or a full M15 world save for scenario continuation.
2. Predator behavior is selected from locally observed world state; no omniscient target or teleportation.
3. No-call/no-attack evaluations use matched starting seeds and identical frozen networks.
4. Legacy observation sizes, action schemas and checkpoint files remain valid.
5. Frozen model comparison never overwrites inputs, and reports source checksums and full per-seed outcomes.
6. Train-and-compare paths pass Python CI on 3.11/3.13 and the ML smoke job.
