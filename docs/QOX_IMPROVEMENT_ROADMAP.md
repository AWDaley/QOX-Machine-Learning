# QOX — Recommended Improvement Roadmap (M12–M18)

**Status:** M12–M15 merged; M16j–M16l studies complete, with no
unqualified experimental models promoted. M17–M18 remain planned.
**Repository:** https://github.com/AWDaley/QOX-Machine-Learning  
**Baseline:** M11 resizable pixel-art viewer, M10 desktop launcher, M0–M9 simulation and neural-network systems.  
**Objective:** Make QOX a readable, reproducible artificial-life laboratory that can demonstrate and measure genuine neural-network learning.

## Strategic assessment

QOX already has five species with neural policies, scripted baselines, deterministic simulation/replay, farming, manure cycling, herding calls, ravager combat, scavenging, festivals, reproduction, a Windows launcher, model-set saves, and a resizable sprite viewer. **Build on these capabilities; do not rewrite them.**

Four important limitations remain:

1. **Behavior is difficult to interpret.** The M11 viewer shows current states and badges, but actions and outcomes are not consistently differentiated. A displayed attempted attack, for example, is not proof that the attack hit.
2. **Training is disconnected from observation.** The Windows menu trains models in separate PPO scenarios; simply running the visible world does not update weights. `.qox` saves models, not the current world.
3. **Complex behaviors remain partial.** Learned cow–nomad herding trails its scripted reference; learned ravagers do not reliably move kills to safety; festival results largely come from prepared scenarios and sometimes scripted partners.
4. **Ecological stability is unproven.** M9 long runs experienced ravager extinction and later cattle decline; lifetime birth quotas and population caps can dominate the outcome.

**Recommended sequence:** M12 → M13 → M14/M15 → M16 → M17 → M18. Do the highest-value UI/diagnostics work first, then use the resulting measurements to improve the models and ecosystem.

| Phase | Priority | Deliverable |
|---|---|---|
| **M12: Event visibility** | First | Clearly visible charge, calls, attacks, births, feeding, festivals, status changes |
| **M13: Analytics and explanation** | High | Decision inspector, graphs, event feed, replay timeline |
| **M14: Neural learning laboratory** | High | Configurable training, progress, evaluation, model comparison and rollback |
| **M15: World save/load** | High | Exact in-progress simulation save, separate from model-set save |
| **M16: Joint behavior research** | Research | Better learned herding, farming, ravager sequence, festivals |
| **M17: Ecological calibration** | Research | Multi-seed stability tests, less artificial demographic bottlenecking |
| **M18: Optimization and release** | Later | Fast simulations, reliable Windows builds, accessibility and scaling |

---

## M12 — Make important events unmistakable

**User outcome:** Within one minute of watching QOX, a new observer can tell when a cow grazes, a farmer collects manure, a nomad signals, a ravager charges and hits/misses, an actor enters refuge, and a festival succeeds or fails.

**Implementation tasks**

1. Add a **read-only event-to-visual adapter** mapping confirmed world events to transient effects. Keep chosen action, legal action, executed action, and successful outcome distinct. For example, a failed attack should never play the same hit animation as an attack that dealt damage.
2. Extend, rather than replace, M11's generated species sprites and status badges. Farmers show planting/harvest/fertilizer/carrying/shelter; cows show grazing/alarm/milk/juvenile; nomads show friendly versus scary calls, dashing and defending; scavengers show waste versus carrion feeding; ravagers show sneaking, wind-up, charge, dragging, eating and hibernation.
3. Render short-lived **impact and status effects**: direction of charge, collision flash, health drop, crop harvest, waste collection, dragging tether, sleeping indicator, birth and death, successful and failed festival.
4. Preserve existing friendly/scary sound rings but make their color and behavior understandable through a permanent icon legend.
5. Add a collapsible **Recent Events** panel listing tick, species, actor and actual outcome. Selecting an event centers the camera on the source if still present, or the event location otherwise.
6. Add toggles for badges, combat effects, sound radii, health bars, group colors, fertile-land overlay and reduced-motion mode. Keep the default display uncluttered.

**Acceptance criteria:** No false-positive success animations; effect lists expire and remain bounded; sprite UI works at minimum window and fullscreen; all visual options preserve identical action traces/world hashes. Add headless UI-model tests plus Windows executable smoke validation.

**Likely files:** `src/qox/viewer.py`, `src/qox/sprites.py`, `src/qox/viewer_layout.py`, new `src/qox/viewer_events.py`, and tests.

## M13 — Decision inspector, metrics, and replay

**User outcome:** Select an actor and explain what it perceived, which options it considered, what it chose, what actually happened, and how those choices affected the ecosystem.

**Implementation tasks**

1. Expand inspection with group/home, life stage, relevant satiety/milk/carried inventory, last attempted and confirmed actions, reason an action failed, and selected target where observable.
2. Display **top five legal neural action probabilities**, action mask, and locally observed inputs. Add optional sensing/cover/scent overlays derived from the selected actor's observation, never from omniscient world coordinates. Training reward components appear only when a training/evaluation record supplies them.
3. Add sampled live charts: population by species/age, deaths by cause, stored food, crops/grass/fertility, dung/manure, carcasses, cattle losses, ravager charge hit rate, nomad calls, shelter time, and farmer/nomad/combined festival counts.
4. Add a bounded event stream with filters (species, actor ID, group, event, tick) and JSONL/CSV exports with seed, policy version, world settings and units.
5. Extend existing `Trace` replay with pause, speed control, tick scrub, jump to event, and periodic checkpoints for faster seeking. Keep recorded hashes for validation.
6. Add matched-seed, frozen-policy comparisons including scripted versus learned, no-calls versus both calls, and reproduction on versus off.

**Acceptance criteria:** An inspector shows exact corresponding observation values and valid masks; graphs and event filtering never alter simulation RNG; seek to a past tick and replay yields the same state hash as uninterrupted playback; telemetry memory is bounded on long runs.

**Likely files:** `src/qox/trace.py`, `src/qox/viewer.py`, `src/qox/perception.py`, new `telemetry.py`, `replay_viewer.py`, `experiment_reports.py`.

## M14 — Full desktop neural-network laboratory

**User outcome:** The Windows application can train, evaluate, compare, save and restore models safely without manual terminal commands.

**Implementation tasks**

1. Display five model cards: fresh/trained/imported, species, checkpoint schema, decisions trained, selected curriculum, last validation, and checkpoint name.
2. Add training scenario selection and staged progression: cows (grazing, call response, predators), farmers (farming, manure, refuge), nomads (single-cow call, herding, defense), scavengers (waste, carcass, competition), ravagers (ambush, dragging, hibernation), and festival-specific policies.
3. Show per-session decisions, finished episodes, return, task-specific outcome metrics, entropy/loss curves and evaluation scores separately. Do not confuse training reward with observed ecological competence.
4. Support stop/cancel, periodic **atomic** checkpoints and robust continuation, including optimizer and RNG state where required for genuine resumption; validate compatibility rather than just incrementing a training counter.
5. Automatically benchmark the old and candidate policies against the **same held-out seeds**, with greedy and sampled modes, random and scripted baselines, and uncertainty/seed-level outcomes. Accept/revert the candidate explicitly.
6. Add named experiment runs, saved comparisons, checkpoint version history and rollback; maintain legacy v1/v2 and separate festival policy action schemas.
7. Later optional **live training monitor:** a separately running headless worker trains with explicit scenario resets; the viewer can display a snapshot of the policy between episodes. Do not present this as weights being updated by the world on screen. True in-world online PPO is a separate research project requiring synchronized rollouts and reproducibility rules.

**Acceptance criteria:** Full Create → Train → Compare → Accept/Reject → Save `.qox` → Reopen workflow via Windows UI; cancellation cannot corrupt the active model set; CPU-only operation; meaningful user-visible errors and progress.

**Likely files:** `src/qox/desktop.py`, `src/qox/desktop_models.py`, `src/qox/learning/*`, `docs/DESKTOP_APP.md`.

## M15 — Save and resume an entire living ecosystem

**User outcome:** Save QOX during a cattle migration or festival, close the program, and reopen it at the exact same simulation point.

**Implementation tasks**

1. Keep `.qox` for neural-model bundles. Add a separate **versioned `.qoxworld`** state format (validated ZIP with JSON/array members), never arbitrary pickle deserialization.
2. Capture the complete state: configuration, seed, tick, all tiles/resources, actors and age/parent/group IDs, inventories, settlements and camps, carcasses and carriers, calls, festivals and escrow, cumulative demographics and ID counters, plus **all environment and policy RNG states**.
3. Bind a world to the correct neural model-set ID/checkpoint hashes. Support either a referenced compatible `.qox` or an explicitly embedded validated model set. Warn on mismatched weights.
4. Add **New World / Save World / Load World / Resume** to the launcher and viewer. Support safe rolling autosave, save names, preview metadata, and recovery after failed writes.
5. Keep deterministic action-trace playback distinct: a replay verifies historical actions; a world save restores a *mutable* world that can continue with new decisions.
6. Validate sizes, decompression bounds, schemas, version migration and atomic writes before activating a save.

**Acceptance criteria:** For cases with reproduction, active festival, live calls, and a dragged carcass, 200 uninterrupted ticks and 100 ticks + save/load + 100 ticks must match in state hash and events under the same action stream. A corrupt save does not overwrite current model sets.

**Likely files:** new `world_save.py`, `src/qox/world.py`, `src/qox/desktop.py`, plus persistence tests.

## M16 — Improve genuinely learned behaviors

**User outcome:** More convincing independent policy coordination, backed by held-out comparisons rather than scripted demonstrations.

**Research workstreams**

**Herding:** Jointly train learned cows and nomads with randomized grazing goals, terrain and predator approaches. Evaluate friendly-only, scary-only, both, none, and reversed calls. Measure arrivals, time, cow survival and cost of calling; improve beyond M5c's approximately 1.96/3 learned-pair baseline and demonstrate a causal advantage from correct calls over 100+ held-out episodes.

**Farmer planning:** Extend training beyond ready-made crop scenarios to planting, waiting, harvesting, collecting/applying manure, storing food and entering refuge. Measure net useful crops, fertility improvements, cow damage, hunger and time sheltered. Use cattle-present/absent and manure-enabled/disabled controls.

**Ravager hunting:** Use a curriculum from charge accuracy to carcass pickup, transporting a carcass away from defenders, feeding and hibernation. Reward *usable nutrition and surviving*, not the drag button. Measure meal efficiency, injury, successful relocation and post-meal survival. Test sneak/charge ablations.

**Festivals and community coordination:** Gradually replace scripted partner contributions with learned farmer and nomad actions. Move from prepared surplus to real crop/milk production, gathering, risk management and choosing farmer-only, nomad-only or combined celebrations. Keep scripted assistance explicitly opt-in and excluded from claims of learned cooperation.

**Method:** Fixed training/validation/test seed partitions; multiple training seeds; identical frozen policies in each ablation; scripted, random and previous checkpoint baselines; report sampled and greedy separately; include confidence intervals or seed-level dispersion. Small CPU models and staged scenarios remain the default.

**Acceptance criteria:** Publish reproducible improvements over prior learned policies for at least one major workstream before introducing additional world mechanics. Keep negative results in the research report.

## M17 — Ecological stability and population dynamics

**User outcome:** Quantify whether the five populations can persist and celebrate without caps or hard quotas artificially determining the answer.

**Implementation tasks**

1. Report why births and deaths occur: starvation, predation, injury, age, reproduction ineligibility, crowding, resource shortage, species cap and lifetime birth limit.
2. Run matched-seed tests at 1,200 and 3,000+ ticks for 20+ seeds, comparing birth-enabled/disabled and frozen learned/scripted policies separately.
3. Keep a configurable concurrent population cap as a compute safety limit; test replacement or removal of the **finite lifetime birth quota** in controlled experiments. Use real energy cost, maturity, cooldown and limited resources to regulate reproduction.
4. Calibrate plants, crop production, manure fertility recycling, cow energy/milk, ravager predation/hibernation, and maturation/lifespan **one parameter group at a time**.
5. Track extinction rate, cap-hit duration, population oscillations, resource production/consumption, predator success, and festivals per time unit. Provide sensitivity reports, not a single best run.
6. Add optional water, weather, seasons, richer vegetation and group trade **only after** stable baseline behavior is characterized.

**Acceptance criteria:** Zero resource duplication, no invisible population rescue mechanics, full parameter disclosure, and distributional survival/festival results across held-out seeds. Do not claim permanent equilibrium from finite runs.

## M18 — Performance, accessibility, and Windows shipping

**User outcome:** The simulator is fast, approachable, and reliable on an ordinary Windows laptop.

**Implementation tasks**

1. Decouple simulation ticks per second from rendering frame rate; add pause, 1×/2×/4×, fast-forward, and optionally run-until-event.
2. Profile rendering, collision and neighbor searches, repeated policy inference, tile aggregation, telemetry storage and event-list growth before optimizing.
3. Cache scaled sprite surfaces, avoid rendering invisible tiles, use bounded event buffers and consider local spatial indexes for actor perception when profiling justifies them.
4. Add map/population presets and multi-seed headless batch mode with progress reporting and CPU usage limits.
5. Add alternate color palette, UI scale, reduced motion, labels not reliant on color alone and remappable controls.
6. Release tagged Windows artifacts with checksums, release notes, packaged-executable smoke checks and save/checkpoint compatibility notes.

**Acceptance criteria:** Instrument actual simulation TPS, render FPS and memory on a specified CPU-only machine; run a 10,000-tick headless stress test without invalid states or growing unbounded effects; pass deterministic replay and Windows launch smoke tests.

---

## Pull-request order

1. **M12-PR1:** Read-only executed-outcome event bridge, deterministic tests and sprite-status semantics.
2. **M12-PR2:** Behavior effects, recent-event log, overlay toggles, viewer/window tests.
3. **M13-PR1:** Telemetry, actor perception/probability inspector, ecosystem graphs.
4. **M13-PR2:** Seekable replay, event jump, research exports.
5. **M14-PR1:** Training dashboard, evaluation, checkpoint rollback and safe resumption.
6. **M15-PR1:** Versioned world save/load, exact continuation tests and desktop controls.
7. **M16 research PRs:** One independently benchmarked learned-behavior improvement at a time.
8. **M17/M18:** Population-calibration experiments, profiling, accessibility and Windows release hardening.

All PRs should include small acceptance tests, no accidental feature/action shape changes to legacy checkpoints, reproducible seeds and evidence from GitHub CI. Do not merge on failing CI.

## Definition of success

A user can launch QOX, recognize exactly what the actors are doing, inspect what each network perceived and chose, follow events and population charts, train and compare candidate models, save both models and world separately, replay a failure, and run reproducible long-term stability studies across different seeds.

**Current recommended follow-up:** M16h interleaved on-policy training was
more consistent than its sequential control on food and threatened health,
but all six policies failed M4's farming/manure gate. M16i tested finite
manure-task training credit (0, 0.3 and 1.0 per applied unit, plus one-fifth
per collected unit) with no physics/reward change during evaluation. All
**nine** new candidates failed the same food/manure/safety development gate;
stronger credit improved manure use *during training* but hurt held-out food
and did not consistently improve held-out fertilization. See
[M16i's completed controlled study](M16I_MANURE_CREDIT.md). M16j tested
selective frozen-M4 food-skill KL on **current** on-policy PPO states, with
M16i's manure credit fixed at 0.3 and a zero-KL control. Strong KL 2.0
reliably increased real held-out manure use across three training seeds
(mean 3.77 versus M4's 2.15), but all candidates continued to lag M4's
food delivery (KL-2.0 mean 2.91 versus 3.82). **0/9 qualified**; the audit
was never opened and M4 remains installed. See
[M16j's complete results](M16J_ON_POLICY_FOOD_ANCHOR.md). The next focused
M16k experiment compared actor-local history of verified
plant/harvest/fertilize/deposit events with a same-architecture masked
control across six matched trained policies (96 training stages). The
event-memory arm descriptively preserved more food and threatened health,
but **0/6 passed** the food-development gate. Independent action-chain
diagnostics showed **zero chosen planting actions across all six** despite
legal planting opportunities; the original M4 plants regularly. No audit
was opened or model promoted. The next M16l hypothesis should target
verified crop-cycle credit and planting/navigation skill rather than
untargeted input-memory tuning. M16l implements finite **training-only**
plant→own-harvest→deposit event credit with two equal-budget arms and a
separate held-out block. In the completed six-policy run, physical
planting events increased during stochastic training (523 versus 432 for
controls) yet **no candidate planted in deterministic evaluation**.
All six failed the M4 food gate, and the treatment also degraded manure;
no audit opened and the original M4 remains untouched. M16m should
focus on the action-ranking and dedicated planting-curriculum failure
rather than increasing crop-credit strength alone.
See [M16k's full study](M16K_FARMER_TASK_MEMORY.md) and
[M16l's predeclared protocol](M16L_VERIFIED_CROP_CYCLE.md).
Then finish M16 replication with
independent partner policies and begin M17
multi-seed ecological calibration before shipping a performance-focused M18
release. The original M12 event-visibility foundation is already merged.
