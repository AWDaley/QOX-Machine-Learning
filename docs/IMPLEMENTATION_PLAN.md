# QOX Machine Learning — Artificial-Life Ecosystem Implementation Plan

**Status:** Proposed greenfield plan  
**Repository:** https://github.com/AWDaley/QOX-Machine-Learning  
**Version:** 0.1 planning baseline  
**Primary goal:** Build a reproducible 2D multi-agent ecology in which five species use neural-network policies to learn survival, herding, production, predation, scavenging, cooperation, and festival decisions.

## 1. Product and research goals

QOX is an **artificial-life simulation**, not a game with predefined AI behavior trees. It must be possible to watch individuals, inspect their observations and action probabilities, train and checkpoint their policies, and compare trained behavior against scripted and random baselines.

Research questions:
1. Can cows learn local grazing, threat avoidance, and responses to friendly/scary nomad calls?
2. Can farmers learn when to cultivate, harvest, apply cattle manure, and retreat into settlements?
3. Can nomads learn to use two opposing cattle calls, protect herds, and intervene in attacks?
4. Can ravagers learn the sequence of stealthy approach, selective charge/attack, carcass relocation, feeding, and hibernation without a scripted strategy?
5. Do farmer-only, nomad-only, and combined festivals create meaningful incentives to build surplus and cooperate?
6. Does the five-species ecology persist in frozen-policy tests, or collapse under resource/predator pressure?

The goal is to make these **possible behaviors**, not guaranteed outcomes. Never hard-code the answer to an experimental question into movement, action-selection, or rewards.

### Non-goals for the first complete release

No 3D world, detailed graphics, language-model agents, online servers, trading currency, formal diplomacy, inter-settlement warfare, genetics/NEAT, mating choice, large-scale procedural construction, or player-controlled combat. These can be later experiments. Do not make Docker, PostgreSQL, or a GPU prerequisites.

## 2. Initial technical choices

| Layer | Initial choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Accessible, testable, ML ecosystem |
| World engine | Pure Python / NumPy | Deterministic headless stepping; independent of the viewer |
| Neural networks | PyTorch | Small CPU-trainable actor-critic networks |
| Environment interface | PettingZoo ParallelEnv | Simultaneous multi-agent decisions, standard observations/actions |
| Learning | PPO with species-level parameter sharing | Practical first reinforcement-learning baseline |
| Rendering | Pygame 2D viewer | Simple shapes, inspector, playback controls |
| Settings | YAML or TOML validated configuration | Reproducible scenarios without source edits |
| Tests | pytest + property/invariant tests | Detect ecology and training-interface errors |
| Lint/format | Ruff | Small maintenance burden |
| Data | CSV/JSONL + PyTorch checkpoints | Easy local inspection and comparison |
| Optional later scale-up | RLlib or vectorized custom trainer | Do not require large Ray dependencies for V1 |

The PettingZoo interface is an **adapter** over a simulator with its own clean step/reset API. Train headlessly; optionally replay a saved trajectory in the Pygame viewer. The viewer must never own authoritative state or influence simulation results.

Suggested initial scale: 32x32 or 40x40 grid; 20–70 actors; 2 farmer settlements; 1–2 nomad bands; a small number of ravagers. Use deterministic seed-controlled maps and settings. Treat every numeric balance value as an experimental parameter, not a scientific constant.

## 3. Core architecture

Suggested repository layout (create gradually):

    README.md
    pyproject.toml
    configs/
      default.yaml
      scenarios/
        grazing.yaml
        herding.yaml
        predator_ambush.yaml
        agriculture.yaml
        festival.yaml
        integrated.yaml
    docs/
      IMPLEMENTATION_PLAN.md
      WORLD_RULES.md
      OBSERVATIONS_ACTIONS.md
      EXPERIMENTS.md
    src/qox/
      core/
        world.py
        grid.py
        entities.py
        config.py
        rng.py
        actions.py
        observations.py
        events.py
        step_systems/
          movement.py
          resources.py
          agriculture.py
          combat.py
          perception.py
          signals.py
          refuge.py
          festivals.py
          reproduction.py
      envs/
        parallel_env.py
      agents/
        scripted.py
        random.py
      learning/
        networks.py
        ppo.py
        rollout.py
        checkpoints.py
        curriculum.py
      viewer/
        app.py
        overlays.py
        inspector.py
      evaluation/
        metrics.py
        scenarios.py
        report.py
      cli.py
    tests/
      unit/
      integration/
      learning/
      regression/

Design rules:
- **Single authoritative world state:** separate tile layers, entities, inventories, stores, group memberships, event sources, and episode metadata.
- **No omniscience:** each actor gets only locally perceptible fields, sound/smell signals, remembered information, and its own/group-known data. No exact locations of unseen enemies or global population counts.
- **Stable entity IDs:** log creation/death and preserve identifiers for event attribution. Introduce a bounded per-episode agent-slot scheme when births are added to the PettingZoo adapter.
- **World physics are deterministic given actions, initial state, and seed:** including random tie breaks.
- **Separate action legality from desirability:** physically impossible actions are masked; risky or wasteful actions remain selectable.
- **No global behavior scheduler:** do not script "herd cows west," "attack during festival," "defend farmer," or "retreat when weak." The network must decide.
- **Each species shares one policy initially**, but each individual has its own sensory inputs and, when implemented, recurrent memory. Group IDs are environmental relationships, not separate policies.

### Fixed-step update contract

At tick t:
1. Snapshot all current actor perceptions from state t (no access to future state).
2. Each live actor submits an action against that snapshot; absent actions use a documented safe fallback.
3. Validate masks, costs, cooldowns, and requested targets.
4. Resolve locomotion and collision **simultaneously**; use deterministic, seeded tie breaks rather than iteration-order advantages.
5. Resolve charge paths, combat, harvesting/grazing/milking, item carrying, shelter entry, and other interactions by defined priority rules.
6. Emit sound/smell/waste/trail changes; signals become visible at their documented tick boundary (prefer next tick).
7. Update metabolic costs, regeneration, crop growth, decomposition, spoilage, and timed festival/hibernation effects.
8. Apply births/deaths and world-boundary conditions, compute reward components, and emit a structured event ledger.
9. Return new observations and terminated/truncated states.

Choose one movement/combat tick duration, plus a separate slower ecological interval for plant growth, fertility, and reproduction. Declare both in configuration, and test against exploits caused by repeated actions inside one interval.

## 4. World and ecology specification

### Tiles and environment

Start with a square grid supporting: passability, terrain type, cover, grass biomass, crop biomass and stage, soil fertility, soil moisture (optional in early milestone), nutrient/dung deposits, carcasses, waste, and occupancy. Mark settlements and temporary nomad camps as bounded regions. Rivers, weather, seasons, and explicit paths are later features.

- Plants regrow based on fertility and a configurable rate. Grazing removes biomass.
- Planting requires viable land and an explicit seed/planting cost or a configurable abstraction.
- Crops grow over time, depend on fertility, can be harvested or eaten by cattle, and may spoil.
- Farmers can collect dung/waste, carry it, and spread it to increase available soil nutrients. Natural decomposition should also return some nutrients.
- Waste can simultaneously be food for scavengers and an input to farming; it is finite and cannot be duplicated by pickup/drop loops.
- Milk production depends on cattle condition and a cooldown; never treat every cow as a perpetual free milk generator. For a simplified model, define a subset of mature, milk-producing cattle; refine lactation/reproduction later.
- Carcasses have finite edible mass. Eating reduces mass; dragging moves the same entity rather than cloning it. Decay eventually removes the remainder.
- Apply explicit **energy/calorie accounting**. Nutrients recycled through manure are not new calories. Plant growth brings new energy into the system via an abstract sunlight source.
- Store all production, consumption, and loss in event records so resource conservation can be audited.

### Entity attributes

Common: ID, species, group ID (optional), position, facing, health, energy, stamina, age, maturity, carried items, action cooldowns, sensory range, and alive state. Species-specific attributes: cow milk/waste production, farmer settlement membership, nomad band/camp, ravager satiety and held carcass, scavenger digestion. Reproduction is a later staged feature.

### Settlements and refuge

Farmers may enter a home settlement through its entrance/radius if space permits. Inside, they are protected from direct ravager attacks, cannot work exterior fields, and may consume finite stored food. Outside siege pressure therefore has an economic effect. Nomads may be admitted as guests only in a later optional rule; cattle do not enter. Shelter is not teleportation: entry requires reaching the designated boundary. A threatened farmer must decide when to flee.

## 5. Five species: observations, actions, incentives

All observations are normalized numeric values with presence flags and fixed-size nearest-K neighbor slots or directional summaries. Missing detections are distinguishable from zero-distance detections. Distances are actor-relative, within sensory range. Every observation schema and action schema gets a version.

### Farmer

**Perceive:** health/energy/stamina, local crops and land fertility, carried resources, settlement direction and refuge capacity, last-known store status if available, nearby cattle/dung/scavengers, visible predators, recent alarms.

**Actions:** move, sprint, plant, tend, harvest, pick up/deliver crops, collect manure, fertilize, eat, rest, defensive attack, enter/exit refuge, alarm, contribute to festival, propose/start festival.

**Value:** viable food supply, surviving farmers, future crop productivity, safe shelter decisions, and festival success. Do not reward arbitrary repetitive fertilizing.

### Cow

**Perceive:** body state, nearby grass/crops, nearest visible cows/nomads/predators, local crowding, terrain, and **friendly/scary cattle call strengths plus relative directions**, each with detection flags.

**Actions:** walk in one of four directions, sprint, graze, rest, strike/kick at close range, issue distress alarm.

**Value:** actual nutrition, survival, reduced injury, eventual reproductive success. Calls must not forcibly control cow movement. Learning whether to approach friendly signals and flee scary signals is an experiment.

### Nomad

**Perceive:** body state, nearby cattle health and position, herd center/stragglers, available milk, camp or settlement direction, pasture quality, predators/defenders, local social signals, festival conditions when known.

**Actions:** move/sprint, collect milk, eat, rest, defend/attack, **friendly cattle call (pull)**, **scary cattle call (push)**, alarm, carry/deliver milk, festival contribute/propose/start.

**Value:** own survival plus herd survival/productivity, food availability, and nomad/combined festival success. Nomads are faster and better fighters than farmers, but a lone nomad is generally weaker than a healthy ravager.

**Signal rule:** friendly and scary calls have radius, intensity falloff, direction, duration, cost, and cooldown. Their effects appear in cow observations; never directly apply attraction/repulsion forces. Train and measure call understanding independently.

### Scavenger

**Perceive:** hunger, carcass/waste proximity, scent gradients, available edible mass, nearby competitors, visible predator/nomad risks, and cover.

**Actions:** move/sprint, consume dung/organic waste or carrion where eligible, rest, flee/dodge through movement, weak defensive attack.

**Value:** actual usable nutrition and survival. Scavengers cannot farm or hunt live prey in the initial release. Their competition with farmers for dung should be observable.

### Ravager

**Perceive:** energy/stamina/satiety, nearby prey positions and apparent vulnerability, defenders, known target detection status only where perceptible, cover, den suitability, carcass mass, holding state, festival noise/smell gradients, and charge-path obstruction.

**Actions:** walk, sprint, **sneak**, **charge in a chosen direction**, standard attack, grab/drop/drag carcass, eat, rest, hibernate.

**Combat rules:**
- Sneak is slower and costs more stamina than walking; it reduces visual/noise detection, **not guaranteed invisibility**. Scent and open terrain can still expose the ravager.
- Charge has a wind-up, short high-speed committed trajectory, sharply increased stamina plus caloric cost, limited turning, loud detection signal, collision/impact resolution, and a miss/recovery penalty.
- Multiple defenders can attack; no arbitrary "group strength bonus" is necessary if geometry and attacks are modeled fairly.
- Dragging a carcass reduces mobility and constrains combat; relocation can make feeding safer but is not mandated.
- Hibernate greatly reduces metabolism after feeding but limits responsiveness and leaves the animal vulnerable if approached.
- Ravagers have slow maturity/reproduction and high hunting costs; slow reproduction alone is **not** sufficient to guarantee ecological stability.

**Value:** usable meat intake, long-term survival, injury avoidance, and energy-efficient hunting. Do not give direct large rewards per kill independent of meat use.

## 6. Perception, signaling, and stealth

Visual sensing: occluded by obstacles/cover, distance-limited, species-dependent; a sneak modifies detectability based on distance, cover and motion. Hearing: direction, intensity and signal class (cow calls, alarm, festival sounds, charge roar), with falloff and optional attenuation. Smell: coarse gradients (carcass/waste/prey concentration/festival food) rather than exact location. Build perception into the engine, not the viewer.

Track **visible target**, **suspected threat**, and **last-seen target** as distinct concepts. Actors should never be told the ground-truth enemy awareness state without an observable cue. Initial version can omit explicit memory and supply last-observation age; recurrent networks may follow later.

A festival should generate spatial, time-limited sound/smell that ravagers can independently detect and investigate. **Do not spawn or teleport ravagers as a festival punishment.** Larger combined gatherings may emit stronger signals, but signals have ranges and ecological consequences.

## 7. Festival system and groups

Introduce farmer settlement IDs and nomad band IDs; individuals identify their own group. Initial inter-group relations: neutral by default; no war or sophisticated reputation model.

Three real, separate festival categories:

| Type | Typical inputs | Core challenge |
|---|---|---|
| Farmer-only | Stored crops, adequate reserve, farmer attendance | Agricultural surplus and settlement safety |
| Nomad-only | Stored milk, herd health, nomad attendance, temporary camp | Mobile herd productivity and protection |
| Combined | Farmer produce + nomad milk, both groups attending, cattle nearby | Cross-group logistics, cooperation, defense |

Mechanics:
1. **Survival stores and committed festival stores are distinct**, with explicit contribution actions and no duplicate claims to resources.
2. Each event requires a configurable minimum of genuine resources, attendance, and health/safety conditions.
3. Festival proposal/start is an action available only to eligible participants; a deterministic rules engine validates quorum, resources, and cooldown. Do not automatically start because a counter crossed a threshold.
4. A festival consumes or reserves resources once, persists for a specified duration, creates noise/smell/waste, and occupies some actors' time.
5. It succeeds only if completion conditions hold (or has a graded score for partial success); threats and failed attendance can end it.
6. Combined events offer a larger potential benefit but higher resource, coordination and predation costs; they must not automatically dominate mono-group events.
7. Prevent instant repeated festivals with cooldowns, material costs, attendance requirements, and per-event reward accounting.
8. Provide observable, **locally known** readiness information to participants. Avoid free global knowledge of remote stores or groups.

Train with short staged festivals first. A sparse festival-only reward is inadequate; combine real production/protection outcomes and a delayed festival bonus without directly rewarding scripted movements.

## 8. Reinforcement-learning architecture

### Initial policy model

One policy and value network per species; shared weights among same-species individuals. Start with a compact 2-layer MLP (e.g. 64 or 128 hidden units per layer). Each role has its own fixed observation schema and discrete action vocabulary. If actions combine a verb, a direction, or a target, encode a documented finite action index and a legal-action mask; avoid neural outputs that directly assign arbitrary world positions.

Training loop: PettingZoo ParallelEnv episodes -> per-role trajectories -> PPO loss (clipped policy objective, value loss, entropy) -> checkpoint -> deterministic validation. Add observation normalization, gradient clipping, reward scales, and seeded samplers. Start on CPU and profile before installing a large distributed stack.

Use **scripted baseline policies first** to validate the physics, and train one species at a time against fixed opponents. Then rotate opponents among saved checkpoints and scripted variants to prevent overfitting. Later train multiple species together; treat population changes as terminated agent trajectories rather than continuing a dead actor's recurrent state. Keep independent per-individual memory if recurrent policies are introduced.

### Reward principles

Reward verifiable outcomes rather than use of a move. Potential terms (all configurable and separately logged):
- Nutrition actually absorbed, useful harvest or milk production, survival, future reserves, healthy dependent animals.
- Group rewards for maintaining a viable local herd/settlement and completing a valid festival.
- Penalties for injury, starvation, wasted calories, and loss of dependent animals.
- Costs must arise primarily in the simulated world; avoid arbitrary duplicate negative reward for every stamina expenditure.

Avoid per-tick survival rewards that encourage endless inactivity and raw "kill count" rewards. Test reward hacking: cycling stored goods, repeated planting/harvesting without resources, tiny food actions, repeatedly proposing festivals, killing without eating, and manure duplication. Publish ablations with and without shaping.

### Evaluation contract

**Training and evaluation are different modes.** Evaluation freezes weights, disables exploration when configured, uses held-out seeds and fixed scenario budgets, and never makes silent interventions to save populations. Report mean and dispersion across seeds, plus raw event-level results. Include random and hand-written heuristic baselines and simple ablations.

## 9. Implementation milestones with acceptance gates

Do not start the next milestone until the named acceptance tests pass. Each milestone produces a working CLI command, tests, an example scenario, and an updated changelog.

### M0 — Repository foundation

Deliver: README, Python package, pyproject, developer setup, CI for pytest/Ruff, typed config, simulation seed handling, directory structure, CLI for headless and viewer modes, documented world units. Python + CPU only.

Accept: clean install on Windows/Linux, tests and lint pass, deterministic seed creation verified, no heavyweight services.

### M1 — Deterministic world and resource ecology

Deliver: grid/tile layers; grass/crops/fertility; waste, manure, milk, carcass mass; time progression; entity base; simultaneous movement/collision; scripted/random agents; early viewer; resource/event ledger.

Accept: same seed + same action stream => same state hash/event log; no negative inventories, no duplicated carcasses or waste, grass/crops regrow within configured bounds, renderer optional.

### M2 — Perception and multi-agent API

Deliver: PettingZoo ParallelEnv adapter; role-specific observation/action spaces; masks; visibility, sound and smell; inspectable observations, structured event logs; per-species baseline agents.

Accept: PettingZoo parallel_api_test where applicable, unseen actors excluded from observations, legal masks correct, predator stealth reduces detection only under defined conditions, no order-of-iteration combat advantage.

### M3 — Neural cow survival

Deliver: PyTorch role policy/value network, PPO rollouts/checkpoints, cow grazing and movement, scripted predator threat, reproducible train/evaluate commands, learning curves and baselines.

Accept: trained cow beats random baseline on held-out grazing and avoidance scenarios, not just a memorized map. Document failures and threshold before choosing final pass criteria.

### M4 — Farmer agriculture, manure, and refuge

Deliver: settlements with bounded shelter/capacity, planting/harvesting/storage, cattle crop consumption and waste, collection/spreading fertilizer, farmer policy.

Accept: healthy crops produce finite yields, manure boosts future fertility without creating energy, cattle create measurable positive and negative effects, farmers can reach refuge via legal movement, hiding prevents direct attacks but incurs opportunity/food costs.

### M5 — Nomad herding and defense

Deliver: nomad movement/combat/milking; friendly + scary cattle calls as perceptible signals; learned responses by cattle; band/camp identity; two-nomad defense scenarios.

Accept: signal direction/intensity are correct, calls consume resources and never directly move cows, trained policy outperforms a no-call/control baseline in a herding scenario, group defense works against at least one ravager under calibrated mechanics.

### M6 — Ravager hunting system

Deliver: sneak, sprint, charge with wind-up/recovery/committed path, target selection, carcass grabbing/dragging/eating, dens/hibernation and satiety, scavenging opportunities for leftover meat.

Accept: missed charges have meaningful costs, collision trace is deterministic, hidden ravager is not undetectable up close, carcasses remain finite while dragged, hibernation slows calories spent, trained hunting evaluated against stationary and evasive prey.

### M7 — Scavengers and integrated nutrient cycling

Deliver: scavenger policy, waste/carrion feeding, natural decomposition, competition with farmer manure gathering, scent gradients, cross-species accounting dashboards.

Accept: scavengers can survive without hunting live prey in appropriate seeded scenarios; fertilizer-vs-food competition is measurable; resources cannot be duplicated by scavenging/decomposition.

### M8 — Single-group and combined festivals

Deliver: farmer/nomad surplus commitment, proposal/start/attendance, festival duration, participant rewards, sound/smell attraction to existing ravagers, failure/interruption conditions, viewer overlays and metrics.

Accept: all three festival types can succeed in scripted feasibility tests; no festival can reuse the same stock; ravagers respond to a detectable attraction cue rather than spawning; combined rewards do not trivially override every solo choice.

### M9 — Reproduction, long-horizon training, and research evaluation

Deliver: configurable age/maturation/reproduction costs and species cooldowns; dynamic life/death and bounded agent slots; multi-species checkpoint rotation; long-run experiments; ablations and reports.

Accept: correct births/deaths in API and logs, no agent-ID reuse bugs, no uncontrolled population explosion from a resource exploit, multi-seed survival/festival statistics generated reproducibly, frozen-policy evaluation works after reload.

## 10. Experiments and dashboards

Prioritize these predefined experiments:
- Cows: learned grazing vs random movement; predator avoidance with/without alarm.
- Herding: no calls vs only friendly vs only scary vs both calls; test across unseen pasture layouts.
- Agriculture: cattle forbidden near farms vs unrestricted cows vs cows plus manure collection.
- Refuge: shelter enabled vs absent; measure farmer survival, food, and hiding time.
- Hunting: ravager without sneak, without charge, without hibernation, and full abilities; measure injuries, kill efficiency, prey persistence, hunger cycles.
- Festivals: farmer-only, nomad-only, combined, and no-festival baseline; measure preparations, attempts, completions, food reserves and predator attack rates.
- Cross-group cooperation: nomads near a settlement vs far away, with no hard-coded ally defense.

Viewer modes: pause, single-step, fast forward, display tile fertility/resources/cover, actor IDs, species, recent sound/scent fields, festival radius, and recent events. Clicking an actor shows its **actual network observation**, action mask, action probabilities, health/stamina/satiety, most recent reward components, and recent local trajectory. Viewer playback must be possible from logged action traces.

Core metrics: species populations, causes of death, calories/resources produced and consumed, soil fertility, cattle crop damage and manure deposition, farmer refuge occupancy/time, nomad calls and herd alignment, ravager charge hit/miss and spent stamina, stealth detection, dragging distance, hibernation proportion, scavenger food source, festival attempts/successes by type, and ecosystem lifetime. Include seed and checkpoint version in every report.

## 11. Risks, countermeasures, and research integrity

| Risk | Countermeasure |
|---|---|
| Reward hacking | Resource ledger, invariant tests, ablation studies, raw world outcomes |
| Ecosystem collapse before learning | Short curricula, fixed scripted opponents, staged birth/predation, multiple seeded worlds |
| Policies overfit a map or opponent | Held-out layouts, opponent-checkpoint rotation, evaluation-only seeds |
| Simultaneous action unfairness | Deterministic tick pipeline and collision/combat tests |
| Excessive complexity on a laptop | CPU-first 2x64 MLPs, headless training, modest initial population, profiling |
| Apparent but scripted emergence | Explicit behavior-vs-physics boundary; never force cow calls or predator tactics |
| Sparse long-horizon festival reward | Staged preparation scenarios, validated contribution accounting, logged component rewards |
| Changing population breaks RL adapter | M9 introduces bounded ID slots and explicit terminal transitions; conformance tests |
| Unclear failures | Per-agent inspector, event replay, fixed seeds, baseline comparisons |

## 12. Definition of a successful first integrated release

A user can clone the repository on an ordinary Windows PC, install Python dependencies, start a world, run it headlessly or view it, train or load five role policies, inspect an individual, and execute reproducible benchmark scenarios.

The integrated world contains functioning farmer agriculture/refuge, cattle grazing/waste/milk/crop damage, nomad friendly and scary calls and combat, scavenger waste/carrion feeding, ravager sneak/charge/drag/eat/hibernate, and three festival types. The system can report **whether** these actors learned effective behavior—without claiming they did simply because an attractive animation occurred.

### Recommended starting task

**Implement M0, then M1 without neural networks.** Make the world deterministic and the accounting correct before training. Use scripted control to validate every mechanic, then replace scripted decisions with learned policies according to the milestone plan. This minimizes the risk of attributing simulator bugs to machine learning.

### Reference interfaces

- PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/
- RLlib multi-agent policy mapping (optional future integration): https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
