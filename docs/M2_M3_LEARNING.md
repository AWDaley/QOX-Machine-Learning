# M2–M3 Implementation: Observation, PettingZoo, Cow PPO

## Status and boundaries

M2 and M3 provide **one trainable cow policy**, not five learned species. Farming/nomad/scavenger decisions in the integrated world remain scripted. The first predator action is a simple adjacent attack used for early avoidance tests; stealth/charge/drag/hibernation will arrive later. PettingZoo is an optional adapter over a dependency-light core environment. It does not own physics or RNG.

## Observation contract (v1)

`qox.perception.FEATURES` contains **60** stable, named float32 features in the range [-1,1]: the actor's body/role resources, current tile, four adjacent tiles' grass/crops/passability, nearest visible food, species-relative neighbor sightings, visible carcass, known home (only for farmers), local population counts, waste-scent gradient, and recent auditory direction/intensity. Vision is limited to Manhattan radius 4 and blocks through intermediate impassable/opaque tiles. Scent can diffuse behind visual obstructions. Hearing uses event coordinates recorded at emission time and the previous tick's emitted noise; it does not transmit arbitrary actor IDs or hidden global positions.

Every species has its own finite, versioned action list. For cows: `wait`, four cardinal `move`s, `rest`, `graze`. Each observation includes an `int8 action_mask`: physically unavailable actions such as grazing on an empty tile are masked, while risky moves remain legal. The PyTorch policy masks logits before sampling; invalid external actions in the core are converted into `wait` and marked `invalid_action` in infos.

The optional PettingZoo `QoxParallelEnv` uses a `gymnasium.spaces.Dict` observation (`observation`: Box float32; `action_mask`: MultiBinary) and per-agent Discrete action spaces. Agent IDs and possible_agents are fixed during these milestones; births and bounded slot reuse are future work. `reset()` returns `(observations, infos)`; `step()` returns `(observations, rewards, terminations, truncations, infos)`. Dead/truncated agents are removed from the next active agent set. Scripted opponents are enabled in the dependency-light core training environment, not the public PettingZoo adapter by default.

## Forage scenario and objective

- 12×12 map, one cow, no other actors, seed-controlled scattered grass patches.
- Cow metabolic consumption is **0.7 energy/tick** instead of the main-world default **0.045/tick**. The accelerated rate makes seeking food important in a short training run.
- Each cow decision receives finite nutrition reward: `0.45 × useful grass units actually absorbed`, with a small per-tick cost, injuries penalized, and an additional death penalty. Consumption at a full energy bar is *not* rewarded. The policy is never rewarded directly for the `graze` action unless it yields usable food.
- State includes local grass signals; training requires selecting when to graze and where to move. A separate `threat` scenario adds scripted predator pressure, but this is not yet a full escape-learning benchmark.

## PPO training implementation

One two-hidden-layer MLP (default hidden size 64) outputs masked discrete action logits and a scalar value. Train with parallel headless worlds, experience rollout, GAE, clipped importance-ratio policy objective, clipped gradient norms, state-value loss, and entropy regularization. Episodes restart on cow death or time limit; bootstrap values are masked across episode boundaries. Random seeds are fixed and recorded. Training does not require PettingZoo, although it uses the same core world transitions as the adapter.

Commands:

```bash
qox train-cow --steps 65536 --seed 47 --scenario forage --output runs/cow.pt
qox benchmark-cow --checkpoint runs/cow.pt --episodes 30 --seed 99999 --report runs/benchmark.json
qox viewer --mode learned-cow --checkpoint runs/cow.pt
```

**One local benchmark run, not a universal performance claim:** training seed 47, 65,536 cow decisions, 30 previously unused evaluation seeds starting at 99,999, 96 steps/episode, policy evaluated greedily. The trained policy was substantially better than random but below a hand-coded scripted baseline.

| Evaluation mode | Mean reward | Mean grass/crop consumed | Mean final cow energy | Survival |
|---|---:|---:|---:|---:|
| Frozen trained PPO | 12.0513 | 31.1338 | 65.4081 | 100% |
| Uniform random legal actions | 3.6262 | 8.6983 | 19.9542 | 100% |
| Scripted greedy forager | 18.0726 | 90.2373 | 97.6666 | 100% |

All cows survived this short evaluation; **survival is not a discriminating outcome in this benchmark**. Energy retained, actual grazing, reward and longer episodes should be compared. The scripted baseline is an optimistic hand-coded local heuristic, so comparisons should distinguish it from the uniformly random baseline. Results will vary with seeds; repeat training with multiple seeds and confidence intervals before making scientific claims.

## Acceptance tests

- All species produce fixed-size bounded observations and valid masks; sight occlusion and distance limits verified.
- Core reset and step track dynamic active-agent lists, death, truncation and deterministic seeded states.
- Minimal contact attack causes real injury (not only a shaped reward).
- Optional `pettingzoo.test.parallel_api_test` checks the installed adapter in CI.
- PyTorch policy only samples allowed actions; smoke train → save → load → evaluate is reproducible.
- Frozen cow inference can be used in the main headless viewer and recorded/replayed without changing world RNG.

## Next gate

M4: farmer refuge, field/settlement decision making, manure collection improvements and farmer policy. Keep run traces and replay tests as mandatory gates.
