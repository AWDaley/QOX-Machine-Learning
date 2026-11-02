# M5 follow-up: controlled learned cattle-call response

Status: **working isolated reinforcement-learning experiment; not full cooperative herding**.

## Problem with the first experiment

The initial M5 call-aware cow policy reached pasture in 50% of tested worlds whether the nomad called or not. That measured some navigation but **did not establish understanding of friendly versus scary calls**. In ordinary migration worlds, pasture, initial layout and visible vegetation can independently provide useful direction cues.

## New `cow_signal` task

A minimal *curriculum only* environment has one cow and one stationary scripted nomad in a 20×20 world. Each episode independently samples one of four cardinal directions and one of two signals. The caller stands slightly to the side of the cow's path. A **friendly** call indicates a hidden waypoint in the direction of the caller; a **scary** call indicates a waypoint away from the caller. The cow starts in the center and never sees a special marker or a high-biomass goal tile. Nomad calls are emitted as ordinary local sounds with real cooldowns, stamina consumption and attenuation.

The cow has the **same 66-input observation schema and action vocabulary** as the existing M5 cow-v2. No new neural input contains the hidden waypoint. Cow movement is never forced by a signal. Rewards use actual movement toward or away from the hidden waypoint plus the one-time arrival bonus, not a reward for taking a particular action or signal being present.

The task is deliberately limited: the hidden target is a *training waypoint*, not actual grass, and the nomad is scripted. Success shows learned response to a two-signal control language under simplified conditions; it does not show herd migration, pasture economics, intelligent nomad calling or predator defense.

## Paired counterfactual evaluation

Every evaluation mode uses the **same map/actor seed**. Only the nomad's emitted signal is manipulated:

- **Correct call:** the signal's semantics match the target.
- **No calls:** nomad never emits a friendly or scary signal.
- **Reversed call:** the same source emits the opposite signal while the hidden waypoint is unchanged.

Also evaluate random legal cow actions and a transparent scripted local-call responder. The flip counterfactual is especially important: it tests whether the trained policy *changes behavior* when the call meaning changes without changing map geometry. Keep all three in reports and review failures by signal type.

### Exploratory trained results (CPU, PPO)

Training: `train-cow --scenario cow_signal --cow-calls --steps 65536 --worlds 4 --rollout 256 --max-steps 24 --seed 513`.
Evaluation: 96 previously unseen seeds starting at 205000; frozen greedy cow-policy actions, 24 ticks/episode.

| Mode | Cows reaching hidden waypoint |
|---|---:|
| Learned, correct calls | **57/96 (59.4%)** |
| Same learned policy, no calls | 38/96 (39.6%) |
| Same learned policy, reversed calls | **0/96 (0.0%)** |
| Random legal actions | 17/96 (17.7%) |
| Scripted call responder | 75/96 (78.1%) |
| Scripted responder, no calls | 0/96 |
| Scripted responder, reversed calls | 0/96 |

The paired improvement of correct calls over no calls is **19/96 = 19.8 percentage points**. A provisional paired, seed-bootstrap 95% percentile interval (2,000 resamples, not adjusted for training/model-selection uncertainty) is approximately **11.5 to 28.1 points**. This supports signal sensitivity in this specific controlled curriculum. It is not proof of sophisticated cross-species cooperation or reliable generalization to unconstrained terrain. Some no-call episodes succeed, and the scripted baseline remains stronger. Repeat across training seeds and larger held-out layouts before making strong claims.

## Reproduce

```powershell
python -m pip install -e ".[dev,ml]"
qox train-cow --scenario cow_signal --cow-calls --steps 65536 --worlds 4 --rollout 256 --max-steps 24 --seed 513 --output runs/cow-signal.pt
qox benchmark-cow --scenario cow_signal --checkpoint runs/cow-signal.pt --episodes 96 --seed 205000 --max-steps 24 --report runs/cow-signal-benchmark.json
```

Evaluation reports contain each episode's seed, original intended call type, final distance to target and whether the cow arrived. The original M3 60-input cow and M5 66-input cow checkpoints remain unchanged in shape and loader logic.

## Follow-on work

1. Replicate across at least three independent training seeds and varied new map geometries, caller offsets and obstacles.
2. Transfer the trained response to cattle in realistic migration fields, and check paired correct/no/reversed-call differences there.
3. Train nomads to emit appropriate calls *and* position themselves, initially with fixed call-responsive cattle; benchmark against the scripted herder and random.
4. Add predator distractions only after a reliable herding benchmark has been achieved.
