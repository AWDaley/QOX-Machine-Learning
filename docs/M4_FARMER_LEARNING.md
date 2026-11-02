# M4: Farmer production, fertilizer, refuge, and PPO

## Delivered functionality

M4 adds a **physical settlement refuge**, not a global invulnerability toggle:

- A farmer standing within one cardinal tile of **their own home settlement** may issue `ENTER_REFUGE` if it has available capacity. Entry costs a tick and stamina. Contention for a small refuge is resolved by a seeded lottery.
- Sheltered farmers occupy interior slots at the settlement's grid cell; they do not block the exterior entrance. No other species can enter. A sheltered farmer can wait, rest, eat/deposit to their home store, or move out to an unoccupied adjacent tile.
- Ravagers cannot attack sheltered farmers, but they can attack farmers outside and can disrupt production by remaining nearby. Refuge costs actual stored food via a configured ration; it does not create new food, and the store can be depleted.
- Cow grazing consumes crops when grass is absent and generates finite manure. Farmers can collect manure and apply it to improve soil fertility; decomposing waste also recycles some fertility. Cultivation still consumes seed grain and fertile soil.
- The scripted farmer baseline reacts to nearby ravagers by going home/entering shelter, and leaves shelter when it no longer detects a nearby threat.

## Observation and action contracts

**Cow policy checkpoint compatibility is preserved.** Cow and other legacy roles keep the existing version 1, 60-feature observation schema. Farmers receive 16 extra numeric features (76 total): shelter state, whether the entrance/store is immediately known, local store quantity and shelter occupancy, field/crop condition, visible ripe-crop direction and amount, visible manure direction and amount, and scaled distance to their known home. Remote settlement inventory is **not** magically disclosed. Farmer actions gain `ENTER_REFUGE`; physical movement handles exiting.

The PettingZoo adapter creates species-specific observation spaces. The `farmer_mixed` curriculum is deliberately restricted to the dependency-light trainer, because its population changes between resets. PettingZoo usage should select `agriculture`, `manure`, or `refuge` individually for stable `possible_agents`.

## Local setup (Windows PowerShell)

```powershell
python -m pip install -e ".[ml,dev,viewer]"
python -m pytest -q
qox inspect --species farmer --seed 42
qox run --ticks 200 --mode scripted --seed 42 --report runs/m4-world.json
qox viewer --mode scripted --seed 42
```

## Train a farmer

```powershell
# Train on a repeating curriculum (agriculture, refuge, manure), mixed by seed.
qox train-farmer --steps 16384 --seed 47 --scenario farmer_mixed --output runs/farmer.pt

# Compare on a held-out seed range; use the SAME scenario/seeds for every mode.
qox benchmark-farmer --checkpoint runs/farmer.pt --scenario agriculture --episodes 30 --seed 220000 --report runs/farmer-agriculture.json
qox benchmark-farmer --checkpoint runs/farmer.pt --scenario manure --episodes 30 --seed 220000 --report runs/farmer-manure.json
qox benchmark-farmer --checkpoint runs/farmer.pt --scenario refuge --episodes 30 --seed 220000 --report runs/farmer-refuge.json

# Use the frozen farmer policy among scripted non-farmers in the full world.
qox run --ticks 120 --mode learned-farmer --checkpoint runs/farmer.pt --trace runs/farmer-trace.json
qox replay --trace runs/farmer-trace.json
qox viewer --mode learned-farmer --checkpoint runs/farmer.pt
```

The observed benchmark is **not** evidence of independently learned festivals, nomad protection, or a stable ecology; those require later milestones.

## Baseline test (12 held-out seeds, seed_start=220000, 96 ticks, frozen 32,768-decision policy)

| Scenario / metric | Farmer PPO | Random legal actions | Scripted farmer |
| --- | ---: | ---: | ---: |
| Agriculture: mean deposited food | 3.40 | 0.47 | 12.00 |
| Agriculture: mean harvested crops | 2.20 | 0.90 | 17.64 |
| Manure: mean applied fertilizer | 2.25 | 0.68 | 11.92 |
| Refuge: mean final health | 100.0 | 0.0 | 100.0 |
| Refuge: mean sheltered ticks | 49.0 | 0.9 | 96.0 |

These are **local, seed- and curriculum-specific observations**, not guarantees. The policy did better than random but remained well below the scripted baseline in efficient agricultural production; after additional training, harvesting did not improve monotonically. Earlier 16,384-decision weights gave better harvest/deposit metrics on the same seeds. The model uses simple one-step observations and has limited long-horizon planning. In particular, fertilizer benefits are delayed and the experiment has not demonstrated reliable manure-driven farm management.

The full per-seed run report lives in local `runs/farmer-m4-32k-benchmark.json` when generated; temporary run data and model weights are ignored by Git. Always compare across multiple training and evaluation seeds; do not treat one checkpoint as conclusive.

## Rewards and validation

Farmers receive reward from **actual harvested food**, **actual deposited resources**, and a small, explicit manure-use shaping term. They are penalized for injuries/death; no positive reward is granted just for entering shelter. Physical ration consumption is independent of reward. Watch for the known limitation that depositing initial carried food can produce reward even without fresh harvest; report harvested biomass and store transfers separately when evaluating "learned farming".

Regression tests cover finite shelter capacity, same-tick attack immunity, physical exit, store consumption, cattle-induced crop damage, manure/fertility accounting, species-specific observation sizes, deterministic replay, malformed checkpoint rejection, and curriculum reproducibility. The pretrained M3 cow checkpoint retains its original 60-observation schema.

## Next research / implementation work

1. Improve the farmer learning curriculum with longer horizons, better accounting for **new production** versus transferred starting resources, and an ablation for fertilizer shaping.
2. Expand perception and introduce nomad-friendly/scary calls in M5. Test whether cows learn to pull toward friendly and push away from scary signals without direct movement scripts.
3. Eventually train multiple policies together and evaluate on random 40×40 maps rather than only short 12×12 curricula.
