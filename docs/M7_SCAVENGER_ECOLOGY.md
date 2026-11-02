# M7: Scavenger learning and nutrient cycling

**Status:** Implemented and locally validated. This is a research prototype, not a demonstrated stable long-term ecosystem. The report preserves negative controls, scripted baselines, and distinction between sampled and greedy policies.

## Ecology and conservation

- Scavengers eat real, finite tile waste (`eat_waste`) and unclaimed carcass meat (`eat_carcass`). They cannot eat a carcass a ravager is currently dragging.
- On consumption, only **12%** of eaten matter is returned to `dung_pending`. At a later ecology tick it becomes `scavenger_residue`, subject to tile waste capacity. Re-eating this residue yields at most another 12%, so repeated digestion cannot create infinite material. Scavenging remains calorically useful, but nutrients are not equated with calories.
- Passive tile-waste decomposition returns finite fertility (`0.015` per unit of decayed waste). Carcass decay removes meat (0.04 per ecology update) and returns fertility locally at a configurable coefficient (`carcass_fertility_return=0.015`). Overflow above soil fertility 1.0 is lost; no food or waste is created by decay.
- Farmers and scavengers act on the **same tile waste inventory**: a farmer's `collect_waste` competes directly with the scavenger's `eat_waste`. Farmers can use collected waste to fertilize fields. Cow dung provides a finite input when cows graze.
- The event log distinguishes `eat_waste`, `scavenger_carrion`, `scavenger_residue`, `collect_waste`, `fertilize`, and `carcass_decay`. New world-summary fields report material flows, not invented resource balances. The existing `eat_carcass` event retains its original carcass-id detail format.

## Sensory and policy contract

Scavengers use the original 60 local features, plus **11 species-specific features**, for a 71-input neural policy. Additional inputs describe whether an edible carcass is on their tile; coarse direction/strength of odor from *unclaimed* carcasses within Manhattan distance 7; a visible nearby waste patch within vision/LOS range 4; and visible farmer, scavenger, and ravager counts. Other species' observation schemas remain unchanged, so previous cow/farmer/nomad/ravager checkpoints still load.

The action vocabulary stays simple: wait, move N/E/S/W, rest, eat local waste, or eat an unclaimed carcass at the actor's tile. The world supplies no magical 'seek food' action or distant exact position.

## Three learnable scenarios

| Scenario | Starting conditions | Learning objective |
|---|---|---|
| `scavenger_waste` | One scavenger, finite dung/waste patch in a random cardinal direction and at varying distance | Follow local signals and consume waste |
| `scavenger_carrion` | One scavenger, finite unclaimed carcass in a random cardinal direction | Follow carcass odor and consume meat |
| `scavenger_competition` | One scavenger, one scripted farmer, one cow and shared waste | Compete with farmer manure collection without conjuring extra matter |

`scavenger_mixed` rotates the three scenarios with seed-controlled population counts; the PettingZoo adapter correctly rejects this changing-population curriculum, but supports all fixed scenarios. Each scenario begins with **38 energy**, finite food, and normal metabolism. Training rewards primarily reflect real, useful intake. A small signed distance-potential term rewards getting closer to already perceptible resources; walking back costs the same amount, preventing back-and-forth farming. Policies share weights among individuals of the same species.

## Training and evaluation

Training used `scavenger_mixed` in two stages: **16,384 PPO decisions with seed 711**, then **32,768 additional decisions initialized from that checkpoint with seed 712** (49,152 total decisions). The 16,384-decision checkpoint learned useful stochastic foraging but its greedy policy mostly rested; we retained the longer-trained checkpoint as the stronger candidate.

The final evaluation used **48 held-out seeds starting at 954000** with 48 ticks per episode, separately for each fixed scenario. All baselines use identical world-seed sequences and physics, with frozen policies and no rescue interventions. The trained policy was never updated on these test worlds.

### Held-out feeding results (units consumed per episode; higher is better)

| Scenario / mode | Greedy learned | Sampled learned | Random | Scripted |
|---|---:|---:|---:|---:|
| Waste, waste consumed | **5.608** | 5.380 | 0.169 | 5.571 |
| Carrion, carcass meat consumed | 8.247 | **9.345** | 0.333 | 9.960 |
| Farmer competition, waste consumed | **7.784** | 5.593 | 0.068 | 7.438 |

The relevant food-source ablation suppresses consumption: without `EAT`, the waste and competition policies consume zero waste; without `EAT_CARCASS`, carcass consumption is zero. A no-carrion ablation leaves waste feeding unchanged, which guards against an incorrectly implemented mask.

### Farmer competition is not automatically win-win

In the farmer competition scenario, the greedy learned scavenger consumes **7.784** waste units per episode, and farmers collect **1.142** and apply **0.917** units of fertilizer on average. With the scavenger's `EAT` action disabled, farmers collect **4.913** and apply **4.125**. This demonstrates **resource competition**: scavengers gain food, but farmers lose some fertilizer opportunity. Natural/animal nutrient recycling partly replenishes material; it does not cancel the short-term tradeoff.

### Scientific limits

- One-to-three-actor curricula are intentionally simpler than the full world. Results do not show scavengers reliably following live ravagers, escaping predators, or stabilizing a 5-species ecosystem.
- The scripted scavenger remains a strong comparator for reliable carcass consumption. The greedy learned carcass policy is behind it, although sampled behavior approaches it.
- Because episodes have abundant finite food and start at 38 energy, 48-tick survival is not evidence of sustainability over generations. Reproduction and population-level cycles are M9.
- Nutrient cycling is an **abstract, bounded fertility model**, not mass- and energy-accurate real-world soil chemistry.
- Statistical uncertainty across multiple training seeds has not yet been established; avoid generalizing from one PPO seed pair.

## Commands

```powershell
python -m pip install -e ".[dev,ml,viewer]"
qox train-scavenger --scenario scavenger_mixed --steps 16384 --seed 711 --output runs/scavenger_stage1.pt
qox train-scavenger --scenario scavenger_mixed --steps 32768 --seed 712 --init-checkpoint runs/scavenger_stage1.pt --output runs/scavenger_selected.pt
qox benchmark-scavenger --scenario scavenger_competition --checkpoint runs/scavenger_selected.pt --episodes 48 --seed 954000 --max-steps 48 --report runs/scavenger_eval.json
qox viewer --mode learned-scavenger --checkpoint runs/scavenger_selected.pt
qox run --ticks 100 --mode learned-scavenger --checkpoint runs/scavenger_selected.pt --trace runs/scavenger_trace.json
qox replay --trace runs/scavenger_trace.json
```

## Suggested follow-ups

1. Stress-test fairness in farmer/scavenger competition across resource scarcity, starting distances, and actor ratios.
2. Add a separate scavenger-vs-active-ravager danger curriculum rather than inferring predator avoidance from food collection.
3. Consider making carcass decay, manure nutrient values, and soil fertility into separate conserved quantities before claiming physical ecology realism.
4. Proceed to M8 only after CI is green; M8 will introduce farmer-only, nomad-only and combined festival eligibility, resource commitment, attendance, sound/smell attraction, interruptions, and event accounting.
