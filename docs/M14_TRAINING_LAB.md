# M14 — Neural-Network Training Laboratory

**Status:** Implemented desktop feature; learning performance is not guaranteed. The active simulation continues to use frozen neural-network weights. Training takes place in separate short curriculum worlds, not live inside the visualized ecosystem.

## Opening the lab

Start `QOX.exe` or `qox-desktop`, and choose **Open training lab**. Choose a species and a compatible scenario, then set PPO decisions, training seed, and number of held-out evaluation episodes.

The current checkpoint's observation schema determines which scenarios appear. Existing v1 60-input cows and v1 nomads cannot be trained on the call-dependent v2 scenarios; they are not silently upgraded. Fresh desktop cows and nomads are v2.

| Species | Curriculum scenarios |
| --- | --- |
| Cow | `forage`, `threat`, `nomad_herd`, `cow_signal` (last two require v2) |
| Farmer | `agriculture`, `refuge`, `manure`, `farmer_mixed` |
| Nomad | `nomad_signal` (v2), `nomad_herd`, `nomad_defend`, `nomad_mixed` |
| Scavenger | `scavenger_waste`, `scavenger_carrion`, `scavenger_competition`, `scavenger_mixed` |
| Ravager | `ravager_ambush`, `ravager_carcass`, `ravager_hibernate`, `ravager_mixed` |

## The candidate workflow

1. **Train candidate** starts a worker thread and sends rollout-level progress to the Tkinter window. The progress chart shows training episode return (green) and total PPO loss (purple), each on its own scaled axis. No running world or active model is modified.
2. **Cancel** stops at the next safe rollout boundary. An interrupted job never replaces the active checkpoint; a partially completed candidate is discarded. Cancellation is not instantaneous while an individual rollout is running.
3. When training finishes, QOX evaluates the **current model**, **candidate**, and **scripted reference** on identical held-out seed numbers, with frozen deterministic policies and a scenario-specific primary outcome. It reports mean difference and paired standard deviation. The report does not claim that small-sample differences are statistically significant.
4. Choose **Accept candidate** to install the new checkpoint. Choose **Reject candidate** to discard it. **Rollback last accepted** restores a previous active checkpoint for the selected species; the current desktop session keeps up to five undo snapshots per species.
5. Before accepting or rejecting, use **Export experiment JSON** to preserve training history, per-episode comparisons, and checkpoint fingerprints if desired.
6. Use **Save all five model networks (.qox)** to persist the accepted policy set. Undo snapshots and unaccepted candidates are temporary and are not included in `.qox` files.

The desktop's old immediate-replacement behavior was removed. All species now share the same explicit training→evaluation→accept/reject lifecycle.

## Metrics and limitations

Primary metrics are grounded in observed task outcomes where available. Examples: food grazed by cows, food deposited by farmers, cow pasture arrivals for nomads, actual waste/carrion consumption by scavengers, and charge hits or meat eaten for ravagers. Mixed scenarios currently use mean episode reward because no single physical outcome covers all tasks. The UI names this explicitly; reward improvement should not be confused with ecosystem survival.

**Training continuation:** If the checkpoint includes the Adam optimizer state, QOX restores it when training further. Training also preserves cumulative decision counts and the same observation/action contracts. A new run initializes its own seeded environments and training RNG, so it is not an exact mid-rollout resumption of the previous run. A fully resumable interruption system with RNG and rollout restoration is later work.

**Comparison limits:** Default comparison uses only eight held-out episodes for quick CPU feedback. Use more episodes and multiple training seeds for research conclusions. The scripted baseline is a reference, not proof that the learned policy is optimal; current cooperative/long-horizon benchmarks remain difficult. The trained candidate's reward history is not the held-out score.

**Safety:** `.qox` archives use the existing validated model format; the training lab never loads arbitrary Python pickle execution. Existing model-set saves remain supported. No change to ecological physics, rewards, world RNG, or the simulation model schemas is required by M14.

## Developer API

`qox.desktop_models.train_role(species, source, destination, steps=..., seed=..., scenario=..., progress_callback=..., cancel_requested=...)` writes a standalone candidate, reporting progress once per PPO rollout. It raises `TrainingCancelled` before a new rollout when cancellation is requested. Training without the hooks retains CLI compatibility.

`qox.training_lab.compare_checkpoints(...)` produces a matched held-out comparison report. `CheckpointHistory` implements the local, bounded rollback history and validates files before recording them.

## Automated validation

- Compatible scenario selection and v1/v2 filtering.
- Progress and safe cancellation for all five species.
- Optimizer continuation with increasing Adam update counters.
- Same-seed before/after/reference comparison with physically grounded task metrics.
- Explicit active checkpoint immutability until installation; rollback restores the exact previous bytes.
- Existing deterministic replay, physics, and model-set roundtrip tests remain unchanged.
- Windows build, frozen executable smoke test, and separate Python 3.11/3.13 CI checks.

## Recommended follow-ups

M14b could add optional periodic immutable checkpoint snapshots while training, resumable rollout/RNG state on cancellation, larger standardized benchmark suites, training graphs with separate actor/value/entropy losses, and persistent experiment-history exports. M15 will separately save and load an **entire world state**; `.qox` model sets still contain only networks.
