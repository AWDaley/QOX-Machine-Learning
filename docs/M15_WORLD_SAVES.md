# M15 — Save and Resume an Entire QOX World

## What this adds

A `.qoxworld` save contains the complete mutable QOX ecosystem: terrain fertility and crops, actors, age and births, settlements/camps and inventories, milk/dung and carcasses, active calls and festivals, recent event history, stable ID counters, and **both world-physics and policy-sampling PRNG states**. A load continues without regenerating terrain or changing a random stream.

The existing `.qox` format still contains **only the five neural-network weights**. It is not a world save. A `.qoxworld` file intentionally does not contain executable model weights or pickle data. A learned world's save contains cryptographic fingerprints for the precise model checkpoints it used; loading the world requires those same five model weights. Save a `.qox` **model set first** so that those weights are available on a future launch.

## Windows desktop workflow

1. In the launcher, use **Save model set** to save your five networks as `my-agents.qox` (especially important if any are freshly generated or trained).
2. Click **Start simulation** (neural) or **Scripted simulation**.
3. While watching the world, press **F5** to quick-save to `Documents/QOX/Worlds/seed<seed>_tick<tick>.qoxworld`. Or press **Ctrl+S** to choose a custom filename.
4. Exit the viewer. The original simulation state remains saved on disk.
5. On a later launch, load `my-agents.qox` with **Load model set** if your saved world used neural networks. Choose **File > Open saved world...** to load the `.qoxworld` file and resume. The resumed viewer starts **paused**; press Space to continue.
6. A policy mismatch or incomplete save results in an error; the currently active neural networks are not overwritten.

**Replay vs. resume:** A `.json` trace replays an existing sequence of past actions; a `.qoxworld` is a mutable simulation that can continue generating new actions. Replay mode cannot be saved as a mutable world through the F5 shortcut.

## Headless reproducible research

For scripted, random, or idle populations:

```powershell
qox run --mode scripted --ticks 200 --reproduction --festivals --save-world runs/after200.qoxworld
qox resume-world --input runs/after200.qoxworld --ticks 300 --output runs/after500.qoxworld
qox viewer --load-world runs/after500.qoxworld
```

To resume a *learned five-policy* world, use the Windows desktop File menu and its matching `.qox` model set. This prevents continuing under a different set of neural weights and accidentally treating the result as the same experiment.

## Data format and security

`world.json` and `manifest.json` are the only entries in a compressed `.qoxworld` ZIP. All state uses JSON, **not pickle**. The manifest records format version, mode, seed, tick, state hash, SHA-256 of the world payload, model SHA-256 fingerprints and controller sampling options. Unknown formats, altered data, unexpected ZIP entries, oversized payloads, invalid simulation states and mismatched policies are refused.

Writes are atomic: a failure while writing or validating does not replace a previous successful world save. The world serializer calls `World.__new__` and directly restores all state including both PRNGs, rather than running the world's random initialization again.

### Scope and limitations

- Saved worlds capture the complete **QOX world event log** and world state but not the GUI camera position, panels, telemetry chart sampling, or a running training optimizer. These are presentation/training states, not determinants of the simulation.
- The viewer does **not** train a policy while playing. Model weights are frozen, including after resuming the world.
- For the moment, Save World supports the desktop's **five-neural-network simulation** and standard scripted/random/idle runs. Single-species training scenarios remain separate model evaluation workflows.
- Large worlds with very long event histories may reach the 64 MB uncompressed save limit. This is a deliberate loading-safety constraint and can be revisited alongside M18 event-history optimization.

## Tests and acceptance

- Same configuration and action stream: 30 ticks + save/load + 40 ticks matches uninterrupted 70-tick run for scripted, random and idle actors, with births/festivals both off and on.
- Persist and continue active festival, pending ravager charge, dragged carcass, live sound wave, birth counters, stored food and event history.
- Five-network checkpoint SHA-256 matching; changed or missing weights must reject a saved learned world.
- Corrupt payload, unexpected archive members, incorrect version and inconsistent IDs fail without mutating existing saves.
- Headless CLI smoke test and Windows Pygame F5 save smoke test, plus a packaged-app smoke test that resumes live multi-policy inference.
