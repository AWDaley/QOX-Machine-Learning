# M13 — Live ecosystem analytics, neural inspection, and seekable replay

**Status:** Implemented research UI, not a change to the simulator's physical rules.

M13 builds on the M11 resizable viewer and M12 confirmed-event observer. It adds read-only time-series telemetry, an actor decision snapshot showing exactly what a neural policy was allowed to observe **before choosing its action**, and a seekable action-trace replay.

## Desktop / viewer controls

| Control | Action |
| --- | --- |
| `1` | Population and selected-actor overview |
| `2` or `I` | Neural action inspector for a selected actor |
| `3` or `G` | Live ecosystem graphs |
| `J` / `K` | Scroll observed input features up/down |
| `X` | Export the retained telemetry window as JSON + CSV to `~/Documents/QOX/` |
| `E`, `T`, `V`, `H` | Previous M12 events, major/all events, effects, and badges |
| `B` / `Y` | Cycle event-feed species filter / filter to the selected actor |
| `Space`, `N`, mouse, WASD, wheel | Pause, step, select, pan and zoom |

The Windows desktop menu's **File** menu adds *Open recorded world replay…* and *Record next simulation as trace*. The recorded `.json` trace is saved in the user's Documents/QOX folder after the simulator window closes. These traces are **not** resumable world saves, which are planned for M15.

### Replay controls

In replay mode, the simulation starts paused; it never calls the active neural networks or scripted policies to choose new actions.

| Control | Action |
| --- | --- |
| `Space` | Play/pause recorded actions |
| `,` / `.` | Previous/next tick |
| `PageUp` / `PageDown` | Jump back/forward 25 ticks |
| `Home` / `End` | Beginning/end of the trace |
| `[` / `]` | Previous/next major event (may index a long trace) |
| Click the thin timeline | Seek directly to a tick |

The replay UI always verifies each processed frame against its recorded state hash. In-memory deep-copy checkpoints accelerate backward and forward seeks, with a limited cache. The viewer's event effects and telemetry are reset on nonsequential jumps to avoid displaying events that did not occur in the current playback window. The underlying world is faithfully reconstructed. Recorded input actions are displayed; no active policy is reconstructed from the trace.

```powershell
# Make a deterministic trace from a headless run
qox run --seed 42 --ticks 500 --mode scripted --reproduction --festivals --trace runs/experiment.json

# Inspect it with pause, timeline, and event navigation
qox replay-viewer --trace runs/experiment.json

# Record a new trace while using the regular interactive viewer
qox viewer --mode scripted --festivals --record-trace runs/from_viewer.json
```

## What the neural inspector does (and does NOT do)

- Captures the selected agent's **actual local observation vector**, feature names, and legal action mask from the world state *before* the next action is selected.
- Loads the same schema flags as the selected checkpoint (e.g. cow calls v2, nomad v2, festival-extended actions) and labels direction-specific actions separately.
- Displays five highest-probability **legal** actions, selected action, and confirmed/unconfirmed outcome. Probability values belong to the **frozen network** used for inference, not a reward or claim of intent.
- If a species is controlled by scripted behavior in a mixed-policy experiment, displays only local observations and accurately labels the lack of a neural policy.
- A `.json` action trace contains actions, not stored network observations/weights; replay inspection therefore shows observations reconstructed from the recorded world and explicitly labels policy as scripted/recorded. For research requiring frozen neural probability history, export decision data as a separate future artifact.
- Reading the inspector does not change RNG state or introduce unseen predator, cattle, or food locations.

## Telemetry and research export

A bounded ring buffer (default **240 samples**, sampling every **8** ticks) captures:

- Individual populations and juvenile counts for all five species.
- Grass biomass, crops, waste, settlement food, average soil fertility, carcass meat, and sheltered population.
- Festival successes and failures, total birth and death events, charge hits, grazing events, and manure collected.
- Event counts accumulated only for ticks observed by the viewer.

The event panel supports species and selected-actor filters. The X export also writes a bounded JSON file of confirmed, visible events. The chart tab shows population series, grass/crop/stored-food series, and current birth/death/festival totals. JSON exports contain the seed, full world configuration, declared policy mode, sampling interval, bounded sample window, and observed event totals. CSV exports flatten species counts into named columns. This provides direct inputs for reproducible research analysis.

**Limitations:** The telemetry window is bounded; exports are **not** guaranteed to contain an entire long simulation. After a replay seek, cumulative event counts begin from the new observation point, and plots start a new window. `World.events` is still an authoritative unbounded event history; making that memory-bounded without changing existing summary/replay contracts is separate engineering work. The GUI charts are for immediate feedback, not statistical proof of learned cooperation or ecosystem stability.

## Architecture and acceptance tests

- `src/qox/telemetry.py` — no Pygame, torch, or mutable world interactions; bounded time-series capture and exports.
- `src/qox/decision_inspector.py` — immutable, version-aware pre-action policy snapshot and action mask.
- `src/qox/viewer_analytics.py` — pure presentation of two graph panels and a detailed neural panel.
- `src/qox/replay_session.py` — exact hash-verified action replay, back/forward seeking, sparse cache, and major-event navigation.
- `src/qox/viewer.py`, `src/qox/desktop.py`, `src/qox/cli.py` — UI and CLI integration.
- `tests/test_m13_analytics.py`, `tests/test_m13_pygame.py` — read-only invariants, schema matching, exports, bounded buffers, random-access replay, offscreen drawing, and old command compatibility.

Test requirements: unchanged world state hash under telemetry and inspection, exact full replay hash after arbitrary seek sequences, Windows-build offscreen smoke checks, and no changes to any existing checkpoint or action schema. The Windows executable is rebuilt by GitHub Actions to include the new modules.

## Next development goal

M14 should add training progress curves, scenario selection, model evaluation and rollback through the Windows launcher. The viewer should continue making it clear that **running a frozen policy does not train it**.
