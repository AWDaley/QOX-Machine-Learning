# M12 — Confirmed action feedback and event overlays

**Status:** Implemented as read-only viewer presentation. World physics and neural policies are unchanged.

## What changed

M11 displayed a pose and a corner badge for an actor's **requested** action, even if that action could not execute (for example, grazing an empty tile or charging on cooldown). M12 keeps the M11 artwork but distinguishes **confirmed events** from attempted actions.

A viewer-only EventObserver reads World.last_events **after** each world tick. An attempted action without a matching emitted event is shown in the actor inspector as **No confirmed outcome**, and its success pose is suppressed. Neutral rest/wait states have no event contract. Persistent states like hibernation remain visible.

### Effects driven by actual simulation events

- **Ravager:** orange charge wind-up, directional charge trail, red impact marker at the victim's position, miss/obstacle results, carcass dragging, and hibernation halo.
- **Nomad:** distinct cool-blue friendly and warm-orange scary cattle calls, defense impacts, and milk collection.
- **Farmer:** planting, harvest/fertilizer effects, food contribution, and shelter entry/exit.
- **Cow and scavenger:** confirmed grazing and feeding.
- **World:** births/deaths and gold/red festival completion/failure bursts.

The earlier cattle-call rings and active-festival radius remain. No world interactions or sound mechanisms were changed.

## Recent-event feed

A compact, translucent overlay lists real events newest first with tick, actor ID and outcome. Clicking a row focuses the camera and selects a living actor. The default **major** filter avoids flooding the feed with ordinary grazing. Toggle **all** to see those lower-priority events.

The major/all histories have separately bounded 160-entry queues. At most 120 transient effects are retained; they expire after a few world ticks. Pausing freezes effects as it freezes simulation ticks.

## Controls

| Control | Behavior |
| --- | --- |
| E | Show/hide recent-event overlay |
| T | Toggle major/all event filter |
| V | Show/hide effects |
| H | Show/hide small status badges |
| Click an event row | Focus location; select actor if alive |
| Space and N | Pause/resume, then single-step |
| Mouse wheel, WASD, F, F11 | Zoom, pan, fit map, fullscreen |

## Safety and research guarantees

- Observer and renderer have no authority to change actions, physics, rewards, random state, or actor positions.
- A ravager's charge wind-up is **not** a hit; subsequent impact events determine its outcome.
- Failed or skipped actions cannot create success VFX. The viewer states **No confirmed outcome**, rather than claiming to know why an action was skipped.
- Impact markers target the victim's position. Pre-tick actor stamps preserve locations even when a victim dies during the step.
- UI event history is bounded independently of the simulation's longer research event ledger.
- Visual filters and camera operations do not affect world hashes or deterministic trace replay.

## Tests

The pure-Python event suite verifies action confirmation, absent results, charge timing, impact/death locations, festival failure labels, bounded memory, high-priority event retention, camera-only navigation and matching seeded state hashes with and without an observer.

A separate Pygame smoke test renders effects and clickable feed rows on an offscreen surface. It skips only when Pygame is not installed. GitHub CI and the Windows executable workflow are merge gates.

**Limitations:** Visual cues are tick-based rather than interpolated. This is a small recent-event panel, not a full historical timeline (planned for M13). QOX still trains neural networks in separate PPO scenarios, not in the displayed world.
