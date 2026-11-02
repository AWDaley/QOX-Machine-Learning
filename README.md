# QOX Machine Learning

**M0–M1 foundation implemented.** QOX is an experimental deterministic, headless-first, 2D multi-agent artificial-life simulator. Its planned learning phase uses five species-specific neural-network policies: farmers, cows, nomads, scavengers, and ravagers.

## Run it

Python 3.11+; no Docker, database, GPU, or PyTorch needed for M0–M1.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
qox run --ticks 200 --mode scripted --seed 42
python -m pytest -q
```

On macOS/Linux, use `source .venv/bin/activate` instead of the PowerShell activation command.

To save a report and an exact action trace:

```bash
qox run --ticks 200 --seed 42 --mode random --report runs/report.json --trace runs/trace.json
qox replay --trace runs/trace.json
```

For the optional animated viewer:

```bash
python -m pip install -e '.[viewer]'
qox viewer --seed 42 --mode scripted
```

Viewer controls: **Space** pause/resume, **N** single step, **+/-** speed, **Esc** quit. The viewer is observational only; simulation physics do not depend on its framerate. Initial viewer uses colored circles: blue farmers, white cows, orange nomads, purple scavengers, red ravagers; violet squares mark settlements.

Use `qox run --config configs/default.toml --ticks 300` to load a TOML configuration. CLI `--seed` overrides the config seed. Also supported: `python -m qox run ...`.

## Implemented (M0–M1)

- Seeded world creation and validated config.
- 2D terrain with soil fertility, grass, crops, cover, finite waste and carcasses.
- Simultaneous cardinal movement with seeded conflict resolution, support for swaps and blocked chains.
- Five actor types with energy/stamina and simple scripted/random **baselines**.
- Cows consume grass or farmer crops and convert real intake into milk and dung.
- Farmers can plant, harvest, collect manure, fertilize, deposit and eat food.
- Nomads can collect nearby cow milk and consume it.
- Scavengers can eat manure or carrion; ravagers can eat carrion.
- Hunger, death and finite carcasses; ecology advances at a configurable slower interval.
- Structured event ledger, state hashes, seeded action trace and replay verification.
- CLI, basic optional Pygame viewer, pytest and GitHub Actions CI.

## Not yet implemented

**These are planned, not existing functionality:** neural-network training, learned perception, farmer refuge, nomad friendly/scary calls and combat, ravager stealth/charge/hunting/dragging/hibernation, group festivals, births and long-term policy evaluation. The M1 baseline actors are not learned policies, and M1 resource tuning is preliminary.

Read the [full implementation plan](docs/IMPLEMENTATION_PLAN.md) for milestone gates M0–M9 and research experiments. **Next milestone: M2 (local perception and the PettingZoo adapter), followed by M3 (first learned cow PPO policy).**

## Engineering rules

World rules and finite resource accounting are explicit; learned strategies will live outside the world engine. Always distinguish scripted baseline behavior from learned behavior, keep deterministic scenario seeds and action traces, and evaluate frozen policies on held-out worlds. Reports and logs in `runs/` are local artifacts and are ignored by git.
