# QOX Desktop — Windows Application

The desktop launcher lets you **start with fresh randomly initialized networks**, **train** a chosen actor type, **save** all five neural networks together, **load** a saved set, **import** an existing compatible `.pt` checkpoint for one species, and **run the 2D simulation** without typing command-line arguments.

## Download and open the Windows app

1. Open the latest successful **Windows desktop executable** run under [GitHub Actions](https://github.com/AWDaley/QOX-Machine-Learning/actions/workflows/desktop-windows.yml).
2. Under **Artifacts**, download `QOX-Windows-Desktop` (GitHub may require sign-in).
3. Extract the outer artifact ZIP and then `QOX-Windows.zip` if GitHub provides both layers.
4. Keep the `QOX` folder **intact**, including its `_internal` dependencies.
5. Double-click `QOX.exe` inside the folder. No Python installation is needed.

> The Windows build is made by a `windows-latest` GitHub Actions runner. It is an unsigned research prototype, not a Microsoft-signed installer. Windows SmartScreen may warn on first launch. Only run software from sources you trust.

## Run from Python instead

On Windows with Python 3.11+:

```powershell
python -m pip install -e ".[viewer,ml]"
qox-desktop
```

Or run `python qox_desktop.py` from the repo root.

## Menu choices

- **New neural nets:** initializes five distinct networks with random weights and a configurable seed. They are initially untrained.
- **Train selected species:** continues training its current weights with PPO in a dedicated short scenario. Default 4,096 decisions; longer curricula are possible. The main menu remains responsive. Upon successful training, the trained checkpoint becomes the active model for that species.
- **Start simulation:** runs all five currently active policies simultaneously in a new seeded world. Actions are sampled from learned probabilities by default; uncheck *Sample neural actions* to pick the highest-probability action.
- **Scripted simulation:** runs a scripted reference simulation with the same world settings.
- **Save model set:** writes one `.qox` archive with all five neural weights, observation/action schema metadata, and training decision counters. Saving is a manual step after training.
- **Load model set:** restores all five trained/fresh networks in one operation, with checkpoint validation before switching the session.
- **Import one .pt:** choose a species from the dropdown first, then bring in an existing single-species QOX checkpoint. Legacy cow-v1 and nomad-v1 models remain supported.
- **World options:** toggles reproduction, festivals, and stochastic neural action selection.

**Important distinction:** Pressing Start runs *inference*, **not learning**. QOX's PPO trainers use explicit training scenarios; watching a fresh set of random networks does not update their weights. You need to press **Train selected species** and then **Save model set** to retain learning. This release saves neural networks, **not** the current world, agent positions, or ecosystem history. That can be added as a separate save-game feature.

**Festival limitation:** the five standard neural policies predate the festival-extended farmer/nomad action contract. With *Festivals* enabled, the launcher explicitly uses **scripted contribution/start assistance** while ordinary actor actions remain neural. This does not demonstrate that neural agents independently learned festival coordination. The CLI retains its separately trained festival policy experiments.

## Package architecture

- `src/qox/desktop.py` — Tkinter launcher and background training worker.
- `src/qox/desktop_models.py` — fresh network serialization, zipped model bundles, native QOX policy validation, training dispatch, and all-five-policy inference.
- `qox_desktop.py` — frozen executable entry point.
- `.github/workflows/desktop-windows.yml` — repeatable PyInstaller build on Windows; tests the frozen executable before publishing an artifact.

The bundle is deliberately **weights-only**. It is not a Python pickle of an entire world. The loader uses PyTorch's restricted `weights_only=True` mode and rejects unexpected archive paths and incompatible policy schemas. Do not open downloaded checkpoints from untrusted sources.

## Troubleshooting

- **No window opens:** first run `qox-desktop` from PowerShell after a source installation so Python can show an error. For an executable, ensure the entire `QOX` folder was extracted, not just `QOX.exe`.
- **Training feels slow:** PyTorch CPU training can take time. Reduce the *PPO decisions* entry to 1,024 for a smoke test. This will not usually produce a competent policy.
- **Fresh actors look random:** expected. They are untrained until you train or import policies.
- **Festival outcomes look scripted:** only festival initiation/contribution is scripted when that checkbox is on; disable festivals for a strict five-neural-policy experiment.
- **Old `.pt` checkpoint won't load:** choose the correct species in the selector. A checkpoint with incompatible observation or action dimensions cannot be substituted.
