"""Windows-friendly Tkinter launcher for five-species QOX neural simulations."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, SimpleQueue
from uuid import uuid4
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import SimulationConfig
from .desktop_models import (ModelSession, MultiPolicyController, ROLES,
                             train_role)
from .entities import Species


ROLE_LABELS = {species.value: species.value.capitalize() for species in ROLES}


def _parse_positive_integer(text: str, *, name: str, minimum: int = 1) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class QoxDesktop:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("QOX — Artificial Life Lab")
        self.root.geometry("790x690")
        self.root.minsize(680, 620)
        self.root.configure(background="#161e29")
        self.workspace = tempfile.TemporaryDirectory(prefix="qox-desktop-")
        self.session = ModelSession.fresh(Path(self.workspace.name) / "active", seed=42)
        self.queue: SimpleQueue = SimpleQueue()
        self.busy = False
        self.buttons: list[ttk.Button] = []
        self.status_var = tk.StringVar(value="Ready. Five new UNTRAINED networks created.")
        self.seed_var = tk.StringVar(value="42")
        self.steps_var = tk.StringVar(value="4096")
        self.species_var = tk.StringVar(value=Species.COW.value)
        self.reproduction_var = tk.BooleanVar(value=True)
        self.festivals_var = tk.BooleanVar(value=True)
        self.stochastic_var = tk.BooleanVar(value=True)
        self._configure_style()
        self._draw()
        self._refresh_species()
        self.root.after(150, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Qox.TFrame", background="#161e29")
        style.configure("Qox.TLabel", background="#161e29", foreground="#e9eef7",
                        font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#161e29", foreground="#9caabc",
                        font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#161e29", foreground="#f7fafc",
                        font=("Segoe UI Semibold", 22))
        style.configure("Qox.TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Qox.TCheckbutton", background="#161e29", foreground="#e9eef7",
                        font=("Segoe UI", 10))
        style.map("Qox.TCheckbutton", background=[("active", "#161e29")],
                  foreground=[("active", "#ffffff")])
        style.configure("Qox.TCombobox", padding=5)

    def _button(self, frame, text, command, row, column, *, width=19):
        button = ttk.Button(frame, text=text, command=command, style="Qox.TButton", width=width)
        button.grid(row=row, column=column, padx=5, pady=5, sticky="ew")
        self.buttons.append(button)
        return button

    def _draw(self):
        frame = ttk.Frame(self.root, style="Qox.TFrame", padding=20)
        frame.pack(fill="both", expand=True)
        for column in range(3):
            frame.columnconfigure(column, weight=1)
        ttk.Label(frame, text="QOX  |  Artificial Life Lab", style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 2))
        ttk.Label(frame, text="Five neural-network species • CPU training • Save / Load model sets",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=3, sticky="w",
                                               pady=(0, 18))

        ttk.Label(frame, text="CURRENT NETWORKS", style="Qox.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(0, 4))
        self.role_status = tk.Text(frame, height=5, font=("Consolas", 10), bg="#212d3d",
                                   fg="#e7f0fd", relief="flat", padx=10, pady=8,
                                   state="disabled", wrap="none")
        self.role_status.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        self._button(frame, "New neural nets", self._fresh, 4, 0)
        self._button(frame, "Load model set", self._load, 4, 1)
        self._button(frame, "Save model set", self._save, 4, 2)
        self._button(frame, "Import one .pt", self._import_checkpoint, 5, 0)
        self._button(frame, "Start simulation", self._start, 5, 1)
        self._button(frame, "Scripted simulation", self._start_scripted, 5, 2)

        ttk.Separator(frame, orient="horizontal").grid(row=6, column=0, columnspan=3,
                                                         sticky="ew", pady=12)
        ttk.Label(frame, text="TRAIN A NETWORK", style="Qox.TLabel").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(frame, text="Species", style="Muted.TLabel").grid(row=8, column=0,
                                                                      sticky="w")
        ttk.Label(frame, text="PPO decisions", style="Muted.TLabel").grid(row=8, column=1,
                                                                           sticky="w")
        ttk.Label(frame, text="Simulation seed", style="Muted.TLabel").grid(row=8, column=2,
                                                                             sticky="w")
        self.role_combo = ttk.Combobox(frame, textvariable=self.species_var, state="readonly",
                                       values=[s.value for s in ROLES], style="Qox.TCombobox")
        self.role_combo.grid(row=9, column=0, sticky="ew", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.steps_var).grid(row=9, column=1, sticky="ew",
                                                           padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.seed_var).grid(row=9, column=2, sticky="ew",
                                                          padx=5, pady=5)
        self._button(frame, "Train selected species", self._train, 10, 0, width=20)
        ttk.Label(frame, text="Training uses dedicated PPO scenarios; the open viewer does NOT train.",
                  style="Muted.TLabel", wraplength=460).grid(
            row=10, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Separator(frame, orient="horizontal").grid(row=11, column=0, columnspan=3,
                                                         sticky="ew", pady=10)
        ttk.Label(frame, text="WORLD OPTIONS", style="Qox.TLabel").grid(
            row=12, column=0, columnspan=3, sticky="w", pady=(0, 5))
        ttk.Checkbutton(frame, text="Reproduction", variable=self.reproduction_var,
                        style="Qox.TCheckbutton").grid(row=13, column=0, sticky="w")
        ttk.Checkbutton(frame, text="Festivals", variable=self.festivals_var,
                        style="Qox.TCheckbutton").grid(row=13, column=1, sticky="w")
        ttk.Checkbutton(frame, text="Sample neural actions", variable=self.stochastic_var,
                        style="Qox.TCheckbutton").grid(row=13, column=2, sticky="w")
        ttk.Label(frame, text=("Festival proposals/contributions use a documented scripted "
                               "assist; all other behavior comes from the five neural networks."),
                  style="Muted.TLabel", wraplength=700).grid(
            row=14, column=0, columnspan=3, sticky="w", pady=(9, 5))
        ttk.Label(frame, text=("Save stores model weights and training progress, not the "
                               "current world's positions or resources. Save after training."),
                  style="Muted.TLabel", wraplength=700).grid(
            row=15, column=0, columnspan=3, sticky="w", pady=(0, 9))
        ttk.Separator(frame, orient="horizontal").grid(row=16, column=0, columnspan=3,
                                                         sticky="ew", pady=(4, 12))
        self.status_label = ttk.Label(frame, textvariable=self.status_var,
                                      style="Qox.TLabel", wraplength=710)
        self.status_label.grid(row=17, column=0, columnspan=3, sticky="ew")

    def _refresh_species(self):
        self.role_status.configure(state="normal")
        self.role_status.delete("1.0", "end")
        for role in ROLES:
            data = self.session.status[role]
            self.role_status.insert("end", f"{role.value:<11} {data.origin:<12} "
                                            f"{data.steps:>8,} trained decisions  "
                                            f"schema v{data.observation_version}\n")
        self.role_status.configure(state="disabled")

    def _confirm_discard(self) -> bool:
        return (not self.session.dirty or messagebox.askyesno(
            "Unsaved neural networks", "Discard the current unsaved model changes?",
            parent=self.root))

    def _fresh(self):
        if not self._confirm_discard():
            return
        try:
            seed = _parse_positive_integer(self.seed_var.get(), name="Seed", minimum=0)
            new = ModelSession.fresh(Path(self.workspace.name) / f"new-{uuid4().hex}", seed=seed)
            # A separate directory makes replacement robust after cancelled load.
            self.session = new
            self._refresh_species()
            self.status_var.set(f"Created five UNTRAINED neural networks (seed {seed}).")
        except Exception as exc:
            messagebox.showerror("New networks failed", str(exc), parent=self.root)

    def _load(self):
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(parent=self.root,
                                           title="Load five-neural-network QOX set",
                                           filetypes=[("QOX neural sets", "*.qox")])
        if not path:
            return
        try:
            session = ModelSession.load(Path(path), Path(self.workspace.name) / f"loaded-{uuid4().hex}")
            self.session = session
            self._refresh_species()
            self.status_var.set(f"Loaded all five networks from {Path(path).name}.")
        except Exception as exc:
            messagebox.showerror("Cannot load model set", str(exc), parent=self.root)

    def _save(self):
        default = self.session.saved_path.name if self.session.saved_path else "qox-models.qox"
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".qox",
                                            initialfile=default, title="Save all five networks",
                                            filetypes=[("QOX neural sets", "*.qox")])
        if not path:
            return
        try:
            saved = self.session.save(Path(path))
            self.status_var.set(f"Saved five neural networks to {saved}.")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)

    def _import_checkpoint(self):
        role = Species(self.species_var.get())
        path = filedialog.askopenfilename(parent=self.root,
                                           title=f"Import {role.value} PyTorch checkpoint",
                                           filetypes=[("PyTorch checkpoints", "*.pt")])
        if not path:
            return
        try:
            self.session.replace(role, Path(path), origin="imported")
            self._refresh_species()
            self.status_var.set(f"Imported {role.value} checkpoint; save the model set to keep it.")
        except Exception as exc:
            messagebox.showerror("Checkpoint incompatible", str(exc), parent=self.root)

    def _config(self) -> SimulationConfig:
        seed = _parse_positive_integer(self.seed_var.get(), name="Seed", minimum=0)
        reproduction = self.reproduction_var.get()
        return SimulationConfig(seed=seed, population_dynamics_enabled=reproduction,
                                reproduction_enabled=reproduction,
                                festival_enabled=self.festivals_var.get())

    def _show_simulation(self, *, scripted: bool):
        if self.busy:
            return
        try:
            config = self._config()
            controller = None if scripted else MultiPolicyController(
                self.session, sample_actions=self.stochastic_var.get(),
                festival_assist=self.festivals_var.get())
        except Exception as exc:
            messagebox.showerror("Cannot start simulation", str(exc), parent=self.root)
            return
        self.root.withdraw()
        try:
            from .viewer import display
            display(config, mode="scripted" if scripted else "multi-learned",
                    active_controller=controller)
            self.status_var.set("Simulation closed. Networks unchanged; train to update weights.")
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc), parent=self.root)
        finally:
            self.root.deiconify()

    def _start(self):
        self._show_simulation(scripted=False)

    def _start_scripted(self):
        self._show_simulation(scripted=True)

    def _set_busy(self, value: bool):
        self.busy = value
        for button in self.buttons:
            button.configure(state="disabled" if value else "normal")
        self.role_combo.configure(state="disabled" if value else "readonly")

    def _train(self):
        if self.busy:
            return
        try:
            role = Species(self.species_var.get())
            steps = _parse_positive_integer(self.steps_var.get(), name="Training decisions")
            seed = _parse_positive_integer(self.seed_var.get(), name="Seed", minimum=0)
        except Exception as exc:
            messagebox.showerror("Invalid training settings", str(exc), parent=self.root)
            return
        source = self.session.path(role)
        destination = Path(self.workspace.name) / f"training-{role.value}.pt"
        self._set_busy(True)
        self.status_var.set(f"Training {role.value} for {steps:,} PPO decisions... "
                            "This runs in a separate training world.")

        def worker():
            try:
                report = train_role(role, source, destination, steps=steps, seed=seed)
                self.queue.put((role, destination, report, None))
            except Exception as exc:
                self.queue.put((role, destination, None, str(exc)))

        threading.Thread(target=worker, name="qox-ppo-training", daemon=True).start()

    def _poll(self):
        try:
            while True:
                role, destination, report, error = self.queue.get_nowait()
                self._set_busy(False)
                if error:
                    self.status_var.set(f"Training failed: {error}")
                    messagebox.showerror("Training error", error, parent=self.root)
                else:
                    try:
                        self.session.replace(role, destination, origin="trained")
                        self._refresh_species()
                        self.status_var.set(
                            f"Trained {role.value}; {self.session.status[role].steps:,} decisions "
                            f"total ({int(report.get('completed_episodes', 0))} training episodes). "
                            "Save your QOX model set now.")
                    except Exception as exc:
                        messagebox.showerror("Cannot apply checkpoint", str(exc),
                                             parent=self.root)
        except Empty:
            pass
        self.root.after(150, self._poll)

    def _close(self):
        if self.busy and not messagebox.askyesno(
                "Training running", "Training is still running. Quit without saving its result?",
                parent=self.root):
            return
        if self.session.dirty and not messagebox.askyesno(
                "Unsaved neural networks", "Quit without saving the current model set?",
                parent=self.root):
            return
        self.root.destroy()
        # Do not delete files still in use by a running Windows worker thread.
        if not self.busy:
            self.workspace.cleanup()


def smoke_test() -> int:
    """No-window executable check: models, bundle, inference, and action legality."""
    from .world import World

    with tempfile.TemporaryDirectory(prefix="qox-smoke-") as tmp:
        root = Path(tmp)
        session = ModelSession.fresh(root / "fresh", seed=9)
        archive = session.save(root / "neural.qox")
        loaded = ModelSession.load(archive, root / "loaded")
        controller = MultiPolicyController(loaded)
        world = World(SimulationConfig(seed=9, population_dynamics_enabled=True,
                                       reproduction_enabled=True, festival_enabled=True))
        for _ in range(3):
            world.step(controller.actions(world))
        assert world.tick == 3
    return 0


def main() -> int:
    import sys

    if "--smoke-test" in sys.argv:
        return smoke_test()
    root = tk.Tk()
    QoxDesktop(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
