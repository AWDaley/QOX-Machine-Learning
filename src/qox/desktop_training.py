"""Tkinter training lab for QOX: explicitly evaluate before adopting a policy.

Training uses separate PPO simulation worlds. This GUI never trains policies
inside the visible ecology. All Tk mutation happens on the main thread.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from queue import Empty, SimpleQueue
from threading import Event, Thread
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .desktop_models import train_role
from .entities import Species
from .training_lab import (TrainingCancelled, choose_default,
                           compare_checkpoints, scenarios_for)

BG = "#161e29"
FG = "#e9eef7"
MUTED = "#aab9ca"


class TrainingLabWindow:
    def __init__(self, app):
        self.app = app
        self.root = tk.Toplevel(app.root)
        self.root.title("QOX — Neural Network Training Laboratory")
        self.root.geometry("890x740")
        self.root.minsize(735, 620)
        self.root.configure(bg=BG)
        self.root.transient(app.root)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="Ready. Training produces a candidate, never an automatic replacement.")
        self.role_var = tk.StringVar(value=app.species_var.get())
        self.scenario_var = tk.StringVar()
        self.steps_var = tk.StringVar(value=app.steps_var.get())
        self.seed_var = tk.StringVar(value=app.seed_var.get())
        self.episodes_var = tk.StringVar(value="8")
        self.progress = tk.DoubleVar(value=0)
        self.history: list[dict] = []
        self.cancel_event = Event()
        self.queue: SimpleQueue = SimpleQueue()
        self.training = False
        self.pending: tuple[Species, Path, dict, dict] | None = None
        self._build()
        self._role_changed()
        self.root.after(120, self._poll)

    def _build(self):
        panel = ttk.Frame(self.root, padding=18, style="Qox.TFrame")
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(0, weight=1)
        for col in range(1, 3):
            panel.columnconfigure(col, weight=1)

        ttk.Label(panel, style="Qox.TLabel", text="Neural Network Training Laboratory",
                  font=("Segoe UI", 19, "bold")).grid(row=0, column=0,
                                                         columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(panel, style="Qox.TLabel", text="Train in dedicated PPO scenarios; compare frozen policies "
                  "on matched unseen seeds before accepting.", wraplength=760).grid(
                      row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))

        ttk.Label(panel, style="Qox.TLabel", text="Species").grid(row=2, column=0, sticky="w")
        ttk.Label(panel, style="Qox.TLabel", text="Training scenario").grid(row=2, column=1,
                                                          columnspan=2, sticky="w")
        self.role = ttk.Combobox(panel, textvariable=self.role_var,
                                 values=[s.value for s in Species], state="readonly")
        self.role.grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(3, 9))
        self.role.bind("<<ComboboxSelected>>", self._role_changed)
        self.scenario = ttk.Combobox(panel, textvariable=self.scenario_var,
                                     state="readonly")
        self.scenario.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(3, 9))

        for col, label, variable in ((0, "PPO decisions", self.steps_var),
                                     (1, "Training seed", self.seed_var),
                                     (2, "Held-out episodes", self.episodes_var)):
            ttk.Label(panel, style="Qox.TLabel", text=label).grid(row=4, column=col, sticky="w")
            ttk.Entry(panel, textvariable=variable).grid(row=5, column=col,
                                                          sticky="ew", padx=(0, 8) if col < 2 else 0,
                                                          pady=(3, 10))
        self.train_button = ttk.Button(panel, text="Train candidate", command=self.start)
        self.train_button.grid(row=6, column=0, sticky="ew", padx=(0, 8), pady=(2, 10))
        self.cancel_button = ttk.Button(panel, text="Cancel at rollout boundary",
                                        command=self.cancel, state="disabled")
        self.cancel_button.grid(row=6, column=1, columnspan=2, sticky="ew", pady=(2, 10))
        self.progress_bar = ttk.Progressbar(panel, variable=self.progress, maximum=100)
        self.progress_bar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(2, 5))
        ttk.Label(panel, textvariable=self.status, wraplength=810).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(3, 14))

        ttk.Label(panel, style="Qox.TLabel", text="Training metrics — episode return (green), PPO loss (purple)",
                  font=("Segoe UI", 10, "bold")).grid(row=9, column=0,
                                                       columnspan=3, sticky="w", pady=(0, 3))
        self.chart = tk.Canvas(panel, width=790, height=175, bg="#202e40",
                               highlightthickness=0)
        self.chart.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(0, 12))
        panel.rowconfigure(10, weight=1)

        ttk.Label(panel, style="Qox.TLabel", text="Held-out comparison (higher primary score is better)",
                  font=("Segoe UI", 10, "bold")).grid(row=11, column=0,
                                                       columnspan=3, sticky="w")
        self.results = tk.Text(panel, height=6, bg="#202e40", fg=FG,
                               font=("Consolas", 10), relief="flat", wrap="word",
                               padx=10, pady=7, state="disabled")
        self.results.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(5, 10))
        panel.rowconfigure(12, weight=1)

        self.accept_button = ttk.Button(panel, text="Accept candidate", command=self.accept,
                                        state="disabled")
        self.accept_button.grid(row=13, column=0, sticky="ew", padx=(0, 8))
        self.reject_button = ttk.Button(panel, text="Reject candidate", command=self.reject,
                                        state="disabled")
        self.reject_button.grid(row=13, column=1, sticky="ew", padx=(0, 8))
        self.rollback_button = ttk.Button(panel, text="Rollback last accepted",
                                          command=self.rollback)
        self.rollback_button.grid(row=13, column=2, sticky="ew")
        self.save_button = ttk.Button(panel, text="Save all five model networks (.qox)",
                                      command=self.app._save)
        self.save_button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(8, 0),
                              padx=(0, 8))
        self.export_button = ttk.Button(panel, text="Export experiment JSON",
                                        command=self.export, state="disabled")
        self.export_button.grid(row=14, column=2, sticky="ew", pady=(8, 0))
        ttk.Label(panel, style="Qox.TLabel", text="Policies are not modified by the viewer. "
                  "Loss/return are training signals; comparison scores are separate test outcomes.",
                  wraplength=810).grid(row=15, column=0, columnspan=3,
                                       sticky="w", pady=(9, 0))
        self.chart.bind("<Configure>", lambda event: self._draw_chart())

    def _role_changed(self, _event=None):
        if self.training:
            return
        selected = Species(self.role_var.get())
        if self.pending is not None and self.pending[0] != selected:
            if not messagebox.askyesno("Discard unaccepted candidate?",
                                       "Switching species will discard the unaccepted checkpoint.",
                                       parent=self.root):
                self.role_var.set(self.pending[0].value)
                return
            self._discard_pending()
        schema = self.app.session.status[selected].observation_version
        options = scenarios_for(selected, schema)
        self.scenario.configure(values=options)
        if self.scenario_var.get() not in options:
            self.scenario_var.set(choose_default(selected, schema))
        self.status.set(f"Selected {selected.value} — schema v{schema}; "
                        f"{self.app.session.status[selected].steps:,} trained decisions.")

    def _parse(self) -> tuple[Species, str, int, int, int]:
        from .desktop import _parse_positive_integer
        role = Species(self.role_var.get())
        scenario = self.scenario_var.get()
        if scenario not in scenarios_for(role, self.app.session.status[role].observation_version):
            raise ValueError("Select a scenario compatible with the current network")
        steps = _parse_positive_integer(self.steps_var.get(), name="PPO decisions")
        seed = _parse_positive_integer(self.seed_var.get(), name="Training seed", minimum=0)
        episodes = _parse_positive_integer(self.episodes_var.get(), name="Evaluation episodes")
        if episodes > 200:
            raise ValueError("Evaluation episodes must not exceed 200")
        if steps > 2_000_000:
            raise ValueError("Please use at most 2,000,000 decisions in one run")
        return role, scenario, steps, seed, episodes

    def _set_training(self, value: bool):
        self.training = value
        self.app._set_busy(value)
        self.train_button.configure(state="disabled" if value else "normal")
        self.cancel_button.configure(state="normal" if value else "disabled")
        self.role.configure(state="disabled" if value else "readonly")
        self.scenario.configure(state="disabled" if value else "readonly")
        self.accept_button.configure(state="disabled" if value or not self.pending else "normal")
        self.reject_button.configure(state="disabled" if value or not self.pending else "normal")
        self.export_button.configure(state="disabled" if value or not self.pending else "normal")
        self.rollback_button.configure(state="disabled" if value else "normal")

    def start(self):
        if self.training:
            return
        if self.pending is not None:
            if not messagebox.askyesno("Discard previous candidate?",
                                       "The previous training candidate has not been accepted. "
                                       "Discard it and start again?", parent=self.root):
                return
            self._discard_pending()
        try:
            role, scenario, steps, seed, episodes = self._parse()
        except Exception as exc:
            messagebox.showerror("Invalid training settings", str(exc), parent=self.root)
            return
        self.cancel_event.clear()
        self.history.clear()
        self.progress.set(0)
        self._draw_chart()
        destination = Path(self.app.workspace.name) / f"candidate-{role.value}-{uuid4().hex}.pt"
        source = self.app.session.path(role)
        self._set_training(True)
        self.status.set(f"Training {role.value} in {scenario} for {steps:,} decisions...")
        self._set_results("Training in progress. Active network remains unchanged.")

        def worker():
            try:
                report = train_role(role, source, destination, steps=steps, seed=seed,
                                    scenario=scenario,
                                    progress_callback=lambda point: self.queue.put(("progress", point)),
                                    cancel_requested=self.cancel_event.is_set)
                self.queue.put(("phase", "Training completed. Evaluating both policies "
                                         "on the same held-out worlds..."))
                comparison = compare_checkpoints(role, source, destination,
                                                 scenario=scenario, seed=seed,
                                                 episodes=episodes)
                self.queue.put(("done", (role, destination, report, comparison)))
            except TrainingCancelled as exc:
                destination.unlink(missing_ok=True)
                self.queue.put(("cancelled", str(exc)))
            except Exception as exc:
                destination.unlink(missing_ok=True)
                self.queue.put(("error", str(exc)))

        Thread(target=worker, name="qox-ppo-candidate", daemon=True).start()

    def cancel(self):
        if self.training:
            self.cancel_event.set()
            self.status.set("Cancel requested — stopping at next safe PPO rollout boundary.")

    def _set_results(self, value: str):
        self.results.configure(state="normal")
        self.results.delete("1.0", "end")
        self.results.insert("1.0", value)
        self.results.configure(state="disabled")

    def _show_comparison(self, info: dict):
        self._set_results(
            f"Species: {info['species']}   Scenario: {info['scenario']}\n"
            f"Primary metric: {info['metric_label']} ({info['episodes']} held-out seeds)\n"
            f"Current:   {info['active']:.3f}    Candidate: {info['candidate']:.3f}"
            f"    Scripted reference: {info['scripted']:.3f}\n"
            f"Paired change: {info['delta']:+.3f}    Variation (SD): {info['paired_std']:.3f}"
            f"    Assessment: {info['assessment']}\n"
            "Candidate NOT automatically installed. Accept or reject below."
        )

    def _poll(self):
        if not self.root.winfo_exists():
            return
        try:
            while True:
                kind, item = self.queue.get_nowait()
                if kind == "progress":
                    self.history.append(item)
                    self.history = self.history[-250:]
                    done, total = item["steps"], item["total"]
                    self.progress.set(min(100, 100 * done / max(1, total)))
                    r = item.get("mean_return")
                    self.status.set(f"{done:,}/{total:,} decisions  |  "
                                    f"{item['completed_episodes']} episodes  |  "
                                    f"loss {item['loss']:.3f}  |  "
                                    f"mean return {r if r is not None else 'n/a'}")
                    self._draw_chart()
                elif kind == "phase":
                    self.cancel_button.configure(state="disabled")
                    self.status.set(item)
                elif kind == "done":
                    self.pending = item
                    self._set_training(False)
                    self.progress.set(100)
                    self._show_comparison(item[3])
                    self.status.set("Comparison finished. Review the results before accepting.")
                elif kind in ("cancelled", "error"):
                    self._set_training(False)
                    self.status.set(f"{'Cancelled' if kind == 'cancelled' else 'Training failed'}: {item}")
                    if kind == "error":
                        messagebox.showerror("Training error", item, parent=self.root)
        except Empty:
            pass
        self.root.after(120, self._poll)

    def _draw_chart(self):
        canvas = self.chart
        canvas.delete("all")
        width = max(380, canvas.winfo_width())
        height = max(145, canvas.winfo_height())
        canvas.create_line(40, 10, 40, height - 24, fill="#6e8093")
        canvas.create_line(40, height - 24, width - 12, height - 24, fill="#6e8093")
        if not self.history:
            canvas.create_text(width // 2, height // 2,
                               text="Training chart appears here after the first rollout",
                               fill=MUTED)
            return
        for field, color in (("mean_return", "#5fdb98"), ("loss", "#b6a0f6")):
            pairs = [(index, float(item[field])) for index, item in enumerate(self.history)
                     if item.get(field) is not None]
            if len(pairs) < 2:
                continue
            values = [v for _, v in pairs]
            lo, hi = min(values), max(values)
            span = max(1e-8, hi - lo)
            max_x = max(1, len(self.history) - 1)
            points = []
            for index, value in pairs:
                px = 44 + index * (width - 60) / max_x
                py = 13 + (hi - value) / span * (height - 43)
                points.extend((px, py))
            canvas.create_line(*points, fill=color, width=2, smooth=False)
        canvas.create_text(width - 15, 15, text=f"{self.history[-1]['steps']:,} steps",
                           fill=FG, anchor="ne")

    def _discard_pending(self):
        if self.pending:
            self.pending[1].unlink(missing_ok=True)
        self.pending = None
        self.accept_button.configure(state="disabled")
        self.reject_button.configure(state="disabled")
        self.export_button.configure(state="disabled")

    def export(self):
        if self.training or not self.pending:
            return
        species, candidate, training_report, comparison = self.pending
        filename = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".json",
            initialfile=f"qox-{species.value}-{comparison['scenario']}-experiment.json",
            filetypes=[("JSON experiment report", "*.json")])
        if not filename:
            return
        active = self.app.session.path(species)
        data = {"format": "qox-training-experiment-v1",
                "species": species.value, "scenario": comparison["scenario"],
                "active_sha256": hashlib.sha256(active.read_bytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "training_report": training_report,
                "heldout_comparison": comparison}
        try:
            output = Path(filename)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(output)
            self.status.set(f"Exported research record: {output.name}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def accept(self):
        if self.training or not self.pending:
            return
        role, path, _report, comparison = self.pending
        try:
            self.app.model_history.remember(self.app.session, role)
            self.app.session.replace(role, path, origin="trained")
            self.app._refresh_species()
            self.app.status_var.set(f"Accepted {role.value} candidate. Save your .qox model set.")
            self.status.set(f"Accepted {role.value}. Keep evaluating in diverse worlds.")
            self._discard_pending()
            self._set_results(f"Accepted candidate ({comparison['assessment']} in "
                              "the selected held-out test). Undo using Rollback.")
        except Exception as exc:
            messagebox.showerror("Cannot accept candidate", str(exc), parent=self.root)

    def reject(self):
        self._discard_pending()
        self.status.set("Candidate rejected. The current network is unchanged.")
        self._set_results("Rejected candidate. Current network remains active.")

    def rollback(self):
        if self.training:
            return
        role = Species(self.role_var.get())
        try:
            if self.app.model_history.rollback(self.app.session, role):
                self._discard_pending()
                self.app._refresh_species()
                self.status.set(f"Restored prior {role.value} network. Save model set to retain.")
                self._role_changed()
            else:
                self.status.set(f"No accepted {role.value} checkpoint to roll back.")
        except Exception as exc:
            messagebox.showerror("Rollback failed", str(exc), parent=self.root)

    def close(self):
        if self.training:
            self.cancel()
            messagebox.showinfo("Training in progress",
                                "Cancellation has been requested. Close the lab after "
                                "the current PPO rollout finishes.", parent=self.root)
            return
        if self.pending and not messagebox.askyesno(
                "Discard unaccepted candidate?",
                "Close the training lab without accepting this candidate?",
                parent=self.root):
            return False
        self._discard_pending()
        self.app.training_window = None
        self.root.destroy()
        return True
