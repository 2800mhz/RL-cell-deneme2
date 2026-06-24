from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

from .core import CellSimulator
from .policies import ModelPolicy, heuristic_action


class CellLabUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Cell Lab")
        self.root.geometry("1480x900")
        self.root.minsize(1260, 780)

        self.simulator = CellSimulator(max_steps=500)
        self.observation = self.simulator.reset()
        self.model_policy: ModelPolicy | None = None
        self.running = False
        self.interval_ms = 150
        self.chart_window = 180
        self.chart_redraw_every = 2

        self.mode_var = tk.StringVar(value="heuristic")
        self.status_var = tk.StringVar()
        self.detail_var = tk.StringVar()
        self.badge_var = tk.StringVar()
        self.speed_var = tk.StringVar(value=f"{self.interval_ms} ms")
        self.last_action_var = tk.StringVar(value="No action yet")
        self.model_status_var = tk.StringVar(value="No model loaded. Heuristic will be used.")
        self.slider_vars: Dict[str, tk.DoubleVar] = {
            name: tk.DoubleVar(value=0.5) for name in self.simulator.action_names
        }
        self.metric_labels: Dict[str, ttk.Label] = {}
        self.metric_cards: Dict[str, Tuple[tk.Frame, tk.Label, tk.Label]] = {}

        self.palette = {
            "bg": "#f4f1e8",
            "panel": "#fbfaf6",
            "card": "#fffdf8",
            "stroke": "#ded7c8",
            "text": "#172026",
            "muted": "#667085",
            "accent": "#1f6f5f",
            "accent_soft": "#d8eee8",
            "gold": "#c7921b",
            "gold_soft": "#f7edc8",
            "blue": "#2f80ed",
            "blue_soft": "#deebff",
            "red": "#d64545",
            "red_soft": "#fde1e1",
            "ink": "#24313b",
        }

        self._configure_theme()
        self._build_layout()
        self._refresh()

    def _configure_theme(self) -> None:
        self.root.configure(bg=self.palette["bg"])
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        default_font = ("Segoe UI", 11)
        title_font = ("Segoe UI Semibold", 11)

        style.configure("App.TFrame", background=self.palette["bg"])
        style.configure("Panel.TFrame", background=self.palette["panel"])
        style.configure("Card.TFrame", background=self.palette["card"])
        style.configure(
            "Panel.TLabelframe",
            background=self.palette["panel"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background=self.palette["panel"],
            foreground=self.palette["text"],
            font=("Segoe UI Semibold", 12),
        )
        style.configure("App.TLabel", background=self.palette["panel"], foreground=self.palette["text"], font=default_font)
        style.configure("Muted.TLabel", background=self.palette["panel"], foreground=self.palette["muted"], font=default_font)
        style.configure("Title.TLabel", background=self.palette["bg"], foreground=self.palette["text"], font=("Segoe UI Semibold", 24))
        style.configure("Hero.TLabel", background=self.palette["bg"], foreground=self.palette["muted"], font=("Segoe UI", 11))
        style.configure("Section.TLabel", background=self.palette["panel"], foreground=self.palette["text"], font=title_font)
        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 11),
            background=self.palette["accent"],
            foreground="#ffffff",
            padding=(16, 10),
            borderwidth=0,
        )
        style.map("Primary.TButton", background=[("active", "#195a4f")])
        style.configure(
            "Soft.TButton",
            font=("Segoe UI Semibold", 11),
            background=self.palette["card"],
            foreground=self.palette["text"],
            padding=(14, 10),
            borderwidth=1,
        )
        style.map("Soft.TButton", background=[("active", "#f3efe5")])
        style.configure(
            "App.TCombobox",
            padding=8,
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground=self.palette["text"],
        )
        style.configure(
            "Cell.Horizontal.TScale",
            background=self.palette["panel"],
            troughcolor="#e8e0d2",
        )

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16, style="App.TFrame")
        container.pack(fill="both", expand=True)

        self._build_header(container)

        body = ttk.Frame(container, style="App.TFrame")
        body.pack(fill="both", expand=True, pady=(12, 0))
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", width=390)
        left.grid(row=0, column=0, sticky="nsw")
        left.grid_propagate(False)

        right = ttk.Frame(body, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        self._build_sidebar(left)
        self._build_dashboard(right)

    def _build_header(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="App.TFrame")
        header.pack(fill="x")

        left = ttk.Frame(header, style="App.TFrame")
        left.pack(side="left", fill="x", expand=True)

        ttk.Label(left, text="Cell Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Interactive cell control sandbox with live health, energy, stress and growth tracking.",
            style="Hero.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        badge = tk.Label(
            header,
            textvariable=self.badge_var,
            bg=self.palette["accent_soft"],
            fg=self.palette["accent"],
            font=("Segoe UI Semibold", 11),
            padx=14,
            pady=10,
            relief="flat",
        )
        badge.pack(side="right", anchor="n")

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        summary = tk.Frame(parent, bg=self.palette["accent"], padx=18, pady=18)
        summary.pack(fill="x", padx=0, pady=(0, 14))

        tk.Label(
            summary,
            text="Simulation Status",
            bg=self.palette["accent"],
            fg="#ffffff",
            font=("Segoe UI Semibold", 16),
        ).pack(anchor="w")
        tk.Label(
            summary,
            textvariable=self.status_var,
            bg=self.palette["accent"],
            fg="#ffffff",
            font=("Segoe UI", 11),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", pady=(10, 6))
        tk.Label(
            summary,
            textvariable=self.detail_var,
            bg=self.palette["accent"],
            fg="#dff2ec",
            font=("Consolas", 10),
            wraplength=320,
            justify="left",
        ).pack(anchor="w")

        quick = ttk.Frame(parent, style="Panel.TFrame")
        quick.pack(fill="x", pady=(0, 14))
        self._build_metric_cards(quick)

        self._build_controls(parent)
        self._build_sliders(parent)
        self._build_metrics(parent)

    def _build_metric_cards(self, parent: ttk.Frame) -> None:
        cards = ttk.Frame(parent, style="Panel.TFrame")
        cards.pack(fill="x")

        definitions = [
            ("ATP", "atp", self.palette["gold_soft"], self.palette["gold"]),
            ("Size", "size", self.palette["accent_soft"], self.palette["accent"]),
            ("Stress", "ros", self.palette["blue_soft"], self.palette["blue"]),
            ("Damage", "dna_damage", self.palette["red_soft"], self.palette["red"]),
        ]

        for index, (title, key, bg, fg) in enumerate(definitions):
            row = index // 2
            column = index % 2
            card = tk.Frame(cards, bg=bg, padx=14, pady=12, highlightthickness=1, highlightbackground=bg)
            card.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            cards.grid_columnconfigure(column, weight=1)
            label = tk.Label(card, text=title, bg=bg, fg=fg, font=("Segoe UI Semibold", 11))
            label.pack(anchor="w")
            value = tk.Label(card, text="--", bg=bg, fg=self.palette["ink"], font=("Segoe UI Semibold", 18))
            value.pack(anchor="w", pady=(6, 0))
            sub = tk.Label(card, text="", bg=bg, fg=fg, font=("Segoe UI", 9))
            sub.pack(anchor="w", pady=(2, 0))
            self.metric_cards[key] = (card, value, sub)

    def _build_controls(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Controls", padding=14, style="Panel.TLabelframe")
        controls.pack(fill="x", pady=(0, 14))

        ttk.Label(controls, text="Control Mode", style="Section.TLabel").pack(anchor="w")
        ttk.Label(controls, text="Switch between manual sliders, built-in heuristic or a trained model.", style="Muted.TLabel").pack(anchor="w", pady=(2, 10))
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["manual", "heuristic", "model"],
            state="readonly",
            style="App.TCombobox",
        )
        mode_box.pack(fill="x", pady=(0, 12))
        mode_box.bind("<<ComboboxSelected>>", self._on_mode_changed)
        ttk.Label(controls, textvariable=self.model_status_var, style="Muted.TLabel", wraplength=320).pack(anchor="w", pady=(0, 10))

        button_row = ttk.Frame(controls)
        button_row.pack(fill="x", pady=(0, 8))
        ttk.Button(button_row, text="Start", command=self.start, style="Primary.TButton").pack(side="left", fill="x", expand=True)
        ttk.Button(button_row, text="Pause", command=self.pause, style="Soft.TButton").pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(button_row, text="Step", command=self.step_once, style="Soft.TButton").pack(side="left", fill="x", expand=True)

        second_row = ttk.Frame(controls)
        second_row.pack(fill="x", pady=(0, 8))
        ttk.Button(second_row, text="Reset", command=self.reset, style="Soft.TButton").pack(side="left", fill="x", expand=True)
        ttk.Button(second_row, text="Load Model", command=self.load_model, style="Soft.TButton").pack(side="left", fill="x", expand=True, padx=8)

        speed_row = ttk.Frame(controls, style="Panel.TFrame")
        speed_row.pack(fill="x", pady=(8, 0))
        ttk.Label(speed_row, text="Playback Speed", style="Section.TLabel").pack(side="left")
        ttk.Label(speed_row, textvariable=self.speed_var, style="Muted.TLabel").pack(side="right")
        speed = ttk.Scale(controls, from_=50, to=600, command=self._change_speed, style="Cell.Horizontal.TScale")
        speed.set(self.interval_ms)
        speed.pack(fill="x", pady=(8, 10))

        ttk.Label(controls, text="Last Action Mix", style="Section.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(controls, textvariable=self.last_action_var, style="Muted.TLabel", wraplength=320).pack(anchor="w", pady=(2, 0))

    def _build_sliders(self, parent: ttk.Frame) -> None:
        sliders = ttk.LabelFrame(parent, text="Manual Controls", padding=14, style="Panel.TLabelframe")
        sliders.pack(fill="x", pady=(0, 14))
        ttk.Label(sliders, text="Use these when mode is set to manual.", style="Muted.TLabel").pack(anchor="w", pady=(0, 10))
        for name, variable in self.slider_vars.items():
            row = ttk.Frame(sliders, style="Panel.TFrame")
            row.pack(fill="x", pady=4)
            label_text = name.replace("_", " ").title()
            ttk.Label(row, text=label_text, style="App.TLabel").pack(anchor="w")
            value_label = ttk.Label(row, text=f"{variable.get():.2f}", style="Muted.TLabel")
            value_label.pack(anchor="e")
            slider = ttk.Scale(
                row,
                from_=0.0,
                to=1.0,
                variable=variable,
                command=lambda value, label=value_label: label.configure(text=f"{float(value):.2f}"),
                style="Cell.Horizontal.TScale",
            )
            slider.pack(fill="x", pady=(4, 0))

    def _build_metrics(self, parent: ttk.Frame) -> None:
        metrics = ttk.LabelFrame(parent, text="Detailed Metrics", padding=14, style="Panel.TLabelframe")
        metrics.pack(fill="x")
        fields = [
            "atp",
            "glucose",
            "oxygen",
            "amino_acids",
            "waste",
            "ros",
            "dna_damage",
            "membrane_damage",
            "size",
            "division_progress",
            "mitochondria_efficiency",
            "ribosome_capacity",
            "nucleus_stability",
            "membrane_integrity",
            "lysosome_activity",
            "er_stability",
            "golgi_capacity",
        ]
        for field in fields:
            row = ttk.Frame(metrics, style="Panel.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=field.replace("_", " ").title(), style="App.TLabel").pack(side="left")
            label = ttk.Label(row, text="-", style="Section.TLabel")
            label.pack(side="right")
            self.metric_labels[field] = label

    def _build_dashboard(self, parent: ttk.Frame) -> None:
        hero = tk.Frame(parent, bg=self.palette["card"], padx=20, pady=18, highlightthickness=1, highlightbackground=self.palette["stroke"])
        hero.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        tk.Label(hero, text="Live Dashboard", bg=self.palette["card"], fg=self.palette["text"], font=("Segoe UI Semibold", 18)).pack(anchor="w")
        tk.Label(
            hero,
            text="ATP, growth, stress and damage are rendered as live time series. Manual and heuristic controls use the same simulation core.",
            bg=self.palette["card"],
            fg=self.palette["muted"],
            font=("Segoe UI", 11),
            wraplength=860,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        self.plot_canvas = tk.Canvas(parent, bg=self.palette["bg"], highlightthickness=0)
        self.plot_canvas.grid(row=1, column=0, sticky="nsew")

    def _change_speed(self, value: str) -> None:
        self.interval_ms = int(float(value))
        self.speed_var.set(f"{self.interval_ms} ms")

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model",
            filetypes=[("Supported models", "*.json *.zip"), ("JSON model", "*.json"), ("ZIP model", "*.zip"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.model_policy = ModelPolicy(Path(path))
            self.mode_var.set("model")
            self.model_status_var.set(f"Model loaded: {Path(path).name}")
            messagebox.showinfo("Cell Lab", f"Model loaded:\n{path}")
        except Exception as exc:
            self.model_policy = None
            self.model_status_var.set("Model load failed. Heuristic will be used.")
            messagebox.showerror("Cell Lab", str(exc))

    def _on_mode_changed(self, event: object | None = None) -> None:
        if self.mode_var.get() == "model" and self.model_policy is None:
            self.mode_var.set("heuristic")
            self.model_status_var.set("Model mode requested, but no compatible model is loaded. Switched to heuristic.")
            messagebox.showwarning(
                "Cell Lab",
                "No compatible model is loaded.\n\nThe app switched back to heuristic mode.",
            )
        elif self.mode_var.get() == "model":
            self.model_status_var.set("Model mode active.")
        elif self.mode_var.get() == "manual":
            self.model_status_var.set("Manual mode active. Sliders control the cell.")
        else:
            self.model_status_var.set("Heuristic mode active. Built-in auto-controller is running.")

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._tick()

    def pause(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.running = False
        self.simulator = CellSimulator(max_steps=500)
        self.observation = self.simulator.reset()
        self._refresh()

    def step_once(self) -> None:
        if self.simulator.state.alive and self.simulator.steps < self.simulator.max_steps:
            self._advance()
        self._refresh()

    def _tick(self) -> None:
        if not self.running:
            return
        if self.simulator.state.alive and self.simulator.steps < self.simulator.max_steps:
            self._advance()
            self._refresh()
            self.root.after(self.interval_ms, self._tick)
        else:
            self.running = False
            self._refresh()

    def _advance(self) -> None:
        action = self._get_action()
        result = self.simulator.step(action)
        self.observation = result.observation
        named = [f"{name.replace('_', ' ')} {value:.2f}" for name, value in zip(self.simulator.action_names, action)]
        self.last_action_var.set(" | ".join(named[:4]) + " ...")

    def _get_action(self) -> list[float]:
        mode = self.mode_var.get()
        if mode == "manual":
            return [float(var.get()) for var in self.slider_vars.values()]
        if mode == "model" and self.model_policy is not None:
            return self.model_policy.predict(self.observation)
        return heuristic_action(self.simulator)

    def _refresh(self) -> None:
        metrics = self.simulator.get_metrics()
        for key, label in self.metric_labels.items():
            value = metrics[key]
            label.configure(text=f"{value:.2f}")

        self._refresh_cards(metrics)

        state = "Alive" if self.simulator.state.alive else "Dead"
        ready = "Ready" if self.simulator.ready_to_divide() else "Not ready"
        self.badge_var.set(f"{state} • Step {self.simulator.steps}/{self.simulator.max_steps}")
        active_mode = self.mode_var.get().title()
        if self.mode_var.get() == "model" and self.model_policy is None:
            active_mode = "Heuristic (model missing)"
        self.status_var.set(f"{state} cell. Division status: {ready}. Mode: {active_mode}.")
        self.detail_var.set(self.simulator.summary())
        if (
            not self.running
            or self.simulator.steps <= 2
            or self.simulator.steps % self.chart_redraw_every == 0
            or not self.simulator.state.alive
            or self.simulator.steps >= self.simulator.max_steps
        ):
            self._redraw_plot()

    def _refresh_cards(self, metrics: Dict[str, float]) -> None:
        stress_value = metrics["ros"] + metrics["waste"]
        damage_value = metrics["dna_damage"] + metrics["membrane_damage"]
        card_values = {
            "atp": (metrics["atp"], "Energy reserve"),
            "size": (metrics["size"], "Growth factor"),
            "ros": (stress_value, "ROS + waste"),
            "dna_damage": (damage_value, "DNA + membrane"),
        }
        for key, (value, subtitle) in card_values.items():
            _, value_label, sub_label = self.metric_cards[key]
            precision = 2 if key == "size" else 1
            value_label.configure(text=f"{value:.{precision}f}")
            sub_label.configure(text=subtitle)

    def _redraw_plot(self) -> None:
        self.plot_canvas.delete("all")
        history: List[Dict[str, float]] = self.simulator.history
        if len(history) < 2:
            return
        if len(history) > self.chart_window:
            history = history[-self.chart_window :]
        steps = [row["step"] for row in history]
        atp = [row["atp"] for row in history]
        size = [row["size"] for row in history]
        stress = [row["ros"] + row["waste"] for row in history]
        damage = [row["dna_damage"] + row["membrane_damage"] for row in history]
        charts = [
            ("ATP", atp, 120.0, "#d4a017"),
            ("Size", size, 2.5, "#2d6a4f"),
            ("Stress", stress, 200.0, "#3a86ff"),
            ("Damage", damage, 200.0, "#d00000"),
        ]
        self._draw_chart_grid(charts, steps)

    def _draw_chart_grid(self, charts: List[tuple[str, List[float], float, str]], steps: List[float]) -> None:
        self.plot_canvas.update_idletasks()
        width = max(self.plot_canvas.winfo_width(), 800)
        height = max(self.plot_canvas.winfo_height(), 600)
        card_w = width // 2 - 26
        card_h = height // 2 - 24
        positions = [
            (10, 10),
            (card_w + 22, 10),
            (10, card_h + 22),
            (card_w + 22, card_h + 22),
        ]

        for (title, values, max_value, color), (x, y) in zip(charts, positions):
            self._draw_chart_card(title, values, max_value, color, x, y, card_w, card_h, steps)

    def _draw_chart_card(
        self,
        title: str,
        values: List[float],
        max_value: float,
        color: str,
        x: int,
        y: int,
        card_w: int,
        card_h: int,
        steps: List[float],
    ) -> None:
        self.plot_canvas.create_rectangle(
            x,
            y,
            x + card_w,
            y + card_h,
            fill=self.palette["card"],
            outline=self.palette["stroke"],
            width=1,
        )
        current = values[-1] if values else 0.0
        self.plot_canvas.create_text(x + 18, y + 20, text=title, anchor="w", font=("Segoe UI Semibold", 16), fill=self.palette["text"])
        self.plot_canvas.create_text(
            x + card_w - 18,
            y + 22,
            text=f"{current:.1f}",
            anchor="e",
            font=("Segoe UI Semibold", 13),
            fill=color,
        )
        self._draw_series(x + 18, y + 42, card_w - 36, card_h - 68, steps, values, max_value, color)

    def _draw_series(self, x: int, y: int, width: int, height: int, steps: List[float], values: List[float], max_value: float, color: str) -> None:
        self.plot_canvas.create_rectangle(x, y, x + width, y + height, fill="#fffefe", outline="#edf1f5")
        for ratio in (0.25, 0.5, 0.75):
            gy = y + height * ratio
            self.plot_canvas.create_line(x, gy, x + width, gy, fill="#e3e6ea", dash=(3, 5))

        sampled_steps, sampled_values = self._downsample_series(steps, values, max(width // 2, 60))
        max_step = max(sampled_steps) if sampled_steps else 1.0
        points: List[float] = []
        for step, value in zip(sampled_steps, sampled_values):
            px = x + (step / max_step) * width if max_step else x
            py = y + height - (max(0.0, min(value, max_value)) / max_value) * height
            points.extend([round(px, 2), round(py, 2)])
        if len(points) >= 4:
            self.plot_canvas.create_line(*points, fill=color, width=2.5, smooth=False)
            last_x = points[-2]
            last_y = points[-1]
            self.plot_canvas.create_oval(last_x - 4, last_y - 4, last_x + 4, last_y + 4, fill=color, outline=color)
        self.plot_canvas.create_text(x, y + height + 16, text="0", anchor="w", font=("Segoe UI", 9), fill=self.palette["muted"])
        self.plot_canvas.create_text(x + width, y + height + 16, text=str(int(max_step)), anchor="e", font=("Segoe UI", 9), fill=self.palette["muted"])
        self.plot_canvas.create_text(x + width, y - 10, text=f"max {max_value:g}", anchor="e", font=("Segoe UI", 9), fill=self.palette["muted"])

    def _downsample_series(self, steps: List[float], values: List[float], max_points: int) -> Tuple[List[float], List[float]]:
        if len(steps) <= max_points:
            return steps, values

        stride = max(1, len(steps) // max_points)
        sampled_steps = steps[::stride]
        sampled_values = values[::stride]

        if sampled_steps[-1] != steps[-1]:
            sampled_steps.append(steps[-1])
            sampled_values.append(values[-1])

        return sampled_steps, sampled_values

    def run(self) -> None:
        self.root.mainloop()


def launch_ui() -> None:
    CellLabUI().run()
