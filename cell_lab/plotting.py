from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Optional

from .core import CellSimulator
from .policies import ModelPolicy, heuristic_action


def run_and_plot(mode: str = "heuristic", steps: int = 300, model_path: Optional[str] = None) -> Path:
    simulator = CellSimulator(max_steps=steps)
    observation = simulator.reset()
    policy = ModelPolicy(model_path) if mode == "model" and model_path else None

    for _ in range(steps):
        if mode == "manual":
            action = [0.5] * 8
        elif mode == "model" and policy is not None:
            action = policy.predict(observation)
        else:
            action = heuristic_action(simulator)

        result = simulator.step(action)
        observation = result.observation
        if result.done:
            break

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    output_path = plot_dir / f"cell_lab_{mode}.svg"

    history = simulator.history
    steps_axis = [row["step"] for row in history]
    atp = [row["atp"] for row in history]
    size = [row["size"] for row in history]
    waste = [row["waste"] for row in history]
    damage = [row["dna_damage"] + row["membrane_damage"] for row in history]
    charts = [
        ("ATP", atp, 120.0, "#d4a017"),
        ("Size", size, 2.5, "#2d6a4f"),
        ("Waste", waste, 100.0, "#3a86ff"),
        ("Damage", damage, 200.0, "#d00000"),
    ]
    svg = _build_svg(charts, steps_axis, mode)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _build_svg(charts: list[tuple[str, list[float], float, str]], steps: list[float], mode: str) -> str:
    width = 1200
    height = 820
    card_w = 540
    card_h = 300
    positions = [(40, 90), (620, 90), (40, 430), (620, 430)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f7f2"/>',
        f'<text x="40" y="48" font-size="30" font-family="Segoe UI, Arial" fill="#1f2937">Cell Lab Simulation ({escape(mode)})</text>',
    ]
    max_step = max(steps) if steps else 1.0

    for (title, values, max_value, color), (x, y) in zip(charts, positions):
        parts.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="18" fill="#ffffff" stroke="#d9dee8"/>')
        parts.append(f'<text x="{x + 18}" y="{y + 32}" font-size="22" font-family="Segoe UI, Arial" fill="#111827">{escape(title)}</text>')
        plot_x = x + 20
        plot_y = y + 50
        plot_w = card_w - 40
        plot_h = card_h - 70
        parts.append(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="#edf1f5"/>')

        for ratio in (0.25, 0.5, 0.75):
            gy = plot_y + plot_h * ratio
            parts.append(f'<line x1="{plot_x}" y1="{gy:.1f}" x2="{plot_x + plot_w}" y2="{gy:.1f}" stroke="#e5e7eb" stroke-dasharray="4 4"/>')

        if values:
            path = []
            for index, value in enumerate(values):
                px = plot_x + (steps[index] / max_step) * plot_w if max_step else plot_x
                py = plot_y + plot_h - (min(max(value, 0.0), max_value) / max_value) * plot_h
                command = "M" if index == 0 else "L"
                path.append(f"{command} {px:.2f} {py:.2f}")
            parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="3"/>')

        parts.append(f'<text x="{plot_x}" y="{plot_y + plot_h + 22}" font-size="14" font-family="Segoe UI, Arial" fill="#6b7280">0</text>')
        parts.append(f'<text x="{plot_x + plot_w - 30}" y="{plot_y + plot_h + 22}" font-size="14" font-family="Segoe UI, Arial" fill="#6b7280">{int(max_step)}</text>')
        parts.append(f'<text x="{plot_x + plot_w - 50}" y="{plot_y + 18}" font-size="14" font-family="Segoe UI, Arial" fill="#6b7280">{max_value:g}</text>')

    parts.append("</svg>")
    return "".join(parts)
