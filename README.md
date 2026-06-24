# Cell Lab

Cell Lab is a cleaned-up rebuild of the original repository: a small but coherent project for simulating a single cell, experimenting with control strategies, and optionally training an RL agent on top of the same environment.

The project now has one shared simulation core, one GUI, one CLI, one plotting flow, and a documented entrypoint layout. The goal is to make the repo understandable and usable even before any RL training happens.

## What this project does

- Simulates a simplified cell with energy, nutrients, stress, damage, growth, and division readiness.
- Lets you control the cell manually, with a heuristic controller, or with an optional trained RL model.
- Provides a desktop GUI for interactive experimentation.
- Provides a CLI for quick simulations and plot generation.
- Exposes an optional Gymnasium-compatible environment for Stable-Baselines3 training.

## Project structure

```text
cell_lab/
  cli.py         Command-line entrypoints
  core.py        Shared cell simulation model
  env.py         RL environment wrapper
  plotting.py    Batch simulation + plot export
  policies.py    Manual/heuristic/model control helpers
  training.py    Optional RL training helpers
  ui.py          Desktop GUI
main.py          Primary app entrypoint
gui.py           Legacy wrapper -> GUI
visualize.py     Legacy wrapper -> plotting
train_advanced.py Legacy wrapper -> training
```

## Quick start

### 1. Install dependencies

Minimal usage:

Base simulator usage does not require any third-party package:

```bash
pip install -r requirements.txt
```

The base `requirements.txt` intentionally stays empty so the simulator and GUI can run with stock Python. RL training is optional and needs `gymnasium`, `stable-baselines3`, and `torch`.

### 2. Launch the GUI

```bash
python main.py ui
```

### 3. Run a quick simulation from the terminal

```bash
python main.py simulate --mode heuristic --steps 300
```

### 4. Export a plot

```bash
python main.py plot --mode heuristic --steps 300
```

This saves an SVG chart under `plots/`.

### 5. Train an RL agent

```bash
python main.py train --algorithm ppo --timesteps 100000
```

Trained models are saved under `models/`.

## Control model

The simulator uses eight continuous control channels between `0.0` and `1.0`:

1. Nutrient uptake
2. Oxygen uptake
3. Glycolysis
4. Aerobic respiration
5. Protein synthesis
6. Repair
7. Detox
8. Division focus

Each step balances energy production, maintenance costs, oxidative stress, structural damage, size growth, and long-term division readiness.

## GUI modes

- `Manual`: You move all eight sliders yourself.
- `Heuristic`: A built-in controller tries to keep the cell alive and growing.
- `Model`: A Stable-Baselines3 model drives the environment if loaded.

## Notes

- The previous repo mixed an old API and a newer "advanced" API. This rebuild removes that split and routes everything through one consistent simulation core.
- Legacy filenames still exist as thin wrappers so existing habits and commands do not break immediately.

## Running tests

```bash
python -m unittest discover tests -v
```

## License

MIT, same as the original repository.
