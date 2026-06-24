# Cell Lab

Cell Lab is an interactive single-cell control sandbox. It combines a simplified biological simulation, a desktop UI, and an optional reinforcement learning training pipeline in one repo.

The current project focuses on one consistent model instead of the old mixed legacy structure. You can:

- drive the cell manually
- let the built-in heuristic controller manage it
- train a PPO/SAC policy and load that model back into the UI

## Features

- Live desktop UI with ATP, size, stress, and damage dashboards
- Organelles modeled as explicit subsystems:
  - mitochondria
  - ribosomes
  - nucleus
  - membrane
  - lysosome
  - ER
  - golgi
- Continuous control space with 8 action channels
- Gymnasium-compatible environment for Stable-Baselines3
- Automatic RL run folders with:
  - final model
  - best model
  - checkpoints
  - eval logs
  - TensorBoard logs

## Project Layout

```text
cell_lab/
  cli.py
  core.py
  env.py
  plotting.py
  policies.py
  training.py
  ui.py
main.py
requirements.txt
requirements-rl.txt
```

## Control Modes

`manual`
- You control all 8 action sliders yourself.
- Best for understanding how the simulation reacts.

`heuristic`
- A built-in rule-based controller manages the cell.
- Good for demos and quick baseline behavior.

`model`
- A trained model controls the cell.
- Use this after loading a compatible PPO/SAC/native model file.

## Actions

The simulator uses 8 continuous actions in the range `0.0` to `1.0`:

1. Nutrient uptake
2. Oxygen uptake
3. Glycolysis
4. Respiration
5. Protein synthesis
6. Repair
7. Detox
8. Division focus

## Organelles

Each organelle affects the cell state in a different way:

- `Mitochondria`: ATP efficiency and ROS pressure
- `Ribosomes`: protein synthesis capacity
- `Nucleus`: DNA repair quality and division readiness
- `Membrane`: nutrient intake efficiency and resistance to damage
- `Lysosome`: waste cleanup and autophagy-like detox behavior
- `ER`: protein folding stability and stress load
- `Golgi`: processing/export efficiency for growth-related synthesis

## Setup

### Base local usage

The core simulator and UI work with stock Python and the standard library.

```powershell
python main.py ui
```

### RL training environment

For real PPO/SAC training, this repo was validated with:

- Windows laptop
- Dell
- Intel `i9-13900HX`
- NVIDIA `RTX 4070 8 GB`

During testing on this machine, a PPO run of `200000` timesteps took roughly `4-5 minutes`, depending on current CPU load and logging overhead.

Create and use the Python 3.12 environment:

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements-rl.txt
```

Then run training with:

```powershell
.\.venv312\Scripts\python.exe main.py train --algorithm ppo --timesteps 200000 --run-name ppo_today_01 --eval-freq 10000 --checkpoint-freq 10000
```

## Training Outputs

Each RL run creates a folder like:

```text
models/ppo_today_01/
  cell_lab_ppo_final.zip
  best_model/
  checkpoints/

train_logs/ppo_today_01/
  eval/
  tensorboard/
  summary.json
```

### TensorBoard

```powershell
.\.venv312\Scripts\python.exe -m tensorboard.main --logdir train_logs
```

## Typical Workflow

### 1. Train

```powershell
.\.venv312\Scripts\python.exe main.py train --algorithm ppo --timesteps 200000 --run-name ppo_today_01 --eval-freq 10000 --checkpoint-freq 10000
```

### 2. Open the UI

```powershell
.\.venv312\Scripts\python.exe main.py ui
```

### 3. Load a trained model

Try these in order:

- `models\ppo_today_01\best_model\best_model.zip`
- `models\ppo_today_01\cell_lab_ppo_final.zip`

## Native Trainer

If you do not want to use PyTorch / SB3, there is also a lightweight local trainer:

```powershell
python main.py train --algorithm native --timesteps 5000
```

This creates a simple JSON policy by cloning the heuristic behavior.

## Notes

- The repo keeps legacy wrappers like `gui.py` and `train_advanced.py` so older habits do not break immediately.
- Training artifacts are ignored by git by default.
- The current simulation is intentionally stylized and educational, not a scientific cell biology simulator.

## Tests

```powershell
python -m unittest discover tests -v
```

## License

MIT
