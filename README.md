# A3C Labyrinth Navigator

PyTorch implementation of **Asynchronous Advantage Actor-Critic (A3C)** for 3D maze navigation, based on [Mnih et al. 2016](https://arxiv.org/abs/1602.01783).

![Dashboard](screengrab.gif)

## Overview

This project trains an agent to navigate randomly generated 3D mazes using only visual input (84×84 RGB). The implementation follows the Labyrinth task from Section 5.4 of the paper:

- **Apples** (red 3D objects): +1 reward when collected
- **Portal** (purple ring): +10 reward, respawns agent, regenerates apples  
- **Episode**: Terminates after 60 seconds

The agent uses an LSTM to maintain memory across time steps, which is preserved across portal jumps but reset on episode termination.

## Architecture

| Layer | Output |
|-------|--------|
| Conv2D (8×8, stride 4) | 16 channels |
| Conv2D (4×4, stride 2) | 32 channels |
| Linear | 256 units |
| LSTMCell | 256 units |
| Actor head | action_space |
| Critic head | 1 |

**Total parameters**: ~1.2M (~5 MB saved)

## Installation

```bash
conda create -n chiefmazi python=3.10
conda activate chiefmazi
pip install torch torchvision gymnasium pyglet miniworld gradio pandas
```

## Usage

### Training with Dashboard

```bash
python app.py
```

Opens a Gradio dashboard at `http://localhost:7860` showing:
- Live agent view with visible apples and portal
- Training statistics (frames, episodes, FPS)
- Loss values (policy, value, entropy)
- Action probability distribution
- Score history plots

### Headless Training

```bash
python train.py
```

Prints progress to console without UI overhead.

### Evaluate Trained Model

```python
import torch
from model import A3C_Labyrinth_Net

model = A3C_Labyrinth_Net(action_space=3)
model.load_state_dict(torch.load("checkpoints/a3c_final.pt"))
model.eval()
```

## Project Structure

```
├── app.py            # Gradio training dashboard
├── train.py          # Headless training script
├── worker.py         # A3C worker process
├── model.py          # CNN-LSTM network
├── env_wrapper.py    # Labyrinth task wrapper
├── shared_optim.py   # Shared RMSprop optimizer
└── meshes/
    ├── apple.obj     # 3D apple model
    └── portal.obj    # 3D portal ring model
```

## Environment

The `LabyrinthWrapper` extends MiniWorld's Maze environment with:

| Object | Appearance | Reward | Effect |
|--------|-----------|--------|--------|
| Apple | Red 3D mesh | +1 | Disappears on collection |
| Portal | Purple ring | +10 | Respawns agent, regenerates apples |

Custom OBJ meshes are auto-copied to MiniWorld's mesh directory on first run.

## Hyperparameters

| Parameter | Value
|-----------|-------
| Workers | 16
| Learning rate | 1e-4
| Discount (γ) | 0.99
| Entropy weight (β) | 0.01
| n-step | 5
| Gradient clip | 40.0

## Implementation Details

1. **Visible collectibles**: Apples and portal are rendered as 3D objects using custom OBJ meshes, giving the agent visual cues for navigation.

2. **Shared memory**: Global model and optimizer statistics are shared across processes using `share_memory()`.

3. **LSTM state management**: Hidden state is detached at each update, reset on episode end, but preserved across portal jumps.

4. **Asynchronous updates**: Workers compute gradients locally and accumulate them to the global model (Hogwild-style).

5. **macOS compatibility**: Uses `spawn` multiprocessing for OpenGL compatibility.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Gymnasium
- MiniWorld
- Gradio


Leave am for Chief Mazi 🙈🔥😂