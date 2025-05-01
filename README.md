# 🧠 Artificial Intelligence Programming

## 📘 Project Overview

This repository contains coursework for the **Artificial Intelligence Programming** course.  
It consists of two main projects:

### ✅ Project 1: JAX Controller

Implementation of a classic PID controller and a neural-network-based controller using **JAX** for gradient-based
optimization.

---

### 🧪 Project 2: MuZero Knock-Off

An educational reimplementation of **MuZero**, a state-of-the-art reinforcement learning algorithm by Google Deepmind
that learns a model of the environment and uses MCTS for planning.

---

## 🛠 Usage

Commands to be run from the root directory of the repo.

### 🛠️ Setup Python Virtual Environment

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate

# Or activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🔁 Train MuZero

```bash
python3 -m project2.training.rl_manager
```

### 🎮 Play with Trained Agent & Generate GIF

```bash
python3 -m project2.playground.muzero_play_game
```

Make sure you have the file path for the trained model in the core/config.py file.

### 📊 Launch TensorBoard

```bash
python3 -m project2.training.run_tensorboard
```

### 🌲 Visualize Search Tree &#40;Perfect Model&#41;

```bash
python3 -m project2.playground.playground_mcts_perfect_model
```

### ✅ On Windows

You can also use "python -m project2..." as an alternative to "python3 -m project2...".

![MuZero](https://github.com/user-attachments/assets/5d458336-f129-40ff-8192-59433d5fa8b7)