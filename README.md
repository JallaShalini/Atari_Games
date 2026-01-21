# DQN Agent for Atari Games

A comprehensive reinforcement learning system implementing Deep Q-Networks (DQN) to train an AI agent capable of playing Atari games.

## Project Overview

This project implements a production-grade MLOps pipeline around a DQN agent, including:
- Environment preprocessing and frame stacking
- DQN agent with target networks and experience replay
- Training loop with stability features
- REST API for inference
- Docker containerization
- Experiment tracking

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py --game Pong-v5
python scripts/evaluate.py --game Pong-v5
python scripts/play.py --game Pong-v5
```

## Documentation

See METHODOLOGY.md for implementation details.
