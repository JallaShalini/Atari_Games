import pytest
import numpy as np
from src.atari_env_wrapper import AtariEnvWrapper
from src.dqn_agent import DQNAgent
import torch

def test_env_wrapper():
    """Test AtariEnvWrapper initialization and basic operations"""
    env = AtariEnvWrapper("Pong-v5")
    state = env.reset()
    assert state.shape == (1, 4, 84, 84), f"Expected shape (1, 4, 84, 84), got {state.shape}"
    env.close()

def test_dqn_agent():
    """Test DQNAgent initialization and basic operations"""
    agent = DQNAgent(state_dim=(4, 84, 84), action_dim=6)
    state = np.random.randn(1, 4, 84, 84).astype(np.float32)
    q_values = agent.get_q_values(state)
    assert q_values.shape == (1, 6), f"Expected shape (1, 6), got {q_values.shape}"

def test_replay_buffer():
    """Test Experience Replay Buffer"""
    from src.dqn_agent import ExperienceReplayBuffer
    buffer = ExperienceReplayBuffer(capacity=100)
    state = np.random.randn(4, 84, 84).astype(np.float32)
    action = 0
    reward = 1.0
    next_state = np.random.randn(4, 84, 84).astype(np.float32)
    done = False
    buffer.push(state, action, reward, next_state, done)
    assert len(buffer) == 1
    batch = buffer.sample(1)
    assert batch is not None
