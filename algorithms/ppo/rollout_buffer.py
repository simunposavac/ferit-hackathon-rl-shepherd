"""Rollout buffer for on-policy algorithms like PPO."""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Generator


class RolloutBuffer:
    """
    Buffer for storing rollouts/trajectories for on-policy algorithms.
    
    Unlike replay buffers, this stores complete trajectories and is cleared
    after each update.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        metadata_dim: int,
        action_dim: int,
        capacity: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Args:
            observation_shape: Shape of spatial observation (C, H, W)
            metadata_dim: Dimension of metadata vector
            action_dim: Dimension of action space
            capacity: Maximum number of steps to store
            gamma: Discount factor for returns
            gae_lambda: Lambda for GAE (Generalized Advantage Estimation)
        """
        self.observation_shape = observation_shape
        self.metadata_dim = metadata_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage arrays
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.metadata_vectors = np.zeros((capacity, metadata_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)  # Value function predictions
        self.log_probs = np.zeros(capacity, dtype=np.float32)  # Log probabilities of actions
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Computed after rollout completion
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)

        self.pos = 0  # Current position in buffer
        self.size = 0  # Number of stored transitions
        self.episode_start = 0  # Start of current episode

    def add(
        self,
        observation: np.ndarray,
        metadata_vector: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        if self.pos >= self.capacity:
            raise RuntimeError("Rollout buffer is full. Call compute_returns_and_advantages() and clear().")
        
        self.observations[self.pos] = observation
        self.metadata_vectors[self.pos] = metadata_vector
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done

        self.pos += 1
        self.size = self.pos

    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        
        Args:
            last_value: Value estimate for the state after the last transition
                       (0 if episode ended, V(s') if truncated)
        """
        # Compute advantages using GAE
        last_gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE: A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Returns = advantages + values
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]

    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all stored transitions as PyTorch tensors.
        
        Returns:
            Tuple of (observations, metadata_vectors, actions, returns, advantages, log_probs)
        """
        assert self.size > 0, "Buffer is empty"
        
        # Return only the filled portion
        return (
            torch.from_numpy(self.observations[:self.size]),
            torch.from_numpy(self.metadata_vectors[:self.size]),
            torch.from_numpy(self.actions[:self.size]),
            torch.from_numpy(self.returns[:self.size]),
            torch.from_numpy(self.advantages[:self.size]),
            torch.from_numpy(self.log_probs[:self.size]),
        )

    def sample_batches(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate random mini-batches for training.
        
        Args:
            batch_size: Size of each mini-batch
            
        Yields:
            Tuples of (observations, metadata_vectors, actions, returns, advantages, old_log_probs)
        """
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.from_numpy(self.observations[batch_indices]),
                torch.from_numpy(self.metadata_vectors[batch_indices]),
                torch.from_numpy(self.actions[batch_indices]),
                torch.from_numpy(self.returns[batch_indices]),
                torch.from_numpy(self.advantages[batch_indices]),
                torch.from_numpy(self.log_probs[batch_indices]),
            )

    def clear(self):
        """Clear the buffer (called after policy update)."""
        self.pos = 0
        self.size = 0
        self.episode_start = 0

    def __len__(self) -> int:
        return self.size
