"""Replay buffer implementation for TD3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ReplaySample:
    observations: np.ndarray
    metadata_vectors: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    next_metadata_vectors: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size circular buffer storing transitions."""

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        metadata_dim: int,
        action_dim: int,
        capacity: int,
    ) -> None:
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.metadata_vectors = np.zeros((capacity, metadata_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.next_metadata_vectors = np.zeros((capacity, metadata_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        observation: np.ndarray,
        metadata_vector: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        next_metadata_vector: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.position
        self.observations[idx] = observation.astype(np.float32)
        self.metadata_vectors[idx] = metadata_vector.astype(np.float32)
        self.actions[idx] = action.astype(np.float32)
        self.rewards[idx] = np.array([reward], dtype=np.float32)
        self.next_observations[idx] = next_observation.astype(np.float32)
        self.next_metadata_vectors[idx] = next_metadata_vector.astype(np.float32)
        self.dones[idx] = np.array([float(done)], dtype=np.float32)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplaySample:
        if self.size < batch_size:
            raise ValueError("Replay buffer does not have enough samples")
        indices = np.random.choice(self.size, batch_size, replace=False)
        return ReplaySample(
            observations=self.observations[indices],
            metadata_vectors=self.metadata_vectors[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            next_metadata_vectors=self.next_metadata_vectors[indices],
            dones=self.dones[indices],
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[: self.size],
            "metadata_vectors": self.metadata_vectors[: self.size],
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "next_observations": self.next_observations[: self.size],
            "next_metadata_vectors": self.next_metadata_vectors[: self.size],
            "dones": self.dones[: self.size],
        }

    def __len__(self) -> int:  # pragma: no cover - simple accessor
        return self.size
