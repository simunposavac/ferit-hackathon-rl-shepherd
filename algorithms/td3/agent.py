"""TD3 agent implementation compatible with the project BaseAgent interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Type

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from actions import DogAction
from agents.base_agent import BaseAgent

from .replay_buffer import ReplayBuffer


@dataclass
class TD3AgentConfig:
    observation_shape: Sequence[int]
    actor_class: Type[nn.Module]
    critic_class: Type[nn.Module]
    metadata_dim: int = 2  # Size of metadata vector (2 for local, 6 for global)
    action_dim: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    batch_size: int = 128
    replay_capacity: int = 200_000
    exploration_noise: float = 0.1
    warmup_steps: int = 5_000
    steps_per_update: int = 1  # Do 1 update every N environment steps
    device: str = "cpu"


class TD3Agent(BaseAgent):
    """TD3 agent controlling a single actor via deterministic policy gradients."""

    def __init__(
        self,
        config: TD3AgentConfig,
        action_cls: Type[DogAction] = DogAction,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = config.action_dim
        self.action_cls = action_cls

        # Instantiate actor and critic using provided classes
        self.actor = config.actor_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)
        self.actor_target = config.actor_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)

        self.critic_1 = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)
        self.critic_2 = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)

        self.critic_target_1 = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)
        self.critic_target_2 = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=config.critic_lr)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=config.critic_lr)

        self.replay_buffer = ReplayBuffer(
            observation_shape=tuple(config.observation_shape),
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
            capacity=config.replay_capacity,
        )

        self.total_steps = 0
        self.update_step = 0
        self.training_mode = True

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    def act(self, observation: np.ndarray, metadata_vector: np.ndarray):
        obs_np = np.asarray(observation, dtype=np.float32)
        metadata_np = np.asarray(metadata_vector, dtype=np.float32)
        observation_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        metadata_tensor = torch.from_numpy(metadata_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(observation_tensor, metadata_tensor)
        if self.training_mode and self.config.exploration_noise > 0.0:
            noise = np.random.normal(0.0, self.config.exploration_noise, size=self.action_dim)
            noise_tensor = torch.from_numpy(noise.astype(np.float32)).to(self.device)
            action = torch.clamp(action + noise_tensor, -1.0, 1.0)
        action_np = action.squeeze(0).cpu().numpy()

        env_action_vector = self._normalized_to_env_action(action_np)
        return self.action_cls.from_vector(env_action_vector)

    def observe(
        self,
        observation: np.ndarray,
        metadata_vector: np.ndarray,
        action,
        reward: float,
        next_observation: np.ndarray,
        next_metadata_vector: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        action_vector = action.to_vector()
        normalized_action = self._env_to_normalized_action(action_vector)

        self.replay_buffer.add(
            observation,
            metadata_vector,
            normalized_action,
            reward,
            next_observation,
            next_metadata_vector,
            done,
        )

        self.total_steps += 1
        if self.total_steps < self.config.warmup_steps:
            return

        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Update only every N steps (decouples SGD frequency from env FPS)
        if self.total_steps % self.config.steps_per_update == 0:
            self._update()

    def episode_start(self):
        pass

    def episode_end(self, total_reward: float):
        pass

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_target_1": self.critic_target_1.state_dict(),
            "critic_target_2": self.critic_target_2.state_dict(),
            "replay": self.replay_buffer.to_dict(),
            "total_steps": self.total_steps,
            "update_step": self.update_step,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = TD3AgentConfig(**checkpoint["config"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_target_1.load_state_dict(checkpoint["critic_target_1"])
        self.critic_target_2.load_state_dict(checkpoint["critic_target_2"])

        replay_data = checkpoint.get("replay")
        if replay_data is not None:
            self._load_replay(replay_data)
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_step = checkpoint.get("update_step", 0)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_eval_mode(self, eval_mode: bool) -> None:
        self.training_mode = not eval_mode
        if eval_mode:
            self.actor.eval()
        else:
            self.actor.train()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update(self) -> None:
        sample = self.replay_buffer.sample(self.config.batch_size)

        observations = torch.from_numpy(sample.observations).float().to(self.device)
        metadata_vectors = torch.from_numpy(sample.metadata_vectors).float().to(self.device)
        actions = torch.from_numpy(sample.actions).float().to(self.device)
        rewards = torch.from_numpy(sample.rewards).float().to(self.device)
        next_observations = torch.from_numpy(sample.next_observations).float().to(self.device)
        next_metadata_vectors = torch.from_numpy(sample.next_metadata_vectors).float().to(self.device)
        dones = torch.from_numpy(sample.dones).float().to(self.device)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.policy_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = self.actor_target(next_observations, next_metadata_vectors)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)

            target_q1 = self.critic_target_1(next_observations, next_metadata_vectors, next_actions)
            target_q2 = self.critic_target_2(next_observations, next_metadata_vectors, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_values = rewards + self.config.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic_1(observations, metadata_vectors, actions)
        current_q2 = self.critic_2(observations, metadata_vectors, actions)
        critic_loss = F.mse_loss(current_q1, target_values) + F.mse_loss(current_q2, target_values)

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        if self.update_step % self.config.policy_delay == 0:
            actor_actions = self.actor(observations, metadata_vectors)
            actor_loss = -self.critic_1(observations, metadata_vectors, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_target_1)
            self._soft_update(self.critic_2, self.critic_target_2)

        self.update_step += 1

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )

    def _env_to_normalized_action(self, action_vector: np.ndarray) -> np.ndarray:
        normalized_forward = np.clip(action_vector[0], 0.0, 1.0) * 2.0 - 1.0
        normalized_turn = np.clip(action_vector[1], -1.0, 1.0)
        return np.array([normalized_forward, normalized_turn], dtype=np.float32)

    def _normalized_to_env_action(self, normalized_action: np.ndarray) -> np.ndarray:
        forward = np.clip((normalized_action[0] + 1.0) * 0.5, 0.0, 1.0)
        turn = np.clip(normalized_action[1], -1.0, 1.0)
        return np.array([forward, turn], dtype=np.float32)

    def _load_replay(self, replay_data: dict) -> None:
        obs = replay_data.get("observations")
        # Handle both old (pen_vectors) and new (metadata_vectors) formats
        pen = replay_data.get("metadata_vectors")
        if pen is None:
            pen = replay_data.get("pen_vectors")
        actions = replay_data.get("actions")
        rewards = replay_data.get("rewards")
        next_obs = replay_data.get("next_observations")
        # Handle both old (next_pen_vectors) and new (next_metadata_vectors) formats
        next_pen = replay_data.get("next_metadata_vectors")
        if next_pen is None:
            next_pen = replay_data.get("next_pen_vectors")
        dones = replay_data.get("dones")

        if obs is None:
            return

        capacity = self.replay_buffer.capacity
        self.replay_buffer.position = 0
        self.replay_buffer.size = 0

        for i in range(len(obs)):
            if i >= capacity:
                break
            self.replay_buffer.add(
                obs[i],
                pen[i],
                actions[i],
                float(rewards[i, 0]),
                next_obs[i],
                next_pen[i],
                bool(dones[i, 0]),
            )

    def train(self) -> None:  # pragma: no cover - convenience alias
        self.set_eval_mode(False)

    def eval(self) -> None:  # pragma: no cover - convenience alias
        self.set_eval_mode(True)
