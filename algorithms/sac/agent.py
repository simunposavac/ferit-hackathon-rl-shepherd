"""SAC agent implementation compatible with the project BaseAgent interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, Type

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from actions import DogAction
from agents.base_agent import BaseAgent
from algorithms.td3.replay_buffer import ReplayBuffer  # Reuse TD3's replay buffer


@dataclass
class SACAgentConfig:
    observation_shape: Sequence[int]
    actor_class: Type[nn.Module]
    critic_class: Type[nn.Module]
    metadata_dim: int = 2  # Size of metadata vector (2 for local, 6 for global)
    action_dim: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4  # Temperature parameter learning rate
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    replay_capacity: int = 200_000
    warmup_steps: int = 10_000
    steps_per_update: int = 1  # Do 1 update every N environment steps
    target_entropy: float = -2.0  # Target entropy for automatic tuning (usually -action_dim)
    automatic_entropy_tuning: bool = True
    device: str = "cpu"


class SACAgent(BaseAgent):
    """SAC agent with automatic entropy tuning and stochastic policy."""

    def __init__(
        self,
        config: SACAgentConfig,
        action_cls: Type[DogAction] = DogAction,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = config.action_dim
        self.action_cls = action_cls

        # Instantiate stochastic actor
        self.actor = config.actor_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)

        # Instantiate twin critics
        self.critic = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)
        
        # Target critic (no target actor in SAC)
        self.critic_target = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Automatic entropy tuning
        self.automatic_entropy_tuning = config.automatic_entropy_tuning
        self.target_entropy = config.target_entropy
        
        if self.automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            # Fixed alpha
            self.log_alpha = torch.tensor(np.log(0.2), device=self.device, requires_grad=False)
            self.alpha_optimizer = None

        # Replay buffer (reuse from TD3)
        self.replay_buffer = ReplayBuffer(
            observation_shape=tuple(config.observation_shape),
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
            capacity=config.replay_capacity,
        )

        self.total_steps = 0
        self.update_step = 0
        self.training_mode = True

    @property
    def alpha(self):
        """Temperature parameter (controls exploration vs exploitation)."""
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    def act(self, observation: np.ndarray, metadata_vector: np.ndarray):
        obs_np = np.asarray(observation, dtype=np.float32)
        metadata_np = np.asarray(metadata_vector, dtype=np.float32)
        observation_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        metadata_tensor = torch.from_numpy(metadata_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.training_mode:
                # Sample stochastic action during training
                action, _, _ = self.actor.sample(observation_tensor, metadata_tensor)
            else:
                # Use mean action during evaluation
                _, _, action = self.actor.sample(observation_tensor, metadata_tensor)
        
        action_np = action.squeeze(0).cpu().numpy()

        # Convert from [-1, 1] to environment action space
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
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "replay": self.replay_buffer.to_dict(),
            "total_steps": self.total_steps,
            "update_step": self.update_step,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = SACAgentConfig(**checkpoint["config"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        
        # Load temperature parameter
        log_alpha_val = checkpoint.get("log_alpha", np.log(0.2))
        if isinstance(log_alpha_val, np.ndarray):
            log_alpha_val = float(log_alpha_val[0])
        self.log_alpha.data = torch.tensor([log_alpha_val], device=self.device)

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
            self.critic.eval()
        else:
            self.actor.train()
            self.critic.train()

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

        # ----- Critic update -----
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_observations, next_metadata_vectors)
            
            # Compute target Q-values
            q1_target, q2_target = self.critic_target(next_observations, next_metadata_vectors, next_actions)
            q_target = torch.min(q1_target, q2_target)
            
            # V(s') = Q(s', a') - α * log π(a'|s')
            v_target = q_target - self.alpha * next_log_probs
            
            # Bellman backup
            backup = rewards + self.config.gamma * (1.0 - dones) * v_target

        # Current Q estimates
        q1_current, q2_current = self.critic(observations, metadata_vectors, actions)
        
        # MSE loss for both critics
        critic_loss = F.mse_loss(q1_current, backup) + F.mse_loss(q2_current, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----- Actor update -----
        # Sample actions from current policy
        sampled_actions, log_probs, _ = self.actor.sample(observations, metadata_vectors)
        
        # Compute Q-values for sampled actions
        q1_pi, q2_pi = self.critic(observations, metadata_vectors, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        
        # Actor loss: maximize Q - α * log π (minimize negative)
        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ----- Temperature (alpha) update -----
        if self.automatic_entropy_tuning:
            # Temperature loss: α * (log π + target_entropy)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ----- Target network soft update -----
        self._soft_update(self.critic, self.critic_target)

        self.update_step += 1

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )

    def _env_to_normalized_action(self, action_vector: np.ndarray) -> np.ndarray:
        """Convert environment action [0,1], [-1,1] to network output [-1,1], [-1,1]."""
        normalized_forward = np.clip(action_vector[0], 0.0, 1.0) * 2.0 - 1.0
        normalized_turn = np.clip(action_vector[1], -1.0, 1.0)
        return np.array([normalized_forward, normalized_turn], dtype=np.float32)

    def _normalized_to_env_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """Convert network output [-1,1], [-1,1] to environment action [0,1], [-1,1]."""
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
