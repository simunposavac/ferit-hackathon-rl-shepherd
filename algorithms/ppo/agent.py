"""PPO agent implementation compatible with the project BaseAgent interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Type

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from actions import DogAction
from agents.base_agent import BaseAgent
from algorithms.ppo.rollout_buffer import RolloutBuffer


@dataclass
class PPOAgentConfig:
    observation_shape: Sequence[int]
    actor_class: Type[nn.Module]
    critic_class: Type[nn.Module]
    metadata_dim: int = 2  # Size of metadata vector (2 for local, 6 for global)
    action_dim: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95  # Lambda for Generalized Advantage Estimation
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Coefficient for value loss
    entropy_coef: float = 0.01  # Coefficient for entropy bonus
    max_grad_norm: float = 0.5  # Gradient clipping
    n_epochs: int = 10  # Number of epochs to train on each rollout
    batch_size: int = 64  # Mini-batch size for updates
    rollout_capacity: int = 2048  # Maximum steps per rollout
    device: str = "cpu"


class PPOAgent(BaseAgent):
    """PPO agent with clipped surrogate objective and GAE."""

    def __init__(
        self,
        config: PPOAgentConfig,
        action_cls: Type[DogAction] = DogAction,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = config.action_dim
        self.action_cls = action_cls

        # Instantiate actor (stochastic policy)
        self.actor = config.actor_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
        ).to(self.device)

        # Instantiate critic (value function)
        self.critic = config.critic_class(
            observation_shape=config.observation_shape,
            metadata_dim=config.metadata_dim,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            observation_shape=tuple(config.observation_shape),
            metadata_dim=config.metadata_dim,
            action_dim=config.action_dim,
            capacity=config.rollout_capacity,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        self.total_steps = 0
        self.update_count = 0
        self.training_mode = True

    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        """Select action using current policy."""
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        pen_tensor = torch.from_numpy(pen_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, log_std = self.actor(obs_tensor, pen_tensor)
            std = log_std.exp()
            
            if self.training_mode:
                # Sample from policy during training
                dist = torch.distributions.Normal(mean, std)
                action_tensor = dist.sample()
            else:
                # Use mean during evaluation
                action_tensor = mean
            
            # Tanh squashing to [-1, 1]
            action_tensor = torch.tanh(action_tensor)
        
        action = action_tensor.cpu().numpy()[0]
        return self.action_cls(action[0], action[1])

    def observe(
        self,
        observation: np.ndarray,
        pen_vector: np.ndarray,
        action,
        reward: float,
        next_observation: np.ndarray,
        next_pen_vector: np.ndarray,
        done: bool,
        info: dict,
    ):
        """Store transition in rollout buffer."""
        # The action we receive is already the executed action (after tanh squashing)
        # We need to store it as-is for the buffer
        action_array = np.array([action.forward_speed, action.turn_rate], dtype=np.float32)
        
        # Convert to tensors for computing value and log_prob
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        pen_tensor = torch.from_numpy(pen_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get value estimate
            value = self.critic(obs_tensor, pen_tensor).cpu().numpy()[0, 0]
            
            # Get log probability of the action that was taken
            mean, log_std = self.actor(obs_tensor, pen_tensor)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            
            # The action is in [-1, 1] (post-tanh), we need to invert tanh to get pre-tanh action
            # action_pre_tanh = atanh(action)
            action_tensor = torch.from_numpy(action_array).unsqueeze(0).to(self.device)
            
            # Clamp to avoid numerical issues with atanh at boundaries
            action_clamped = torch.clamp(action_tensor, -0.9999, 0.9999)
            action_pre_tanh = torch.atanh(action_clamped)
            
            # Log prob of pre-tanh action
            log_prob_pre_tanh = dist.log_prob(action_pre_tanh).sum(dim=-1)
            
            # Correction for tanh squashing
            log_prob = log_prob_pre_tanh - torch.log(1 - action_clamped.pow(2) + 1e-6).sum(dim=-1)
            log_prob = log_prob.cpu().numpy()[0]

        # Store in rollout buffer (store the actual executed action)
        self.rollout_buffer.add(
            observation=observation,
            metadata_vector=pen_vector,
            action=action_array,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )

        self.total_steps += 1

    def episode_end(self, total_reward: float):
        """Called at end of episode - triggers learning update."""
        # Compute returns and advantages
        # Last value = 0 since episode ended
        self.rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        
        # Perform PPO updates
        self._update_policy()
        
        # Clear buffer for next episode
        self.rollout_buffer.clear()

    def _update_policy(self):
        """Perform PPO policy update using collected rollout."""
        if len(self.rollout_buffer) == 0:
            return

        # Get all data from buffer
        all_obs, all_meta, all_actions, all_returns, all_advantages, all_old_log_probs = self.rollout_buffer.get()
        
        # Normalize advantages (improves stability)
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        # Multiple epochs of updates
        for epoch in range(self.config.n_epochs):
            # Sample mini-batches
            for batch in self.rollout_buffer.sample_batches(self.config.batch_size):
                obs_batch, meta_batch, actions_batch, returns_batch, advantages_batch, old_log_probs_batch = batch
                
                obs_batch = obs_batch.to(self.device)
                meta_batch = meta_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)
                returns_batch = returns_batch.to(self.device)
                advantages_batch = advantages_batch.to(self.device)
                old_log_probs_batch = old_log_probs_batch.to(self.device)

                # Normalize advantages (per batch)
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # Compute new log probabilities and values
                mean, log_std = self.actor(obs_batch, meta_batch)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                
                # Actions in buffer are post-tanh (in [-1, 1])
                # Need to invert tanh to get pre-tanh actions for log_prob calculation
                actions_clamped = torch.clamp(actions_batch, -0.9999, 0.9999)
                actions_pre_tanh = torch.atanh(actions_clamped)
                
                # Log prob of pre-tanh actions
                log_prob_pre_tanh = dist.log_prob(actions_pre_tanh).sum(dim=-1)
                
                # Correction for tanh squashing
                log_probs = log_prob_pre_tanh - torch.log(1 - actions_clamped.pow(2) + 1e-6).sum(dim=-1)
                
                # Entropy (for exploration bonus)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Value predictions
                values = self.critic(obs_batch, meta_batch).squeeze(-1)

                # PPO clipped surrogate objective
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values, returns_batch)

                # Total loss
                loss = actor_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy

                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.update_count += 1

    def train(self):
        """Set agent to training mode."""
        self.training_mode = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        """Set agent to evaluation mode."""
        self.training_mode = False
        self.actor.eval()
        self.critic.eval()

    def save(self, path: str):
        """Save agent state to disk."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "total_steps": self.total_steps,
                "update_count": self.update_count,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
        print(f"[PPO] Loaded checkpoint from {path}")
        print(f"[PPO] Total steps: {self.total_steps}, Updates: {self.update_count}")
