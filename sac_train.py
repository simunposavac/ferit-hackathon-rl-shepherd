"""SAC training entry point for the sheep herding simulation.

Single file to edit:
1. Define your Actor and Critic network classes for SAC
2. Define dog_reward_fn and wolf_reward_fn
3. Adjust training config and run
"""

from __future__ import annotations

import datetime
from typing import List, Sequence, Tuple, Dict, Any

import os
from collections import deque
import time
import pygame

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils import checkpoint

from agents.base_agent import RandomAgent
import config
from algorithms.sac.agent import BaseAgent
from actions import DogAction, WolfAction
from simulator import Simulator


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

SAVE_DIR = os.path.join('saves', 'sac')
SAVE_EVERY_EPISODES = 20  # Save more frequently for hackathon

FOLDER_NAME =  os.path.join(SAVE_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
LOAD_DOG_CHECKPOINT = False
DOG_CHECKPOINT_PATH = r"saves/sac/best_dog.pth"

LOAD_WOLF_CHECKPOINT = False
WOLF_CHECKPOINT_PATH = r"saves/sac/best_wolf.pth"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

EPISODES = 1000 # how many episodes to run (Longer - more training, you can checkpoint and resume)
MAX_STEPS_PER_EPISODE = 2000 # max steps per episode (increased for longer episodes)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DOG = True
TRAIN_WOLF = True


# SAC hyperparameters

# Learning rates for the policy, Q-networks, and temperature. Higher = faster but less stable. Start at
ACTOR_LR = 3e-4      # Actor (policy) learning rate
CRITIC_LR = 3e-4     # Critic (Q-network) learning rate
ALPHA_LR = 3e-4      # Temperature learning rate

GAMMA = 0.99         # Discount factor (importance of future rewards)
TAU = 0.005          # Target network soft-update rate. How fast the target critics track the main critics (0.005 - stable, 0.01 - tracks faster)
BATCH_SIZE = 2048    # [512 - 4096] MAXIMIZED FOR RTX 3060: Large batches = maximum GPU utilization (12GB VRAM can handle 2048 easily)
REPLAY_CAPACITY = 1_000_000 # [500k - 2M] Max transitions stored in replay buffer, Larger = more diverse data (optimized for GPU)
WARMUP_STEPS = 10_000 # Number of env steps collected before any learning (enough to fill replay buffer with large batches)
STEPS_PER_UPDATE = 1 # Do updates every N environment steps (OPTIMIZED: update every step for maximum GPU usage)
UPDATES_PER_STEP = 16 # Number of gradient updates per trigger (MAXIMIZED: many updates to fully saturate GPU, RTX 3060 can handle this)
TARGET_ENTROPY = None  # None => will set -action_dim below (-2 for 2D action) (more negative - stronger exploration)
AUTOMATIC_ENTROPY_TUNING = True # If True, learn temperature α to match TARGET_ENTROPY. (Leave True)

# Rendering
HEADLESS = True  # MAXIMIZED: Headless mode for maximum speed (no window = faster training)
RENDER_EVERY = 0  # 0 disables auto-render - THIS IS KEY FOR SPEED!

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

# Encodes the observation grid (channels × H × W) and concatenates the 2-D pen vector; outputs a feature vector.
# hidden_dim: Size of the MLP output. Higher = more capacity, slower. MAXIMIZED: 1024 for RTX 3060 to use full GPU capacity
class CNNFeature(nn.Module):
    """CNN feature extractor for grid observations - MAXIMIZED for RTX 3060 GPU utilization."""
    def __init__(self, observation_shape, pen_vec_dim=2, hidden_dim=1024):
        super().__init__()
        channels, height, width = observation_shape
        # Improved CNN with better spatial reduction
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Stride 2 for spatial reduction
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Stride 2 for more reduction
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        # Calculate output size after convs: H/4 * W/4 * 128
        conv_out = 128 * (height // 4) * (width // 4)
        self.mlp = nn.Sequential(
            nn.Linear(conv_out + pen_vec_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),  # Added layer norm for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, obs_grid, pen_vec):
        x = self.conv(obs_grid)
        x = torch.cat([x, pen_vec], dim=-1)
        x = self.mlp(x)
        return x

# Given features, outputs a Gaussian (μ, σ), samples, and squashes with tanh to keep actions in [-1,1]. Also returns corrected log_prob
# log_std_bounds: Clamp log-std to avoid extreme variances
class SquashedGaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy with reparameterization + log-prob correction."""
    def __init__(self, feature_net: CNNFeature, action_dim: int, log_std_bounds: Tuple[float, float]=(-5, 2)):
        super().__init__()
        self.feature = feature_net
        hid = self.feature.out_dim
        self.mu = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hid // 2, action_dim),
        )
        self.log_std = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hid // 2, action_dim),
        )
        self.log_std_min, self.log_std_max = log_std_bounds

    def forward(self, obs_grid, pen_vec):
        feat = self.feature(obs_grid, pen_vec)
        mu = self.mu(feat)
        log_std = self.log_std(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs_grid, pen_vec):
        mu, log_std = self.forward(obs_grid, pen_vec)
        std = log_std.exp()
        normal = Normal(mu, std)
        z = normal.rsample()                # reparameterization
        action = torch.tanh(z)
        # log_prob with tanh correction (SAC Appendix C)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_tanh = torch.tanh(mu)
        return action, log_prob, mu_tanh

# Two independent Q-networks (Q1, Q2) each with their own encoder
# SAC uses min(Q1,Q2) to reduce overestimation
class CriticQ(nn.Module):
    """Twin Q networks with their own encoders for stability."""
    def __init__(self, observation_shape, action_dim, pen_vec_dim=2, hidden_dim=384):
        super().__init__()
        self.feat1 = CNNFeature(observation_shape, pen_vec_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(self.feat1.out_dim + action_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.feat2 = CNNFeature(observation_shape, pen_vec_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(self.feat2.out_dim + action_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs_grid, pen_vec, action):
        f1 = self.feat1(obs_grid, pen_vec)
        f2 = self.feat2(obs_grid, pen_vec)
        q1 = self.q1(torch.cat([f1, action], dim=-1))
        q2 = self.q2(torch.cat([f2, action], dim=-1))
        return q1, q2


# ===============================
#         Replay Buffer
# ===============================

# Stores (obs, pen_vec, action, reward, next_obs, next_pen, done) for off-policy learning
# capacity: Max items (memory bound)
# sample(batch_size): Returns random batch tensors on device
class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape, pen_vec_dim=2, action_dim=2, device='cpu'):
        self.capacity = capacity
        self.device = device
        # OPTIMIZATION: Use pinned memory for faster CPU->GPU transfers on CUDA
        self.use_pinned_memory = (device == 'cuda')

        # Use pinned memory buffers if on GPU (faster CPU->GPU transfers)
        if self.use_pinned_memory:
            # Create pinned memory buffers using torch
            self.obs_buf = torch.zeros((capacity, *observation_shape), dtype=torch.float32, pin_memory=True).cpu().numpy()
            self.pen_buf = torch.zeros((capacity, pen_vec_dim), dtype=torch.float32, pin_memory=True).cpu().numpy()
            self.act_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, pin_memory=True).cpu().numpy()
            self.rew_buf = torch.zeros((capacity, 1), dtype=torch.float32, pin_memory=True).cpu().numpy()
            self.next_obs_buf = torch.zeros((capacity, *observation_shape), dtype=torch.float32, pin_memory=True).cpu().numpy()
            self.next_pen_buf = torch.zeros((capacity, pen_vec_dim), dtype=torch.float32, pin_memory=True).cpu().numpy()
            self.done_buf = torch.zeros((capacity, 1), dtype=torch.float32, pin_memory=True).cpu().numpy()
        else:
            # Standard numpy arrays for CPU
            self.obs_buf = np.zeros((capacity, *observation_shape), dtype=np.float32)
            self.pen_buf = np.zeros((capacity, pen_vec_dim), dtype=np.float32)
            self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
            self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
            self.next_obs_buf = np.zeros((capacity, *observation_shape), dtype=np.float32)
            self.next_pen_buf = np.zeros((capacity, pen_vec_dim), dtype=np.float32)
            self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, pen, act, rew, next_obs, next_pen, done):
        self.obs_buf[self.ptr] = obs
        self.pen_buf[self.ptr] = pen
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.next_pen_buf[self.ptr] = next_pen
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample batch with optimized GPU transfers (non-blocking for CUDA)."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # OPTIMIZATION: Use non-blocking transfers for CUDA (allows CPU-GPU overlap)
        pin_memory = (self.device == 'cuda')
        
        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device, pin_memory=pin_memory)
        pen = torch.as_tensor(self.pen_buf[idxs], device=self.device, pin_memory=pin_memory)
        act = torch.as_tensor(self.act_buf[idxs], device=self.device, pin_memory=pin_memory)
        rew = torch.as_tensor(self.rew_buf[idxs], device=self.device, pin_memory=pin_memory)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device, pin_memory=pin_memory)
        next_pen = torch.as_tensor(self.next_pen_buf[idxs], device=self.device, pin_memory=pin_memory)
        done = torch.as_tensor(self.done_buf[idxs], device=self.device, pin_memory=pin_memory)
        
        return obs, pen, act, rew, next_obs, next_pen, done

# ===============================
#             SAC Agent
# ===============================

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic agent implementing BaseAgent interface.
    Optimizations:
      - update_every, max_updates_per_step to decouple SGD from env FPS
      - AMP (mixed precision) on CUDA for actor/critic
    """
    def __init__(
        self,
        observation_shape,
        action_dim,
        agent_type="dog",
        device='cpu',
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        hidden_dim=384,
        buffer_capacity=200_000,
        batch_size=128,
        updates_per_step=1,      # kept for compatibility (unused when update_every set)
        update_every=8,          # do SGD every N env steps
        max_updates_per_step=1,  # at most K updates each trigger
        learning_starts=10_000,
        target_entropy=None,
        automatic_entropy_tuning=True,
        pen_vec_dim=2
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.max_updates_per_step = max_updates_per_step
        self.learning_starts = learning_starts
        self.agent_type = agent_type
        self.action_dim = action_dim
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.use_amp = (device == 'cuda')
        if self.use_amp:
            # OPTIMIZATION: Enable all GPU optimizations for RTX 3060
            torch.backends.cudnn.benchmark = True  # Fixed input sizes, faster convs (significant speedup)
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            # Enable TensorFloat-32 (TF32) for faster matmuls on Ampere GPUs (RTX 3060)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # OPTIMIZATION: Enable cuDNN heuristics for better performance
            torch.backends.cudnn.enabled = True
            # OPTIMIZATION: Use faster attention implementations if available (PyTorch 2.0+)
            try:
                if hasattr(torch.backends.cuda, 'sdp_kernel'):
                    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
            except Exception:
                pass  # Not available in older PyTorch versions
            # OPTIMIZATION: Optimize gradient scaler for large batches
            self.critic_scaler = torch.amp.GradScaler('cuda', init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5)
            self.actor_scaler = torch.amp.GradScaler('cuda', init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5)
        else:
            self.critic_scaler = None
            self.actor_scaler = None

        actor_feat = CNNFeature(observation_shape, pen_vec_dim, hidden_dim)
        self.actor = SquashedGaussianActor(actor_feat, action_dim).to(device)

        self.critic = CriticQ(observation_shape, action_dim, pen_vec_dim, hidden_dim).to(device)
        self.critic_target = CriticQ(observation_shape, action_dim, pen_vec_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # OPTIMIZATION: Use torch.compile for PyTorch 2.0+ (significant speedup)
        # This requires PyTorch 2.0+ and can provide 20-30% speedup
        if device == 'cuda' and hasattr(torch, 'compile'):
            try:
                print(f"  [OPTIMIZATION] Using torch.compile for {agent_type} agent (PyTorch 2.0+)")
                self.actor = torch.compile(self.actor, mode='reduce-overhead')
                self.critic = torch.compile(self.critic, mode='reduce-overhead')
                # Note: critic_target doesn't need compilation as it's rarely used
            except Exception as e:
                print(f"  [WARNING] torch.compile not available or failed: {e}")
                print(f"  [INFO] Continuing without torch.compile")

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        if automatic_entropy_tuning:
            self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=False)
            self.alpha_opt = None

        self.replay = ReplayBuffer(buffer_capacity, observation_shape, pen_vec_dim, action_dim, device)
        self.total_steps = 0
        self._last_action_np = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _to_tensor_inputs(self, observation: np.ndarray, pen_vector: np.ndarray):
        """Optimized tensor conversion with non-blocking transfers for CUDA."""
        # OPTIMIZATION: Use pin_memory for faster CPU->GPU transfers on CUDA
        pin_memory = (self.device == 'cuda')
        obs_grid = torch.as_tensor(observation, dtype=torch.float32, device=self.device, pin_memory=pin_memory).unsqueeze(0)
        pen_vec = torch.as_tensor(pen_vector, dtype=torch.float32, device=self.device, pin_memory=pin_memory).unsqueeze(0)
        return obs_grid, pen_vec

    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        """Optimized action inference - actor stays in eval mode for faster inference."""
        # OPTIMIZATION: Keep actor in eval mode (no need to switch modes for inference)
        # Actor will be set to train() mode only during _update() for gradient computation
        if self.actor.training:
            self.actor.eval()
        
        with torch.no_grad():
            # OPTIMIZATION: Use float16 for faster inference on modern GPUs
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                obs_grid, pen_vec = self._to_tensor_inputs(observation, pen_vector)
                action_t, _, _ = self.actor.sample(obs_grid, pen_vec)
                action = action_t[0].cpu().numpy()

        # Map to env action semantics
        forward_speed = (float(action[0]) + 1.0) / 2.0
        turn_rate = float(action[1])

        self._last_action_np = action

        if self.agent_type == "dog":
            return DogAction(forward_speed=forward_speed, turn_rate=turn_rate)
        else:
            return WolfAction(forward_speed=forward_speed, turn_rate=turn_rate)

    def observe(self, observation: np.ndarray, pen_vector: np.ndarray,
                action, reward: float, next_observation: np.ndarray,
                next_pen_vector: np.ndarray, done: bool, info: dict):
        act_vec = action.to_vector().astype(np.float32)
        inv_a0 = np.clip(2.0 * act_vec[0] - 1.0, -1.0, 1.0)
        inv_a1 = np.clip(act_vec[1], -1.0, 1.0)
        a_store = np.array([inv_a0, inv_a1], dtype=np.float32)

        self.replay.add(
            observation.astype(np.float32),
            pen_vector.astype(np.float32),
            a_store,
            np.array([reward], dtype=np.float32),
            next_observation.astype(np.float32),
            next_pen_vector.astype(np.float32),
            float(done)
        )

        self.total_steps += 1

        # Begin SGD only after warmup, and not every step (decouples FPS from SGD)
        if self.replay.size >= self.learning_starts and (self.total_steps % self.update_every == 0):
            for _ in range(self.max_updates_per_step):
                self._update()

    def _update(self):
        """Optimized update with fused operations and better GPU utilization."""
        # OPTIMIZATION: Ensure actor is in training mode for gradient computation
        if not self.actor.training:
            self.actor.train()
        
        obs, pen, act, rew, next_obs, next_pen, done = self.replay.sample(self.batch_size)

        # ----- Critic update (with AMP when CUDA) -----
        # OPTIMIZATION: Use float16 for faster computation on modern GPUs
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16), torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs, next_pen)
            q1_targ, q2_targ = self.critic_target(next_obs, next_pen, next_action)
            q_targ_min = torch.min(q1_targ, q2_targ)
            target_v = q_targ_min - self.alpha * next_logp
            backup = rew + (1.0 - done) * self.gamma * target_v

        # OPTIMIZATION: Use set_to_none=True for faster zero_grad (PyTorch 1.7+)
        self.critic_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
            q1, q2 = self.critic(obs, pen, act)
            # OPTIMIZATION: Fused loss computation with explicit reduction
            critic_loss = nn.functional.mse_loss(q1, backup, reduction='mean') + nn.functional.mse_loss(q2, backup, reduction='mean')

        if self.critic_scaler is not None:
            self.critic_scaler.scale(critic_loss).backward()
            # OPTIMIZATION: Unscale before gradient clipping for numerical stability
            self.critic_scaler.unscale_(self.critic_opt)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)  # Higher clip for stability with large batches
            self.critic_scaler.step(self.critic_opt)
            self.critic_scaler.update()
        else:
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_opt.step()

        # ----- Actor update (with AMP) -----
        self.actor_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
            action_pi, logp_pi, _ = self.actor.sample(obs, pen)
            q1_pi, q2_pi = self.critic(obs, pen, action_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * logp_pi - q_pi).mean()

        if self.actor_scaler is not None:
            self.actor_scaler.scale(actor_loss).backward()
            self.actor_scaler.unscale_(self.actor_opt)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)  # Higher clip for stability
            self.actor_scaler.step(self.actor_opt)
            self.actor_scaler.update()
        else:
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()

        # ----- Temperature (alpha) update -----
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        # ----- Target network soft update (optimized with in-place operations) -----
        with torch.no_grad():
            # OPTIMIZATION: In-place operations are already used, keeping as-is for efficiency
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.tau)
                p_targ.data.add_(self.tau * p.data)

    def episode_start(self):
        pass

    def episode_end(self, total_reward: float):
        pass

    def save(self, path: str, include_replay: bool = False, max_replay_items: int = 0) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ckpt: Dict[str, Any] = {
            "meta": {
                "agent_type": self.agent_type,
                "action_dim": self.action_dim,
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "gamma": self.gamma,
                "tau": self.tau,
                "batch_size": self.batch_size,
                "update_every": self.update_every,
                "max_updates_per_step": self.max_updates_per_step,
                "learning_starts": self.learning_starts,
                "automatic_entropy_tuning": self.automatic_entropy_tuning,
                "target_entropy": self.target_entropy,
                "total_steps": self.total_steps,
                "device": str(self.device),
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": (self.alpha_opt.state_dict() if self.alpha_opt is not None else None),
            "critic_scaler": (self.critic_scaler.state_dict() if self.critic_scaler is not None else None),
            "actor_scaler": (self.actor_scaler.state_dict() if self.actor_scaler is not None else None),
        }

        if include_replay:
            N = self.replay.size
            if max_replay_items > 0:
                N = min(N, max_replay_items)
            start = (self.replay.ptr - N) % self.replay.capacity
            if N == 0:
                rb = None
            else:
                idxs = (np.arange(N) + start) % self.replay.capacity
                rb = {
                    "capacity": self.replay.capacity,
                    "size": int(N),
                    "ptr": int(N % self.replay.capacity),
                    "obs": self.replay.obs_buf[idxs],
                    "pen": self.replay.pen_buf[idxs],
                    "act": self.replay.act_buf[idxs],
                    "rew": self.replay.rew_buf[idxs],
                    "next_obs": self.replay.next_obs_buf[idxs],
                    "next_pen": self.replay.next_pen_buf[idxs],
                    "done": self.replay.done_buf[idxs],
                }
            ckpt["replay"] = rb

        torch.save(ckpt, path)

    def load(self, path: str, map_location: str | torch.device | None = None, strict: bool = True) -> None:
        if map_location is None:
            map_location = self.device

        ckpt = torch.load(path, map_location=map_location)

        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"], strict=strict)
        if "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"], strict=strict)

        if "critic_target" in ckpt and ckpt["critic_target"] is not None:
            self.critic_target.load_state_dict(ckpt["critic_target"], strict=strict)
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())

        if "actor_opt" in ckpt and ckpt["actor_opt"] is not None:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            # Move optimizer state tensors to the correct device
            for state in self.actor_opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if "critic_opt" in ckpt and ckpt["critic_opt"] is not None:
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
            for state in self.critic_opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if "log_alpha" in ckpt and ckpt["log_alpha"] is not None:
            with torch.no_grad():
                self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device).reshape_as(self.log_alpha.data))
        if "alpha_opt" in ckpt and ckpt["alpha_opt"] is not None and self.alpha_opt is not None:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
            for state in self.alpha_opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if self.critic_scaler is not None and ckpt.get("critic_scaler"):
            try:
                self.critic_scaler.load_state_dict(ckpt["critic_scaler"])
            except Exception:
                pass
        if self.actor_scaler is not None and ckpt.get("actor_scaler"):
            try:
                self.actor_scaler.load_state_dict(ckpt["actor_scaler"])
            except Exception:
                pass

        meta = ckpt.get("meta", {})
        self.total_steps = int(meta.get("total_steps", self.total_steps))

        rb = ckpt.get("replay", None)
        if rb is not None:
            cap = int(rb["capacity"])
            if cap == self.replay.capacity and rb["obs"].shape[1:] == self.replay.obs_buf.shape[1:]:
                N = int(rb["size"])
                self.replay.size = N
                self.replay.ptr = int(rb["ptr"])
                self.replay.obs_buf[:N] = rb["obs"]
                self.replay.pen_buf[:N] = rb["pen"]
                self.replay.act_buf[:N] = rb["act"]
                self.replay.rew_buf[:N] = rb["rew"]
                self.replay.next_obs_buf[:N] = rb["next_obs"]
                self.replay.next_pen_buf[:N] = rb["next_pen"]
                self.replay.done_buf[:N] = rb["done"]

        self.actor.train()
        self.critic.train()
        self.critic_target.eval()

# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

# ----------------------------------------------------------------------------
#                                    Guidelines
# You control the behavior with:
#   dog_reward_fn(info, prev_info)
#   wolf_reward_fn(info, prev_info)
# These run every step inside the simulator and return a single number (reward)
#
# Tips:
# - Keep numbers roughly in the same scale for dog and wolf (tens per step, hundreds per episode)
# - Use differences (current vs previous distances) to reward progress not just position
# - Use event bonuses (sheep enters pen, wolf dies) for big milestones
# - Start simple - with one main term (distance to pen for dog, distance to nearest sheep for wolf), then add events later.
# ----------------------------------------------------------------------------

def dog_reward_fn(info: Dict[str, any], prev_info: Dict[str, any]=None) -> float:
    """
    Reward function for the dog agent.
    Mission: Get sheep into pen and protect them from wolf (can eat wolf).
    
    Sparse rewards:
    - +200 per sheep entering pen (primary objective)
    - -150 per sheep eaten by wolf (major failure)
    - +100 for killing wolf (eliminates threat)
    
    Dense shaping rewards:
    - Progress reward: sheep moving toward pen (scaled by distance)
    - Herding reward: staying near sheep cluster (optimal distance)
    - Protective reward: positioning between wolf and sheep
    - Directional reward: pushing sheep toward pen entrance
    - Proximity to pen: reward when sheep are closer to pen
    """
    reward = 0.0

    # ========================================================================
    # SPARSE REWARDS (Main objectives)
    # ========================================================================
    sheep_entered = info.get("sheep_entered_pen", 0)
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    
    reward += float(sheep_entered) * 200.0  # Major reward for saving sheep
    reward -= float(sheep_eaten) * 150.0     # Major penalty when wolf eats sheep
    reward += float(wolf_killed) * 100.0     # Bonus for killing wolf (eliminates threat)
    
    # ========================================================================
    # DENSE SHAPING REWARDS (Help learning when sparse rewards are rare)
    # ========================================================================
    sheep_positions = info.get("sheep_positions", None)
    dog_pos = info.get("dog_position", None)
    wolf_pos = info.get("wolf_position", None)
    pen_center = info.get("pen_center", None)
    pen_entrance = info.get("pen_entrance_position", None)
    
    if sheep_positions is not None and len(sheep_positions) > 0 and dog_pos is not None and pen_center is not None:
        sheep_positions = np.array(sheep_positions)
        dog_pos = np.array(dog_pos)
        pen_center = np.array(pen_center)
        screen_width = config.SCREEN_WIDTH  # Define once at the start
        
        # Calculate distances
        sheep_dists_to_dog = np.linalg.norm(sheep_positions - dog_pos, axis=1)
        sheep_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
        avg_sheep_dist_to_dog = float(np.mean(sheep_dists_to_dog))
        avg_sheep_dist_to_pen = float(np.mean(sheep_dists_to_pen))
        min_sheep_dist_to_dog = float(np.min(sheep_dists_to_dog))
        
        # 1. PROGRESS REWARD: Reward sheep moving toward pen (stronger when closer)
        if prev_info is not None:
            prev_sheep_pos = prev_info.get("sheep_positions", None)
            if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                prev_sheep_pos = np.array(prev_sheep_pos)
                prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                progress = prev_dists_to_pen - sheep_dists_to_pen
                avg_progress = float(np.mean(progress))
                
                # Scale progress reward: more reward when sheep are closer to pen
                # (encourages finishing the job)
                progress_scale = 1.0 + (1.0 - min(avg_sheep_dist_to_pen / 600.0, 1.0))
                reward += avg_progress * 0.15 * progress_scale
        
        # 2. HERDING REWARD: Optimal distance to sheep cluster (not too close, not too far)
        # Optimal distance is around 150-200 pixels (allows dog to guide sheep)
        optimal_herding_dist = 175.0
        herding_dist_diff = abs(avg_sheep_dist_to_dog - optimal_herding_dist)
        herding_reward = max(0.0, 1.0 - (herding_dist_diff / optimal_herding_dist))
        reward += herding_reward * 0.12
        
        # Bonus for being close enough to influence sheep
        if avg_sheep_dist_to_dog < 250.0:
            reward += 0.05
        
        # 3. PROTECTIVE REWARD: Position between wolf and sheep (stronger)
        if wolf_pos is not None and not info.get("wolf_is_dead", False):
            wolf_pos = np.array(wolf_pos)
            wolf_to_dog_dist = float(np.linalg.norm(dog_pos - wolf_pos))
            avg_wolf_to_sheep_dist = float(np.mean(np.linalg.norm(sheep_positions - wolf_pos, axis=1)))
            
            # Strong reward if dog is between wolf and sheep
            if wolf_to_dog_dist < avg_wolf_to_sheep_dist:
                # Reward is stronger when dog is closer to the interception point
                interception_bonus = (1.0 - (wolf_to_dog_dist / max(avg_wolf_to_sheep_dist, 1.0))) * 0.2
                reward += interception_bonus
                
                # Extra bonus when very close to wolf (about to kill)
                if wolf_to_dog_dist < 60.0:
                    reward += 0.15
            else:
                # Small penalty if dog is on wrong side (encourages repositioning)
                if wolf_to_dog_dist > avg_wolf_to_sheep_dist * 1.5:
                    reward -= 0.05
        
        # 4. DIRECTIONAL REWARD: Pushing sheep toward pen entrance
        # Calculate if sheep are moving in the right direction (toward pen)
        if pen_entrance is not None:
            pen_x = pen_entrance[0]
            # Reward when sheep are moving toward pen (x-axis direction)
            sheep_x_positions = sheep_positions[:, 0]
            avg_sheep_x = float(np.mean(sheep_x_positions))
            
            # Reward based on how close sheep are to pen (x-coordinate)
            # Pen is on the right side, so higher x is better
            x_progress = (avg_sheep_x - (screen_width * 0.3)) / (screen_width * 0.7)  # Normalize
            x_progress = np.clip(x_progress, 0.0, 1.0)
            reward += x_progress * 0.08
        
        # 5. PROXIMITY TO PEN: Reward when sheep are closer to pen
        # Normalize distance (pen is roughly at x=screen_width-20)
        max_pen_dist = np.sqrt(screen_width**2 + config.SCREEN_HEIGHT**2)
        pen_proximity = 1.0 - min(avg_sheep_dist_to_pen / max_pen_dist, 1.0)
        reward += pen_proximity * 0.06
        
        # Bonus when sheep are very close to pen (about to enter)
        if avg_sheep_dist_to_pen < 100.0:
            reward += 0.1
    
    # Small time penalty (encourages faster completion)
    reward -= 0.01

    return reward

def wolf_reward_fn(info, prev_info=None) -> float:
    """
    Reward function for the wolf agent.
    Mission: Distract sheep from getting into pen and eat sheep.
    
    Sparse rewards:
    - +200 per sheep eaten (primary objective)
    - -120 for being killed (major failure)
    - -80 per sheep entering pen (dog succeeding - competing objective)
    
    Dense shaping rewards:
    - Chasing reward: progress toward nearest sheep (stronger when close)
    - Proximity reward: being close to sheep
    - Distraction reward: keeping sheep away from pen (NEW!)
    - Survival reward: avoiding dog when close
    - Bonus for very close to sheep (about to eat)
    """
    reward = 0.0

    # ========================================================================
    # SPARSE REWARDS (Main objectives)
    # ========================================================================
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    wolf_is_dead = info.get("wolf_is_dead", False)
    sheep_entered_pen = info.get("sheep_entered_pen", 0)
    
    reward += float(sheep_eaten) * 200.0      # Major reward for eating sheep
    reward -= float(wolf_killed) * 120.0      # Major penalty for dying
    reward -= float(sheep_entered_pen) * 80.0 # Penalty when dog succeeds (competing objective)
    
    # ========================================================================
    # DENSE SHAPING REWARDS (only when wolf is alive)
    # ========================================================================
    if not wolf_is_dead:
        sheep_positions = info.get("sheep_positions", None)
        wolf_pos = info.get("wolf_position", None)
        dog_pos = info.get("dog_position", None)
        pen_center = info.get("pen_center", None)
        pen_entrance = info.get("pen_entrance_position", None)
        
        if sheep_positions is not None and len(sheep_positions) > 0 and wolf_pos is not None:
            sheep_positions = np.array(sheep_positions)
            wolf_pos = np.array(wolf_pos)
            screen_width = config.SCREEN_WIDTH  # Define once at the start
            
            # Calculate distances to sheep
            sheep_dists = np.linalg.norm(sheep_positions - wolf_pos, axis=1)
            min_dist_to_sheep = float(np.min(sheep_dists))
            avg_dist_to_sheep = float(np.mean(sheep_dists))
            nearest_sheep_idx = int(np.argmin(sheep_dists))
            nearest_sheep_pos = sheep_positions[nearest_sheep_idx]
            
            # 1. CHASING REWARD: Progress toward nearest sheep (stronger when close)
            if prev_info is not None:
                prev_wolf_pos = prev_info.get("wolf_position", None)
                prev_sheep_pos = prev_info.get("sheep_positions", None)
                if prev_wolf_pos is not None and prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                    prev_wolf_pos = np.array(prev_wolf_pos)
                    prev_sheep_pos = np.array(prev_sheep_pos)
                    prev_dists = np.linalg.norm(prev_sheep_pos - prev_wolf_pos, axis=1)
                    prev_min_dist = float(np.min(prev_dists))
                    progress = prev_min_dist - min_dist_to_sheep
                    
                    # Scale progress reward: more reward when closer to sheep
                    # (encourages finishing the chase)
                    progress_scale = 1.0 + (1.0 - min(min_dist_to_sheep / 300.0, 1.0))
                    reward += progress * 0.12 * progress_scale
            
            # 2. PROXIMITY REWARD: Being close to sheep (stronger)
            # Normalized distance reward (closer = better, max distance ~600 pixels)
            proximity_reward = (1.0 - min(min_dist_to_sheep / 500.0, 1.0)) * 0.5
            reward += proximity_reward
            
            # Bonus for being very close to sheep (about to eat)
            if min_dist_to_sheep < 50.0:
                reward += 0.4
            elif min_dist_to_sheep < 80.0:
                reward += 0.2
            
            # 3. DISTRACTION REWARD: Keeping sheep away from pen (NEW!)
            if pen_center is not None:
                pen_center = np.array(pen_center)
                sheep_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
                avg_sheep_dist_to_pen = float(np.mean(sheep_dists_to_pen))
                
                # Reward when sheep are far from pen (wolf is succeeding at distraction)
                # Normalize: pen is roughly at x=screen_width-20, center is at x=screen_width/2
                max_pen_dist = np.sqrt(screen_width**2 + config.SCREEN_HEIGHT**2)
                distraction_score = min(avg_sheep_dist_to_pen / max_pen_dist, 1.0)
                reward += distraction_score * 0.15
                
                # Extra bonus when wolf is actively distracting (near sheep that are moving away from pen)
                if prev_info is not None:
                    prev_sheep_pos = prev_info.get("sheep_positions", None)
                    if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                        prev_sheep_pos = np.array(prev_sheep_pos)
                        prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                        # If sheep are moving away from pen (increasing distance), reward wolf
                        pen_regression = np.mean(sheep_dists_to_pen - prev_dists_to_pen)
                        if pen_regression > 0:  # Sheep moving away from pen
                            reward += pen_regression * 0.08
            
            # 4. DIRECTIONAL DISTRACTION: Push sheep away from pen entrance
            if pen_entrance is not None:
                pen_x = pen_entrance[0]
                sheep_x_positions = sheep_positions[:, 0]
                avg_sheep_x = float(np.mean(sheep_x_positions))
                
                # Reward when sheep are far from pen (left side of screen)
                # Penalize when sheep are close to pen (right side)
                x_distance_from_pen = (pen_x - avg_sheep_x) / screen_width
                x_distance_from_pen = np.clip(x_distance_from_pen, 0.0, 1.0)
                reward += x_distance_from_pen * 0.1
                
                # Bonus when wolf is between sheep and pen (blocking)
                if wolf_pos[0] > avg_sheep_x and wolf_pos[0] < pen_x:
                    reward += 0.15
            
            # 5. SURVIVAL REWARD: Avoid dog when it's close (strategic)
            if dog_pos is not None:
                dog_pos = np.array(dog_pos)
                dog_dist = float(np.linalg.norm(wolf_pos - dog_pos))
                
                # Strong penalty when too close to dog (danger zone)
                if dog_dist < 100.0:
                    # Exponential penalty: very dangerous when very close
                    danger_penalty = (1.0 - (dog_dist / 100.0)) ** 2 * 0.4
                    reward -= danger_penalty
                    
                    # If dog is between wolf and sheep, extra penalty (wolf is trapped)
                    if dog_dist < 60.0:
                        wolf_to_sheep_dir = nearest_sheep_pos - wolf_pos
                        dog_to_wolf_dir = wolf_pos - dog_pos
                        # Check if directions are opposite (wolf is blocked)
                        dot_product = np.dot(wolf_to_sheep_dir, dog_to_wolf_dir)
                        if dot_product > 0:  # Dog is blocking path to sheep
                            reward -= 0.2
                elif dog_dist < 150.0:  # Moderate danger zone
                    danger_penalty = (1.0 - (dog_dist / 150.0)) * 0.15
                    reward -= danger_penalty
                elif dog_dist > 200.0:
                    # Small reward for staying safe from dog
                    reward += 0.05
            
            # 6. STRATEGIC POSITIONING: Reward for being near sheep cluster center
            # (makes wolf more effective at scattering sheep)
            if len(sheep_positions) > 1:
                sheep_center = np.mean(sheep_positions, axis=0)
                dist_to_sheep_center = float(np.linalg.norm(wolf_pos - sheep_center))
                # Optimal distance: close enough to scatter, but not too close
                optimal_scatter_dist = 120.0
                scatter_reward = max(0.0, 1.0 - (dist_to_sheep_center / optimal_scatter_dist))
                reward += scatter_reward * 0.08
    
    # Small time penalty (encourages action)
    reward -= 0.01

    return reward

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train(
    num_episodes=1000,
    max_steps=2000,
    save_interval=50,
    device='cpu',
    headless=False,               # show window by default
    render_every_n_steps=20,      # render every 20 steps by default
    render_fps=0,                 # Frame cap when rendering (<=60)
    log_interval=1,               # Print progress every N episodes
):
    print("=" * 70)
    print("STARTING MULTI-AGENT SAC TRAINING (optimized)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    if not headless:
        status = "disabled" if render_every_n_steps == 0 else f"every {render_every_n_steps} steps"
        print(f"Rendering: {status} (press H to toggle 0 ↔ 20)")
        print(f"Controls: ESC to quit, H to toggle rendering")
    else:
        print("Headless mode: rendering disabled (no window will appear)")
    print()

    observation_shape = (config.OBSERVATION_CHANNELS, config.OBSERVATION_GRID_SIZE, config.OBSERVATION_GRID_SIZE)
    action_dim = 2

    print(f"Observation shape: {observation_shape}")
    print(f"Observation range: {config.OBSERVATION_RANGE} pixels")
    print()

    common_kwargs = dict(
        observation_shape=observation_shape,
        action_dim=action_dim,
        device=device,
        gamma=GAMMA,
        tau=TAU,
        lr=ACTOR_LR,
        hidden_dim=1024,  # MAXIMIZED: Large network to fully utilize RTX 3060 GPU (12GB VRAM can handle 1024 easily)
        buffer_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        updates_per_step=1,          # kept for compatibility
        update_every=STEPS_PER_UPDATE,  # train every N steps
        max_updates_per_step=UPDATES_PER_STEP,  # MAXIMIZED: 16 updates to saturate GPU with large batches
        learning_starts=WARMUP_STEPS,  # start learning after warmup
        automatic_entropy_tuning=AUTOMATIC_ENTROPY_TUNING,
        pen_vec_dim=2
    )

    if TRAIN_DOG or LOAD_DOG_CHECKPOINT:
        dog_agent = SACAgent(agent_type="dog", **common_kwargs)
        if LOAD_DOG_CHECKPOINT:
            print(f"Loading dog checkpoint from: {DOG_CHECKPOINT_PATH}")
            dog_agent.load(DOG_CHECKPOINT_PATH)
            print("Dog checkpoint loaded successfully!")
    else:
        dog_agent = RandomAgent(agent_type="dog", max_value=0)

    if TRAIN_WOLF or LOAD_WOLF_CHECKPOINT:
        wolf_agent = SACAgent(agent_type="wolf", **common_kwargs)
        if LOAD_WOLF_CHECKPOINT:
            print(f"Loading wolf checkpoint from: {WOLF_CHECKPOINT_PATH}")
            wolf_agent.load(WOLF_CHECKPOINT_PATH)
            print("Wolf checkpoint loaded successfully!")
    else:
        wolf_agent = RandomAgent(agent_type="wolf", max_value=0)

    simulator = Simulator(headless=headless, dog_reward_fn=dog_reward_fn, wolf_reward_fn=wolf_reward_fn)

    dog_rewards_history = deque(maxlen=100)
    wolf_rewards_history = deque(maxlen=100)
    sheep_saved_history = deque(maxlen=100)

    start_time = time.time()

    for episode in range(num_episodes):
        (dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec) = simulator.reset()

        # ensure window appears immediately when not headless
        if not simulator.headless and render_every_n_steps > 0:
            simulator.render(fps=render_fps)

        dog_agent.episode_start()
        wolf_agent.episode_start()

        dog_episode_reward = 0.0
        wolf_episode_reward = 0.0

        for step in range(max_steps):
            if not simulator.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nTraining interrupted by user")
                        simulator.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("\nTraining interrupted by user (ESC pressed)")
                            simulator.close()
                            return
                        elif event.key == pygame.K_h:
                            # Toggle 0 ↔ 20
                            render_every_n_steps = 1 if render_every_n_steps == 0 else 0
                            status = "disabled" if render_every_n_steps == 0 else f"every {render_every_n_steps} steps"
                            print(f"\nRendering {status} (press H to toggle)")
                            if render_every_n_steps > 0:
                                simulator.render(fps=render_fps)

                        elif event.key == pygame.K_d:
                            config.DEBUG_SHOW_OBSERVATIONS = not config.DEBUG_SHOW_OBSERVATIONS
                            config.DEBUG_SHOW_PEN_DIRECTION = not config.DEBUG_SHOW_PEN_DIRECTION

            dog_action = dog_agent.act(dog_obs, dog_pen_vec)
            wolf_action = wolf_agent.act(wolf_obs, wolf_pen_vec)

            (next_dog_obs, next_dog_pen_vec, dog_reward,
             next_wolf_obs, next_wolf_pen_vec, wolf_reward, info) = simulator.step(dog_action, wolf_action)

            if not simulator.headless and render_every_n_steps > 0 and (step % render_every_n_steps == 0):
                simulator.render(fps=render_fps)

            done = info['done']

            if TRAIN_DOG:
                dog_agent.observe(dog_obs, dog_pen_vec, dog_action, dog_reward,
                                next_dog_obs, next_dog_pen_vec, done, info)
            if TRAIN_WOLF:
                wolf_agent.observe(wolf_obs, wolf_pen_vec, wolf_action, wolf_reward,
                                next_wolf_obs, next_wolf_pen_vec, done, info)

            dog_episode_reward += dog_reward
            wolf_episode_reward += wolf_reward

            dog_obs = next_dog_obs
            dog_pen_vec = next_dog_pen_vec
            wolf_obs = next_wolf_obs
            wolf_pen_vec = next_wolf_pen_vec

            if done:
                break

        dog_agent.episode_end(dog_episode_reward)
        wolf_agent.episode_end(wolf_episode_reward)

        dog_rewards_history.append(dog_episode_reward)
        wolf_rewards_history.append(wolf_episode_reward)
        sheep_saved_history.append(simulator.env.sheep_in_pen)

        if (episode + 1) % log_interval == 0:
            avg_dog_reward = float(np.mean(dog_rewards_history)) if len(dog_rewards_history) > 0 else 0.0
            avg_wolf_reward = float(np.mean(wolf_rewards_history)) if len(wolf_rewards_history) > 0 else 0.0
            avg_sheep_saved = float(np.mean(sheep_saved_history)) if len(sheep_saved_history) > 0 else 0.0
            elapsed = time.time() - start_time

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Dog R: {avg_dog_reward:.2f} | "
                f"Wolf R: {avg_wolf_reward:.2f} | "
                f"Sheep Saved: {avg_sheep_saved:.1f} | "
                f"Steps: {step + 1} | "
                f"Dog Replay: {dog_agent.replay.size}/{dog_agent.replay.capacity} | " if TRAIN_DOG else ""
                f"Wolf Replay: {wolf_agent.replay.size}/{wolf_agent.replay.capacity} | " if TRAIN_WOLF else ""
                f"α(dog): {dog_agent.alpha.item():.3f} | " if TRAIN_DOG else ""
                f"α(wolf): {wolf_agent.alpha.item():.3f} | " if TRAIN_WOLF else ""
                f"Time: {elapsed:.1f}s"
            )

        if (episode + 1) % save_interval == 0:
            if TRAIN_DOG:
                dog_agent.save(f"{FOLDER_NAME}/dog_sac_episode_{episode + 1}.pth")
            if TRAIN_WOLF:
                wolf_agent.save(f"{FOLDER_NAME}/wolf_sac_episode_{episode + 1}.pth")
            print(f"Saved checkpoint at episode {episode + 1}")

    if TRAIN_DOG:
        dog_agent.save(f"{FOLDER_NAME}/dog_sac_final.pth")
    if TRAIN_WOLF:
        wolf_agent.save("{FOLDER_NAME}/wolf_sac_final.pth")
    print("\nTraining complete!")
    print(f"Final dog reward (100-ep avg): {np.mean(dog_rewards_history):.2f}")
    print(f"Final wolf reward (100-ep avg): {np.mean(wolf_rewards_history):.2f}")
    print(f"Final sheep saved (100-ep avg): {np.mean(sheep_saved_history):.1f}")

    simulator.close()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(f"{FOLDER_NAME}", exist_ok=True)


    train(
        num_episodes=EPISODES,
        max_steps=MAX_STEPS_PER_EPISODE,
        save_interval=SAVE_EVERY_EPISODES,
        device=device,
        headless=HEADLESS,          # Uses config from top (True = faster)
        render_every_n_steps=RENDER_EVERY,  # Uses config from top (0 = no render)
        render_fps=60,
        log_interval=1,
    )
