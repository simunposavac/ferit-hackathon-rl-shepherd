"""DQN training entry point for the sheep herding simulation.

FIXED VERSION - Improvements:
1. Per-step epsilon decay with linear annealing
2. Soft target network updates (polyak averaging)
3. Increased updates per step for better sample efficiency
4. LayerNorm instead of BatchNorm for stability
5. Better hyperparameter balance
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

from agents.base_agent import RandomAgent
import config
from algorithms.sac.agent import BaseAgent
from actions import DogAction, WolfAction
from simulator import Simulator


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

SAVE_DIR = os.path.join('saves', 'dqn')
SAVE_EVERY_EPISODES = 20

FOLDER_NAME = os.path.join(SAVE_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
LOAD_DOG_CHECKPOINT = True
DOG_CHECKPOINT_PATH = r"saves/dqn/best_dog.pth"

LOAD_WOLF_CHECKPOINT = True
WOLF_CHECKPOINT_PATH = r"saves/dqn/best_wolf.pth"


def find_latest_checkpoint(save_dir: str, agent_type: str) -> str | None:
    """
    Find the latest checkpoint file for an agent by searching all timestamped folders.
    Prefers the most recently modified checkpoint (by file modification time).
    
    Args:
        save_dir: Directory to search (e.g., 'saves/dqn')
        agent_type: 'dog' or 'wolf'
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(save_dir):
        return None
    
    # Pattern: {agent_type}_dqn_episode_{number}.pth or {agent_type}_dqn_final.pth
    final_pattern = f"{agent_type}_dqn_final.pth"
    
    all_checkpoints = []  # List of (path, episode_num, mtime, is_final)
    
    # Search in all subdirectories (timestamp folders)
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            path = os.path.join(root, file)
            if not os.path.exists(path):
                continue
                
            mtime = os.path.getmtime(path)
            
            if file == final_pattern:
                # Final checkpoint - treat as episode number 999999 for sorting
                all_checkpoints.append((path, 999999, mtime, True))
            elif file.startswith(f"{agent_type}_dqn_episode_") and file.endswith(".pth"):
                # Extract episode number
                try:
                    ep_num = int(file.split("_")[-1].replace(".pth", ""))
                    all_checkpoints.append((path, ep_num, mtime, False))
                except ValueError:
                    continue
    
    if not all_checkpoints:
        return None
    
    # Sort by: 1) modification time (newest first), 2) episode number (higher first)
    # This ensures we get the most recently saved checkpoint
    all_checkpoints.sort(key=lambda x: (x[2], x[1]), reverse=True)
    
    return all_checkpoints[0][0]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

EPISODES = 1000
MAX_STEPS_PER_EPISODE = 3600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DOG = True
TRAIN_WOLF = True

# FIXED: Improved DQN hyperparameters
LR = 3e-4
GAMMA = 0.99
TAU = 0.005         # FIXED: Now actually used for soft target updates
BATCH_SIZE = 128
REPLAY_CAPACITY = 250_000
WARMUP_STEPS = 3_000
UPDATE_EVERY = 4
MAX_UPDATES_PER_STEP = 2  # FIXED: Increased from 1 to 2 for better sample efficiency
TARGET_UPDATE_EVERY = 1   # FIXED: Update target every step (soft update), not every 100

# FIXED: Per-step epsilon decay with linear annealing
TOTAL_TRAINING_STEPS = EPISODES * MAX_STEPS_PER_EPISODE
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = int(TOTAL_TRAINING_STEPS * 0.5)  # Decay over first 50% of training

# Rendering
HEADLESS = False
RENDER_EVERY = 0

# ============================================================================
# DISCRETE ACTION SPACE
# ============================================================================

FORWARD_SPEEDS = np.array([0.0, 0.33, 0.67, 1.0])
TURN_RATES = np.array([-1.0, -0.33, 0.33, 1.0])
NUM_FORWARD_SPEEDS = len(FORWARD_SPEEDS)
NUM_TURN_RATES = len(TURN_RATES)
NUM_ACTIONS = NUM_FORWARD_SPEEDS * NUM_TURN_RATES


def action_to_discrete(forward_speed: float, turn_rate: float) -> int:
    """Convert continuous action to discrete action index."""
    forward_idx = np.argmin(np.abs(FORWARD_SPEEDS - forward_speed))
    turn_idx = np.argmin(np.abs(TURN_RATES - turn_rate))
    action_idx = forward_idx * NUM_TURN_RATES + turn_idx
    return int(action_idx)


def discrete_to_action(action_idx: int) -> Tuple[float, float]:
    """Convert discrete action index to continuous action."""
    forward_idx = action_idx // NUM_TURN_RATES
    turn_idx = action_idx % NUM_TURN_RATES
    forward_speed = float(FORWARD_SPEEDS[forward_idx])
    turn_rate = float(TURN_RATES[turn_idx])
    return forward_speed, turn_rate


# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

# FIXED: Replaced BatchNorm with LayerNorm for better RL stability
class CNNFeature(nn.Module):
    """CNN feature extractor for grid observations - FIXED with LayerNorm."""
    def __init__(self, observation_shape, pen_vec_dim=2, hidden_dim=512):
        super().__init__()
        channels, height, width = observation_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(8, 64),  # FIXED: GroupNorm instead of BatchNorm (works better in RL)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(16, 128),  # FIXED: GroupNorm
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 256),  # FIXED: GroupNorm
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, 256),  # FIXED: GroupNorm
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        conv_out = 256 * 4 * 4
        self.mlp = nn.Sequential(
            nn.Linear(conv_out + pen_vec_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, obs_grid, pen_vec):
        x = self.conv(obs_grid)
        x = torch.cat([x, pen_vec], dim=-1)
        x = self.mlp(x)
        return x


class DQN(nn.Module):
    """Deep Q-Network for discrete action space."""
    def __init__(self, observation_shape, pen_vec_dim=2, hidden_dim=512, num_actions=NUM_ACTIONS):
        super().__init__()
        self.feature = CNNFeature(observation_shape, pen_vec_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(self.feature.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, obs_grid, pen_vec):
        features = self.feature(obs_grid, pen_vec)
        q_values = self.q_head(features)
        return q_values


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape, pen_vec_dim=2, action_dim=1, device='cpu'):
        self.capacity = capacity
        self.device = device

        self.obs_buf = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.pen_buf = np.zeros((capacity, pen_vec_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.int64)
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
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device)
        pen = torch.as_tensor(self.pen_buf[idxs], device=self.device)
        act = torch.as_tensor(self.act_buf[idxs], device=self.device)
        rew = torch.as_tensor(self.rew_buf[idxs], device=self.device)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device)
        next_pen = torch.as_tensor(self.next_pen_buf[idxs], device=self.device)
        done = torch.as_tensor(self.done_buf[idxs], device=self.device)
        return obs, pen, act, rew, next_obs, next_pen, done


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent - FIXED VERSION
    - Per-step epsilon decay with linear annealing
    - Soft target updates (polyak averaging)
    - Increased updates per step
    - Better stability
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
        hidden_dim=512,
        buffer_capacity=250_000,
        batch_size=128,
        update_every=4,
        max_updates_per_step=2,
        target_update_every=1,
        learning_starts=3_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100000,
        pen_vec_dim=2,
        num_actions=NUM_ACTIONS,
        use_torch_compile=False
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau  # FIXED: Now used for soft updates
        self.batch_size = batch_size
        self.update_every = update_every
        self.max_updates_per_step = max_updates_per_step
        self.target_update_every = target_update_every
        self.learning_starts = learning_starts
        self.agent_type = agent_type
        self.action_dim = action_dim
        self.num_actions = num_actions

        # FIXED: Per-step epsilon decay with linear annealing
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.use_amp = (device == 'cuda')
        if self.use_amp:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        self.q_network = DQN(observation_shape, pen_vec_dim, hidden_dim, num_actions).to(device)
        self.target_network = DQN(observation_shape, pen_vec_dim, hidden_dim, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-7)

        self.replay = ReplayBuffer(buffer_capacity, observation_shape, pen_vec_dim, action_dim=1, device=device)
        self.total_steps = 0
        self.update_count = 0
        self._last_action_np = None

    def _update_epsilon(self):
        """FIXED: Per-step epsilon decay with linear annealing."""
        if self.total_steps < self.epsilon_decay_steps:
            # Linear annealing
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
                self.total_steps / self.epsilon_decay_steps
            )
        else:
            self.epsilon = self.epsilon_end

    def _to_tensor_inputs(self, observation: np.ndarray, pen_vector: np.ndarray):
        obs_grid = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        pen_vec = torch.as_tensor(pen_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs_grid, pen_vec

    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        self.q_network.eval()
        with torch.inference_mode():
            obs_grid, pen_vec = self._to_tensor_inputs(observation, pen_vector)
            q_values = self.q_network(obs_grid, pen_vec)
            
            if np.random.random() < self.epsilon:
                action_idx = np.random.randint(0, self.num_actions)
            else:
                action_idx = q_values.argmax().item()
        self.q_network.train()

        forward_speed, turn_rate = discrete_to_action(action_idx)
        self._last_action_np = action_idx

        if self.agent_type == "dog":
            return DogAction(forward_speed=forward_speed, turn_rate=turn_rate)
        else:
            return WolfAction(forward_speed=forward_speed, turn_rate=turn_rate)

    def observe(self, observation: np.ndarray, pen_vector: np.ndarray,
                action, reward: float, next_observation: np.ndarray,
                next_pen_vector: np.ndarray, done: bool, info: dict):
        act_vec = action.to_vector()
        action_idx = action_to_discrete(act_vec[0], act_vec[1])
        action_idx = np.array([action_idx], dtype=np.int64)

        self.replay.add(
            observation.astype(np.float32),
            pen_vector.astype(np.float32),
            action_idx,
            np.array([reward], dtype=np.float32),
            next_observation.astype(np.float32),
            next_pen_vector.astype(np.float32),
            float(done)
        )

        self.total_steps += 1
        self._update_epsilon()  # FIXED: Update epsilon every step

        # FIXED: Multiple updates per trigger for better sample efficiency
        if self.replay.size >= self.learning_starts and (self.total_steps % self.update_every == 0):
            for _ in range(self.max_updates_per_step):
                self._update()
            
            if self.update_count % 500 == 0 and self.device == 'cuda':
                torch.cuda.empty_cache()

    def _update(self):
        """FIXED: Uses soft target updates (polyak averaging)."""
        obs, pen, act, rew, next_obs, next_pen, done = self.replay.sample(self.batch_size)

        # Compute target Q-values
        with torch.amp.autocast('cuda', enabled=self.use_amp), torch.no_grad():
            next_q_values = self.target_network(next_obs, next_pen)
            next_q_max = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rew + (1.0 - done) * self.gamma * next_q_max

        # Compute current Q-values and loss
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            q_values = self.q_network(obs, pen)
            q_selected = q_values.gather(1, act.long())
            loss = F.mse_loss(q_selected, target_q)

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.optimizer.step()

        self.update_count += 1

        # FIXED: Soft target network update (polyak averaging) every step
        if self.update_count % self.target_update_every == 0:
            with torch.no_grad():
                for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

    def episode_start(self):
        pass

    def episode_end(self, total_reward: float):
        pass  # FIXED: Epsilon decay now happens per-step, not per-episode

    def save(self, path: str, include_replay: bool = False, max_replay_items: int = 0) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ckpt: Dict[str, Any] = {
            "meta": {
                "agent_type": self.agent_type,
                "action_dim": self.action_dim,
                "num_actions": self.num_actions,
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "gamma": self.gamma,
                "tau": self.tau,
                "batch_size": self.batch_size,
                "update_every": self.update_every,
                "max_updates_per_step": self.max_updates_per_step,
                "target_update_every": self.target_update_every,
                "learning_starts": self.learning_starts,
                "epsilon": self.epsilon,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay_steps": self.epsilon_decay_steps,
                "total_steps": self.total_steps,
                "update_count": self.update_count,
                "device": str(self.device),
            },
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": (self.scaler.state_dict() if self.scaler is not None else None),
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

        if "q_network" in ckpt:
            self.q_network.load_state_dict(ckpt["q_network"], strict=strict)
        if "target_network" in ckpt:
            self.target_network.load_state_dict(ckpt["target_network"], strict=strict)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if self.scaler is not None and ckpt.get("scaler"):
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass

        meta = ckpt.get("meta", {})
        self.total_steps = int(meta.get("total_steps", self.total_steps))
        self.update_count = int(meta.get("update_count", self.update_count))
        self.epsilon = float(meta.get("epsilon", self.epsilon))
        self.epsilon_decay_steps = int(meta.get("epsilon_decay_steps", self.epsilon_decay_steps))

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

        self.q_network.train()
        self.target_network.eval()


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def dog_reward_fn(info: Dict[str, any], prev_info: Dict[str, any] = None) -> float:
    """
    Dog reward function: Get most sheep into pen, protect from wolf.
    Mission: Guide sheep to pen while avoiding entering cluster and protecting from wolf.
    """
    reward = 0.0

    # Get state information
    sheep_entered = info.get("sheep_entered_pen", 0)
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    sheep_positions = info.get("sheep_positions", None)
    dog_pos = info.get("dog_position", None)
    dog_velocity = info.get("dog_velocity", None)
    wolf_pos = info.get("wolf_position", None)
    pen_center = info.get("pen_center", None)
    
    # ========================================================================
    # 1. SHEEP ENTERING PEN REWARD (with FOV check)
    # ========================================================================
    if sheep_entered > 0:
        # Check if dog can see the pen (within FOV/distance)
        dog_can_see_pen = False
        if dog_pos is not None and pen_center is not None:
            dog_to_pen = pen_center - dog_pos
            dist_to_pen = np.linalg.norm(dog_to_pen)
            max_vision_range = 350.0  # FOV range
            if dist_to_pen < max_vision_range:
                dog_can_see_pen = True
        
        if dog_can_see_pen:
            reward += float(sheep_entered) * 110.0  # Full reward
        else:
            reward += float(sheep_entered) * 110.0 * 0.15  # Only 15% if not seen
    
    # ========================================================================
    # 2. WOLF KILLED REWARD
    # ========================================================================
    if wolf_killed:
        reward += 70.0
    
    # ========================================================================
    # 3. PENALTY FOR SHEEP EATEN
    # ========================================================================
    if sheep_eaten > 0:
        reward -= float(sheep_eaten) * 120.0
    
    if sheep_positions is not None and len(sheep_positions) > 0 and dog_pos is not None and pen_center is not None:
        # ========================================================================
        # 4. REWARD: Dog vector close to collinear with vector from dog to pen
        # ========================================================================
        if dog_velocity is not None and np.linalg.norm(dog_velocity) > 0.1:
            dog_velocity_normalized = dog_velocity / np.linalg.norm(dog_velocity)
            dog_to_pen = pen_center - dog_pos
            dog_to_pen_norm = np.linalg.norm(dog_to_pen)
            
            if dog_to_pen_norm > 1.0:
                dog_to_pen_normalized = dog_to_pen / dog_to_pen_norm
                alignment = np.dot(dog_velocity_normalized, dog_to_pen_normalized)
                # Reward when close to collinear (alignment close to 1.0)
                if alignment > 0.7:  # Close to collinear
                    reward += alignment * 0.3  # Stronger reward for better alignment
        
        # ========================================================================
        # 5. CLUSTER DETECTION AND PENALTY: Punish dog if entering cluster
        # ========================================================================
        # Calculate cluster center and radius (using smaller radius for manipulation)
        if len(sheep_positions) >= 2:
            cluster_center = np.mean(sheep_positions, axis=0)
            sheep_dists_to_cluster = np.linalg.norm(sheep_positions - cluster_center, axis=1)
            actual_cluster_radius = float(np.max(sheep_dists_to_cluster)) if len(sheep_dists_to_cluster) > 0 else 0.0
            
            # Use smaller radius (70% of actual) to allow manipulation but prevent splitting
            effective_cluster_radius = actual_cluster_radius * 0.7
            
            dog_dist_to_cluster = float(np.linalg.norm(dog_pos - cluster_center))
            
            # Penalty if dog enters inside the effective cluster radius
            if dog_dist_to_cluster < effective_cluster_radius:
                # Stronger penalty the deeper inside the cluster
                penetration = 1.0 - (dog_dist_to_cluster / max(effective_cluster_radius, 1.0))
                reward -= penetration * 0.5  # Penalty for entering cluster
        
        # ========================================================================
        # 6. REWARD: Dog coming to wolf when wolf is eating sheep
        # ========================================================================
        if wolf_pos is not None and not info.get("wolf_is_dead", False):
            wolf_to_dog = np.linalg.norm(dog_pos - wolf_pos)
            wolf_is_eating = info.get("wolf_just_ate", False) or sheep_eaten > 0
            
            if wolf_is_eating and wolf_to_dog < 200.0:  # Wolf is eating and dog is close
                # Reward for being close to wolf when it's eating
                proximity_reward = (1.0 - wolf_to_dog / 200.0) * 0.4
                reward += proximity_reward
                
                # Additional reward if dog is moving toward wolf
                if dog_velocity is not None and np.linalg.norm(dog_velocity) > 0.1:
                    dog_to_wolf = wolf_pos - dog_pos
                    dog_to_wolf_norm = np.linalg.norm(dog_to_wolf)
                    if dog_to_wolf_norm > 1.0:
                        dog_to_wolf_normalized = dog_to_wolf / dog_to_wolf_norm
                        dog_velocity_normalized = dog_velocity / np.linalg.norm(dog_velocity)
                        alignment_to_wolf = np.dot(dog_velocity_normalized, dog_to_wolf_normalized)
                        if alignment_to_wolf > 0.5:  # Moving toward wolf
                            reward += alignment_to_wolf * 0.3
        
        # ========================================================================
        # 7. PENALTY: Dog pushing sheep perpendicular from pen vector
        # ========================================================================
        if dog_velocity is not None and np.linalg.norm(dog_velocity) > 0.1 and prev_info is not None:
            dog_velocity_normalized = dog_velocity / np.linalg.norm(dog_velocity)
            dog_to_pen = pen_center - dog_pos
            dog_to_pen_norm = np.linalg.norm(dog_to_pen)
            
            if dog_to_pen_norm > 1.0:
                dog_to_pen_normalized = dog_to_pen / dog_to_pen_norm
                
                # Check if sheep are in front of dog
                prev_sheep_pos = prev_info.get("sheep_positions", None)
                if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                    sheep_in_front = 0
                    for i, sheep_pos in enumerate(sheep_positions):
                        dog_to_sheep = sheep_pos - dog_pos
                        dist_to_sheep = np.linalg.norm(dog_to_sheep)
                        if dist_to_sheep > 1.0:
                            dog_to_sheep_normalized = dog_to_sheep / dist_to_sheep
                            front_score = np.dot(dog_velocity_normalized, dog_to_sheep_normalized)
                            if front_score > 0.3 and dist_to_sheep < 250.0:  # Sheep in front
                                sheep_in_front += 1
                                
                                # Check if dog is pushing perpendicular to pen direction
                                perpendicular_score = abs(np.dot(dog_velocity_normalized, 
                                                                  np.array([-dog_to_pen_normalized[1], 
                                                                           dog_to_pen_normalized[0]])))
                                if perpendicular_score > 0.7:  # Close to perpendicular
                                    # Check if sheep moved away from pen
                                    prev_dist_to_pen = np.linalg.norm(prev_sheep_pos[i] - pen_center)
                                    curr_dist_to_pen = np.linalg.norm(sheep_pos - pen_center)
                                    if curr_dist_to_pen > prev_dist_to_pen:  # Moving away
                                        reward -= 0.3  # Penalty for pushing perpendicular
        
        # ========================================================================
        # 8. ADDITIONAL: Reward for overall progress toward pen
        # ========================================================================
        if prev_info is not None:
            prev_sheep_pos = prev_info.get("sheep_positions", None)
            if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                curr_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
                progress = float(np.mean(prev_dists_to_pen - curr_dists_to_pen))
                if progress > 0:  # Moving toward pen
                    reward += progress * 0.1
    
    # Small time penalty
    reward -= 0.01

    return reward


def wolf_reward_fn(info, prev_info=None) -> float:
    """
    Wolf reward function: Eat sheep, distract dog from pen.
    Mission: Push sheep away from pen, eat sheep, avoid dog.
    """
    reward = 0.0

    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    wolf_is_dead = info.get("wolf_is_dead", False)
    sheep_entered_pen = info.get("sheep_entered_pen", 0)
    
    # Get positions early for distance calculations
    wolf_pos = info.get("wolf_position", None)
    pen_center = info.get("pen_center", None)
    
    # ========================================================================
    # 1. REWARD: Eating sheep (scaled by distance to pen)
    # ========================================================================
    if sheep_eaten > 0:
        base_reward = 200.0
        # Scale reward based on distance to pen: closer to pen = smaller reward
        if wolf_pos is not None and pen_center is not None:
            wolf_to_pen_dist = float(np.linalg.norm(wolf_pos - pen_center))
            # Define thresholds: close to pen (< 300) = reduced reward, far (> 500) = full reward
            close_threshold = 300.0  # Within this distance, reward is reduced
            far_threshold = 500.0   # Beyond this distance, full reward
            
            if wolf_to_pen_dist < close_threshold:
                # Very close to pen: minimal reward (20% of base)
                distance_factor = 0.2 + (wolf_to_pen_dist / close_threshold) * 0.3  # 0.2 to 0.5
            elif wolf_to_pen_dist < far_threshold:
                # Medium distance: partial reward (50% to 100%)
                distance_factor = 0.5 + ((wolf_to_pen_dist - close_threshold) / (far_threshold - close_threshold)) * 0.5
            else:
                # Far from pen: full reward
                distance_factor = 1.0
            
            reward += float(sheep_eaten) * base_reward * distance_factor
        else:
            # Fallback if positions not available
            reward += float(sheep_eaten) * base_reward
    
    # ========================================================================
    # 2. PENALTY: Wolf killed/eaten
    # ========================================================================
    if wolf_killed:
        reward -= 150.0
    
    # ========================================================================
    # 3. PENALTY: Sheep entering pen (dog succeeding)
    # ========================================================================
    if sheep_entered_pen > 0:
        reward -= float(sheep_entered_pen) * 80.0
    
    if not wolf_is_dead:
        sheep_positions = info.get("sheep_positions", None)
        # wolf_pos and pen_center already retrieved above
        wolf_velocity = info.get("wolf_velocity", None)
        dog_pos = info.get("dog_position", None)
        
        if sheep_positions is not None and len(sheep_positions) > 0 and wolf_pos is not None:
            # ========================================================================
            # 4. PENALTY: Wolf vector close to collinear with pen vector + cluster in front
            # ========================================================================
            if wolf_velocity is not None and np.linalg.norm(wolf_velocity) > 0.1 and pen_center is not None:
                wolf_velocity_normalized = wolf_velocity / np.linalg.norm(wolf_velocity)
                wolf_to_pen = pen_center - wolf_pos
                wolf_to_pen_norm = np.linalg.norm(wolf_to_pen)
                
                if wolf_to_pen_norm > 1.0:
                    wolf_to_pen_normalized = wolf_to_pen / wolf_to_pen_norm
                    alignment_to_pen = np.dot(wolf_velocity_normalized, wolf_to_pen_normalized)
                    
                    # Check if wolf is moving toward pen (close to collinear)
                    if alignment_to_pen > 0.7:  # Close to collinear with pen
                        # Check if there's a cluster of sheep in front
                        sheep_in_front_count = 0
                        for sheep_pos in sheep_positions:
                            wolf_to_sheep = sheep_pos - wolf_pos
                            dist_to_sheep = np.linalg.norm(wolf_to_sheep)
                            if dist_to_sheep > 1.0:
                                wolf_to_sheep_normalized = wolf_to_sheep / dist_to_sheep
                                front_score = np.dot(wolf_velocity_normalized, wolf_to_sheep_normalized)
                                if front_score > 0.3 and dist_to_sheep < 300.0:  # Sheep in front
                                    sheep_in_front_count += 1
                        
                        # If cluster (2+ sheep) in front while moving toward pen, penalize
                        if sheep_in_front_count >= 2:
                            reward -= alignment_to_pen * 0.5  # Penalty for guiding sheep to pen
            
            # ========================================================================
            # 5. REWARD: Wolf vector close to opposite from pen vector + sheep in front
            # ========================================================================
            if wolf_velocity is not None and np.linalg.norm(wolf_velocity) > 0.1 and pen_center is not None:
                wolf_velocity_normalized = wolf_velocity / np.linalg.norm(wolf_velocity)
                wolf_to_pen = pen_center - wolf_pos
                wolf_to_pen_norm = np.linalg.norm(wolf_to_pen)
                
                if wolf_to_pen_norm > 1.0:
                    wolf_to_pen_normalized = wolf_to_pen / wolf_to_pen_norm
                    alignment_to_pen = np.dot(wolf_velocity_normalized, wolf_to_pen_normalized)
                    
                    # Check if wolf is moving away from pen (opposite direction)
                    if alignment_to_pen < -0.7:  # Close to opposite from pen
                        # Check if sheep are in front
                        sheep_in_front_count = 0
                        for sheep_pos in sheep_positions:
                            wolf_to_sheep = sheep_pos - wolf_pos
                            dist_to_sheep = np.linalg.norm(wolf_to_sheep)
                            if dist_to_sheep > 1.0:
                                wolf_to_sheep_normalized = wolf_to_sheep / dist_to_sheep
                                front_score = np.dot(wolf_velocity_normalized, wolf_to_sheep_normalized)
                                if front_score > 0.3 and dist_to_sheep < 300.0:  # Sheep in front
                                    sheep_in_front_count += 1
                        
                        # Reward for pushing sheep away from pen
                        if sheep_in_front_count >= 1:
                            reward += abs(alignment_to_pen) * 0.4  # Reward for opposite direction
            
            # ========================================================================
            # 6. PENALTY: Wolf going close to dog (avoid dog)
            # ========================================================================
            if dog_pos is not None:
                dog_dist = float(np.linalg.norm(wolf_pos - dog_pos))
                
                if dog_dist < 100.0:  # Very close - strong penalty
                    penalty = (1.0 - dog_dist / 100.0) * 0.6
                    reward -= penalty
                elif dog_dist < 150.0:  # Close - moderate penalty
                    penalty = (1.0 - dog_dist / 150.0) * 0.3
                    reward -= penalty
                elif dog_dist < 250.0:  # Moderate distance - small penalty
                    penalty = (1.0 - dog_dist / 250.0) * 0.1
                    reward -= penalty
                
                # Additional penalty if moving toward dog
                if wolf_velocity is not None and np.linalg.norm(wolf_velocity) > 0.1:
                    wolf_to_dog = dog_pos - wolf_pos
                    wolf_to_dog_norm = np.linalg.norm(wolf_to_dog)
                    if wolf_to_dog_norm > 1.0:
                        wolf_to_dog_normalized = wolf_to_dog / wolf_to_dog_norm
                        wolf_velocity_normalized = wolf_velocity / np.linalg.norm(wolf_velocity)
                        alignment_to_dog = np.dot(wolf_velocity_normalized, wolf_to_dog_normalized)
                        if alignment_to_dog > 0.5 and dog_dist < 200.0:  # Moving toward dog
                            reward -= alignment_to_dog * 0.4  # Additional penalty
            
            # ========================================================================
            # 7. REWARD: Pushing sheep away from pen (stronger when close to pen)
            # ========================================================================
            if pen_center is not None and wolf_pos is not None:
                wolf_to_pen_dist = float(np.linalg.norm(wolf_pos - pen_center))
                sheep_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
                avg_dist_to_pen = float(np.mean(sheep_dists_to_pen))
                
                # Base reward for sheep being far from pen
                reward += min(avg_dist_to_pen / 600.0, 1.0) * 0.2
                
                # Reward for pushing sheep further from pen (stronger when wolf is close to pen)
                if prev_info is not None:
                    prev_sheep_pos = prev_info.get("sheep_positions", None)
                    if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                        prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                        prev_avg_dist = float(np.mean(prev_dists_to_pen))
                        if avg_dist_to_pen > prev_avg_dist:  # Sheep moving away from pen
                            progress = avg_dist_to_pen - prev_avg_dist
                            
                            # Scale reward based on wolf's distance to pen
                            # Closer to pen = stronger reward for pushing away
                            close_threshold = 300.0
                            if wolf_to_pen_dist < close_threshold:
                                # Very close to pen: strong reward for pushing away (2x multiplier)
                                distance_multiplier = 2.0 - (wolf_to_pen_dist / close_threshold)  # 2.0 to 1.0
                            elif wolf_to_pen_dist < 500.0:
                                # Medium distance: moderate reward
                                distance_multiplier = 1.0 - ((wolf_to_pen_dist - close_threshold) / 200.0) * 0.5  # 1.0 to 0.5
                            else:
                                # Far from pen: smaller reward (can eat instead)
                                distance_multiplier = 0.5
                            
                            reward += progress * 0.15 * distance_multiplier
            
            # ========================================================================
            # 8. ADDITIONAL: Reward for getting close to sheep (hunting behavior)
            # ========================================================================
            sheep_dists_to_wolf = np.linalg.norm(sheep_positions - wolf_pos, axis=1)
            min_sheep_dist = float(np.min(sheep_dists_to_wolf))
            
            # Reward for being close to sheep
            reward += (1.0 - min(min_sheep_dist / 400.0, 1.0)) * 0.3
            
            # Bonus for being very close (about to eat)
            if min_sheep_dist < 50.0:
                reward += 0.4
            
            # Reward for moving toward sheep
            if prev_info is not None:
                prev_wolf_pos = prev_info.get("wolf_position", None)
                prev_sheep_pos = prev_info.get("sheep_positions", None)
                if prev_wolf_pos is not None and prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                    prev_min_dist = float(np.min(np.linalg.norm(prev_sheep_pos - prev_wolf_pos, axis=1)))
                    progress = prev_min_dist - min_sheep_dist
                    if progress > 0:  # Moving toward sheep
                        reward += progress * 0.08
    
    # Small time penalty
    reward -= 0.01

    return reward


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train(
    num_episodes=1000,
    max_steps=3600,
    save_interval=50,
    device='cpu',
    headless=False,
    render_every_n_steps=20,
    render_fps=0,
    log_interval=1,
):
    print("=" * 70)
    print("FIXED MULTI-AGENT DQN TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Total training steps: {num_episodes * max_steps}")
    print(f"Number of discrete actions: {NUM_ACTIONS}")
    print()
    print("KEY IMPROVEMENTS:")
    print(f"  ✓ Per-step epsilon decay (linear annealing over {EPSILON_DECAY_STEPS:,} steps)")
    print(f"  ✓ Soft target updates (TAU={TAU}, polyak averaging)")
    print(f"  ✓ Updates per step: {MAX_UPDATES_PER_STEP} (improved sample efficiency)")
    print(f"  ✓ GroupNorm instead of BatchNorm (better RL stability)")
    print(f"  ✓ Update frequency: Every {UPDATE_EVERY} step(s)")
    print(f"  ✓ Batch size: {BATCH_SIZE}")
    print(f"  ✓ Replay buffer: {REPLAY_CAPACITY:,} capacity")
    if not headless:
        status = "disabled" if render_every_n_steps == 0 else f"every {render_every_n_steps} steps"
        print(f"Rendering: {status} (press H to toggle)")
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
        lr=LR,
        hidden_dim=512,
        buffer_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        update_every=UPDATE_EVERY,
        max_updates_per_step=MAX_UPDATES_PER_STEP,
        target_update_every=TARGET_UPDATE_EVERY,
        learning_starts=WARMUP_STEPS,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        pen_vec_dim=2,
        num_actions=NUM_ACTIONS,
        use_torch_compile=False
    )

    if TRAIN_DOG or LOAD_DOG_CHECKPOINT:
        dog_agent = DQNAgent(agent_type="dog", **common_kwargs)
        if LOAD_DOG_CHECKPOINT:
            # Try to find latest checkpoint, fallback to hardcoded path
            checkpoint_path = find_latest_checkpoint(SAVE_DIR, "dog")
            if checkpoint_path is None:
                checkpoint_path = DOG_CHECKPOINT_PATH
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: Checkpoint not found at {checkpoint_path}, starting from scratch")
                    checkpoint_path = None
            
            if checkpoint_path:
                print(f"Loading dog checkpoint from: {checkpoint_path}")
                dog_agent.load(checkpoint_path)
                print("Dog checkpoint loaded successfully!")
    else:
        dog_agent = RandomAgent(agent_type="dog", max_value=0)

    if TRAIN_WOLF or LOAD_WOLF_CHECKPOINT:
        wolf_agent = DQNAgent(agent_type="wolf", **common_kwargs)
        if LOAD_WOLF_CHECKPOINT:
            # Try to find latest checkpoint, fallback to hardcoded path
            checkpoint_path = find_latest_checkpoint(SAVE_DIR, "wolf")
            if checkpoint_path is None:
                checkpoint_path = WOLF_CHECKPOINT_PATH
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: Checkpoint not found at {checkpoint_path}, starting from scratch")
                    checkpoint_path = None
            
            if checkpoint_path:
                print(f"Loading wolf checkpoint from: {checkpoint_path}")
                wolf_agent.load(checkpoint_path)
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
                f"Dog ε: {dog_agent.epsilon:.3f} | " if TRAIN_DOG else ""
                f"Wolf ε: {wolf_agent.epsilon:.3f} | " if TRAIN_WOLF else ""
                f"Dog Updates: {dog_agent.update_count} | " if TRAIN_DOG else ""
                f"Wolf Updates: {wolf_agent.update_count} | " if TRAIN_WOLF else ""
                f"Time: {elapsed:.1f}s"
            )

        if (episode + 1) % save_interval == 0:
            if TRAIN_DOG:
                dog_agent.save(f"{FOLDER_NAME}/dog_dqn_episode_{episode + 1}.pth")
            if TRAIN_WOLF:
                wolf_agent.save(f"{FOLDER_NAME}/wolf_dqn_episode_{episode + 1}.pth")
            print(f"Saved checkpoint at episode {episode + 1}")

    if TRAIN_DOG:
        dog_agent.save(f"{FOLDER_NAME}/dog_dqn_final.pth")
    if TRAIN_WOLF:
        wolf_agent.save(f"{FOLDER_NAME}/wolf_dqn_final.pth")
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
        headless=HEADLESS,
        render_every_n_steps=RENDER_EVERY,
        render_fps=60,
        log_interval=1,
    )