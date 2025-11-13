"""DQN training entry point for the sheep herding simulation.

Single file to edit:
1. Define your Q-network class for DQN
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

from agents.base_agent import RandomAgent
import config
from algorithms.sac.agent import BaseAgent
from actions import DogAction, WolfAction
from simulator import Simulator


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

SAVE_DIR = os.path.join('saves', 'dqn')
SAVE_EVERY_EPISODES = 20  # Save more frequently for hackathon

FOLDER_NAME = os.path.join(SAVE_DIR, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
LOAD_DOG_CHECKPOINT = False
DOG_CHECKPOINT_PATH = r"saves/dqn/best_dog.pth"

LOAD_WOLF_CHECKPOINT = False
WOLF_CHECKPOINT_PATH = r"saves/dqn/best_wolf.pth"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

EPISODES = 1000  # how many episodes to run (Longer - more training, you can checkpoint and resume)
MAX_STEPS_PER_EPISODE = 2000  # max steps per episode
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DOG = True
TRAIN_WOLF = True

# DQN hyperparameters - OPTIMIZED FOR RTX 3060 (12GB VRAM)
LR = 3e-4           # Learning rate for Q-network
GAMMA = 0.99        # Discount factor (importance of future rewards)
TAU = 0.005         # Target network soft-update rate (unused for hard updates)
BATCH_SIZE = 512    # OPTIMIZED: Large batch size to utilize RTX 3060 fully (was 96)
REPLAY_CAPACITY = 500_000  # OPTIMIZED: Larger buffer for more diverse samples (was 250k)
WARMUP_STEPS = 5_000  # OPTIMIZED: More warmup for better initial samples (was 3k)
UPDATE_EVERY = 1    # OPTIMIZED: Update every step (was 2) - we do multiple updates per trigger
MAX_UPDATES_PER_STEP = 4  # OPTIMIZED: Do 4 updates when triggered (utilizes GPU fully)
TARGET_UPDATE_EVERY = 200  # OPTIMIZED: Update target network every N updates (hard update, was 100)
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01   # Final exploration rate
EPSILON_DECAY = 0.995  # Epsilon decay per episode

# Rendering
HEADLESS = False  # Window opens but we won't render (still fast!)
RENDER_EVERY = 0  # 0 disables auto-render - THIS IS KEY FOR SPEED!

# ============================================================================
# DISCRETE ACTION SPACE
# ============================================================================

# Discretize continuous action space for DQN
# Forward speed: [0, 0.33, 0.67, 1.0] = 4 levels
# Turn rate: [-1, -0.33, 0.33, 1] = 4 levels
# Total: 16 discrete actions
FORWARD_SPEEDS = np.array([0.0, 0.33, 0.67, 1.0])
TURN_RATES = np.array([-1.0, -0.33, 0.33, 1.0])
NUM_FORWARD_SPEEDS = len(FORWARD_SPEEDS)
NUM_TURN_RATES = len(TURN_RATES)
NUM_ACTIONS = NUM_FORWARD_SPEEDS * NUM_TURN_RATES  # 16 actions


def action_to_discrete(forward_speed: float, turn_rate: float) -> int:
    """Convert continuous action to discrete action index."""
    # Find closest forward speed
    forward_idx = np.argmin(np.abs(FORWARD_SPEEDS - forward_speed))
    # Find closest turn rate
    turn_idx = np.argmin(np.abs(TURN_RATES - turn_rate))
    # Convert to single action index
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

# Encodes the observation grid (channels × H × W) and concatenates the 2-D pen vector; outputs Q-values for each action.
class CNNFeature(nn.Module):
    """CNN feature extractor for grid observations - OPTIMIZED for best performance."""
    def __init__(self, observation_shape, pen_vec_dim=2, hidden_dim=512):
        super().__init__()
        channels, height, width = observation_shape
        # Optimized CNN with better feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),  # BatchNorm for better training stability
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Stride 2 for spatial reduction
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Stride 2 for more reduction
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling for robustness
            nn.Flatten()
        )
        # Calculate output size after convs: 256 * 4 * 4 = 4096
        conv_out = 256 * 4 * 4
        self.mlp = nn.Sequential(
            nn.Linear(conv_out + pen_vec_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),  # Layer norm for stability
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
    """Deep Q-Network for discrete action space - Optimized architecture."""
    def __init__(self, observation_shape, pen_vec_dim=2, hidden_dim=512, num_actions=NUM_ACTIONS):
        super().__init__()
        self.feature = CNNFeature(observation_shape, pen_vec_dim, hidden_dim)
        # Improved Q-head with better capacity
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


# ===============================
#         Replay Buffer
# ===============================

# Stores (obs, pen_vec, action, reward, next_obs, next_pen, done) for off-policy learning
class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape, pen_vec_dim=2, action_dim=1, device='cpu'):
        self.capacity = capacity
        self.device = device

        self.obs_buf = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.pen_buf = np.zeros((capacity, pen_vec_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.int64)  # Discrete actions
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
        # OPTIMIZED: Use torch.from_numpy for faster conversion (shares memory, non-blocking transfer)
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.from_numpy(self.obs_buf[idxs]).to(self.device, non_blocking=True)
        pen = torch.from_numpy(self.pen_buf[idxs]).to(self.device, non_blocking=True)
        act = torch.from_numpy(self.act_buf[idxs]).to(self.device, non_blocking=True)
        rew = torch.from_numpy(self.rew_buf[idxs]).to(self.device, non_blocking=True)
        next_obs = torch.from_numpy(self.next_obs_buf[idxs]).to(self.device, non_blocking=True)
        next_pen = torch.from_numpy(self.next_pen_buf[idxs]).to(self.device, non_blocking=True)
        done = torch.from_numpy(self.done_buf[idxs]).to(self.device, non_blocking=True)
        return obs, pen, act, rew, next_obs, next_pen, done


# ===============================
#             DQN Agent
# ===============================

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent implementing BaseAgent interface.
    Uses epsilon-greedy exploration and target network.
    OPTIMIZED FOR RTX 3060: Large batches, multiple updates per step, torch.compile
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
        buffer_capacity=500_000,
        batch_size=512,
        update_every=1,
        max_updates_per_step=4,
        target_update_every=200,
        learning_starts=5_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        pen_vec_dim=2,
        num_actions=NUM_ACTIONS,
        use_torch_compile=True
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.max_updates_per_step = max_updates_per_step
        self.target_update_every = target_update_every
        self.learning_starts = learning_starts
        self.agent_type = agent_type
        self.action_dim = action_dim
        self.num_actions = num_actions

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_amp = (device == 'cuda')
        if self.use_amp:
            torch.backends.cudnn.benchmark = True  # Faster convs for fixed input sizes
            torch.backends.cudnn.deterministic = False
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Q-network and target network
        self.q_network = DQN(observation_shape, pen_vec_dim, hidden_dim, num_actions).to(device)
        self.target_network = DQN(observation_shape, pen_vec_dim, hidden_dim, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # OPTIMIZATION: Compile network for faster execution (PyTorch 2.0+)
        if use_torch_compile and device == 'cuda' and hasattr(torch, 'compile'):
            try:
                print(f"Compiling {agent_type} Q-network with torch.compile for faster execution...")
                self.q_network = torch.compile(self.q_network, mode='reduce-overhead')
                print(f"Successfully compiled {agent_type} Q-network!")
            except Exception as e:
                print(f"Warning: torch.compile failed for {agent_type}: {e}. Continuing without compilation.")
        
        # Set target network to eval mode (no gradients needed)
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-7)

        self.replay = ReplayBuffer(buffer_capacity, observation_shape, pen_vec_dim, action_dim=1, device=device)
        self.total_steps = 0
        self.update_count = 0
        self._last_action_np = None

    def _to_tensor_inputs(self, observation: np.ndarray, pen_vector: np.ndarray):
        obs_grid = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        pen_vec = torch.as_tensor(pen_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        return obs_grid, pen_vec

    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        # OPTIMIZED: Use inference mode for faster execution (no gradient tracking)
        self.q_network.eval()
        with torch.inference_mode():  # OPTIMIZED: inference_mode is faster than no_grad
            obs_grid, pen_vec = self._to_tensor_inputs(observation, pen_vector)
            q_values = self.q_network(obs_grid, pen_vec)
            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:
                action_idx = np.random.randint(0, self.num_actions)
            else:
                action_idx = q_values.argmax().item()
        self.q_network.train()

        # Convert discrete action to continuous action
        forward_speed, turn_rate = discrete_to_action(action_idx)

        self._last_action_np = action_idx

        if self.agent_type == "dog":
            return DogAction(forward_speed=forward_speed, turn_rate=turn_rate)
        else:
            return WolfAction(forward_speed=forward_speed, turn_rate=turn_rate)

    def observe(self, observation: np.ndarray, pen_vector: np.ndarray,
                action, reward: float, next_observation: np.ndarray,
                next_pen_vector: np.ndarray, done: bool, info: dict):
        # Convert action to discrete
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

        # OPTIMIZED: Begin learning after warmup, do multiple updates per trigger
        if self.replay.size >= self.learning_starts and (self.total_steps % self.update_every == 0):
            # Do multiple updates to utilize GPU fully
            for _ in range(self.max_updates_per_step):
                self._update()

    def _update(self):
        # OPTIMIZED: Sample batch (already on GPU from ReplayBuffer)
        obs, pen, act, rew, next_obs, next_pen, done = self.replay.sample(self.batch_size)

        # Compute target Q-values using target network (no gradients)
        with torch.amp.autocast('cuda', enabled=self.use_amp), torch.no_grad():
            next_q_values = self.target_network(next_obs, next_pen)
            next_q_max = next_q_values.max(dim=1, keepdim=True)[0]
            target_q = rew + (1.0 - done) * self.gamma * next_q_max

        # Compute current Q-values and loss
        self.optimizer.zero_grad(set_to_none=True)  # OPTIMIZED: set_to_none for faster
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            q_values = self.q_network(obs, pen)
            q_selected = q_values.gather(1, act.long())
            loss = F.mse_loss(q_selected, target_q)

        # Backward pass and optimization
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)  # Increased clip for large batches
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.optimizer.step()

        self.update_count += 1

        # Update target network (hard update every N steps)
        if self.update_count % self.target_update_every == 0:
            # OPTIMIZED: Only update target network periodically (saves computation)
            with torch.no_grad():
                for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(param.data)

    def episode_start(self):
        pass

    def episode_end(self, total_reward: float):
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

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
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
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
            # Move optimizer state tensors to the correct device
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

# ----------------------------------------------------------------------------
#                                    Guidelines
# You control the behavior with:
#   dog_reward_fn(info, prev_info)
#   wolf_reward_fn(info, prev_info)
# These run every step inside the simulator and return a single number (reward)
#
# Mission:
# - Dog: Get sheep to get in pen and can eat wolf
# - Wolf: Distract sheep from getting in pen and can eat sheep
# ----------------------------------------------------------------------------

def dog_reward_fn(info: Dict[str, any], prev_info: Dict[str, any] = None) -> float:
    """
    Dog reward function:
    Mission: Get sheep into pen, and can eat/kill wolf
    """
    reward = 0.0

    # Main objectives (sparse rewards) - MISSION CRITICAL
    sheep_entered = info.get("sheep_entered_pen", 0)
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    
    # Big rewards/penalties for main events
    reward += float(sheep_entered) * 200.0  # Major reward for saving sheep (increased)
    reward -= float(sheep_eaten) * 150.0    # Major penalty when wolf eats sheep (increased)
    reward += float(wolf_killed) * 150.0    # Big bonus for killing wolf (mission objective, increased)
    
    # Dense shaping rewards (help learning when sparse rewards are rare)
    sheep_positions = info.get("sheep_positions", None)
    dog_pos = info.get("dog_position", None)
    wolf_pos = info.get("wolf_position", None)
    pen_center = info.get("pen_center", None)
    
    if sheep_positions is not None and len(sheep_positions) > 0 and dog_pos is not None and pen_center is not None:
        # Calculate distances
        sheep_dists_to_dog = np.linalg.norm(sheep_positions - dog_pos, axis=1)
        sheep_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
        avg_sheep_dist_to_dog = float(np.mean(sheep_dists_to_dog))
        avg_sheep_dist_to_pen = float(np.mean(sheep_dists_to_pen))
        
        # Reward for herding sheep toward pen (primary mission)
        # Reward being behind sheep (pushing them toward pen)
        dog_dist_to_pen = float(np.linalg.norm(dog_pos - pen_center))
        
        # Reward for sheep making progress toward pen
        if prev_info is not None:
            prev_sheep_pos = prev_info.get("sheep_positions", None)
            if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                progress = float(np.mean(prev_dists_to_pen - sheep_dists_to_pen))
                # Stronger reward for progress toward pen
                reward += progress * 0.12  # Increased from 0.08
        
        # Reward for positioning behind sheep (herding position)
        # Dog should be further from pen than sheep average (pushing them)
        if avg_sheep_dist_to_pen < dog_dist_to_pen:
            reward += 0.2  # Bonus for herding position
        
        # Reward for keeping sheep together (herding behavior)
        sheep_spread = float(np.std(sheep_dists_to_pen))
        reward += (1.0 - min(sheep_spread / 200.0, 1.0)) * 0.1
        
        # Reward for being close enough to influence sheep (herding distance)
        optimal_herding_dist = 200.0
        if avg_sheep_dist_to_dog < optimal_herding_dist * 2:
            reward += (1.0 - min(avg_sheep_dist_to_dog / (optimal_herding_dist * 2), 1.0)) * 0.15
        
        # Protective behavior: Position between wolf and sheep (can lead to killing wolf)
        if wolf_pos is not None and not info.get("wolf_is_dead", False):
            wolf_to_dog = float(np.linalg.norm(dog_pos - wolf_pos))
            wolf_to_sheep_avg = float(np.mean(np.linalg.norm(sheep_positions - wolf_pos, axis=1)))
            
            # Reward for being between wolf and sheep (protective + hunting position)
            if wolf_to_dog < wolf_to_sheep_avg:
                reward += 0.25  # Increased reward for protective positioning
            
            # Bonus for being close to wolf (can kill it)
            if wolf_to_dog < 150.0:  # Within kill range
                reward += 0.3  # Strong incentive to engage wolf
            elif wolf_to_dog < 250.0:  # Approaching kill range
                reward += 0.15
    
    # Reward for getting sheep into pen (completion bonus)
    total_sheep_in_pen = info.get("total_sheep_in_pen", 0)
    if total_sheep_in_pen > 0:
        reward += total_sheep_in_pen * 0.5  # Small bonus for each saved sheep
    
    # Small time penalty (encourages faster completion)
    reward -= 0.01

    return reward


def wolf_reward_fn(info, prev_info=None) -> float:
    """
    Wolf reward function:
    Mission: Distract sheep from getting in pen, and can eat sheep
    """
    reward = 0.0

    # Main objectives (sparse rewards) - MISSION CRITICAL
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    wolf_is_dead = info.get("wolf_is_dead", False)
    sheep_entered_pen = info.get("sheep_entered_pen", 0)
    
    # Big rewards/penalties for main events
    reward += float(sheep_eaten) * 200.0      # Major reward for eating sheep (mission objective, increased)
    reward -= float(wolf_killed) * 150.0      # Major penalty for dying (increased)
    reward -= float(sheep_entered_pen) * 80.0  # Penalty when dog succeeds (mission failure, increased)
    
    # Dense shaping rewards (only when wolf is alive)
    if not wolf_is_dead:
        sheep_positions = info.get("sheep_positions", None)
        wolf_pos = info.get("wolf_position", None)
        dog_pos = info.get("dog_position", None)
        pen_center = info.get("pen_center", None)
        
        if sheep_positions is not None and len(sheep_positions) > 0 and wolf_pos is not None:
            # Calculate distances
            sheep_dists_to_wolf = np.linalg.norm(sheep_positions - wolf_pos, axis=1)
            min_sheep_dist = float(np.min(sheep_dists_to_wolf))
            avg_sheep_dist = float(np.mean(sheep_dists_to_wolf))
            
            # Primary mission: Get close to sheep (can eat them)
            # Strong shaping: reward for getting close to nearest sheep
            reward += (1.0 - min(min_sheep_dist / 400.0, 1.0)) * 0.5  # Increased from 0.4
            
            # Bonus for being very close to sheep (about to eat)
            if min_sheep_dist < 50.0:
                reward += 0.5  # Increased from 0.3
            
            # Reward for moving toward sheep (progress-based)
            if prev_info is not None:
                prev_wolf_pos = prev_info.get("wolf_position", None)
                prev_sheep_pos = prev_info.get("sheep_positions", None)
                if prev_wolf_pos is not None and prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                    prev_min_dist = float(np.min(np.linalg.norm(prev_sheep_pos - prev_wolf_pos, axis=1)))
                    progress = prev_min_dist - min_sheep_dist
                    reward += progress * 0.08  # Increased from 0.05
            
            # Secondary mission: Distract sheep from pen
            if pen_center is not None:
                sheep_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
                avg_dist_to_pen = float(np.mean(sheep_dists_to_pen))
                
                # Reward if sheep are far from pen (distracted - mission success)
                reward += min(avg_dist_to_pen / 600.0, 1.0) * 0.2  # Increased from 0.1
                
                # Reward for pushing sheep away from pen (progress-based)
                if prev_info is not None:
                    prev_sheep_pos = prev_info.get("sheep_positions", None)
                    if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                        prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                        prev_avg_dist = float(np.mean(prev_dists_to_pen))
                        # Reward if sheep are moving away from pen
                        if avg_dist_to_pen > prev_avg_dist:
                            reward += (avg_dist_to_pen - prev_avg_dist) * 0.1
            
            # Strategic: Avoid dog when it's close (survival behavior)
            if dog_pos is not None:
                dog_dist = float(np.linalg.norm(wolf_pos - dog_pos))
                
                # Strong penalty for being too close to dog (danger zone)
                if dog_dist < 100.0:  # Dog can kill wolf at this distance
                    # Large penalty for being in kill range
                    reward -= (1.0 - dog_dist / 100.0) * 0.5  # Increased from 0.25
                elif dog_dist < 150.0:  # Moderate danger zone
                    reward -= (1.0 - dog_dist / 150.0) * 0.2  # Increased from 0.1
                elif dog_dist < 250.0:  # Caution zone
                    reward -= (1.0 - dog_dist / 250.0) * 0.05
            
            # Reward for strategic positioning: between dog and sheep
            # This allows wolf to intercept sheep while avoiding dog
            if dog_pos is not None and pen_center is not None:
                wolf_to_dog = float(np.linalg.norm(wolf_pos - dog_pos))
                wolf_to_sheep_avg = avg_sheep_dist
                
                # Reward for being closer to sheep than dog is (intercept position)
                dog_to_sheep_avg = float(np.mean(np.linalg.norm(sheep_positions - dog_pos, axis=1)))
                if wolf_to_sheep_avg < dog_to_sheep_avg:
                    reward += 0.15  # Bonus for better positioning
    
    # Small time penalty to encourage action
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
    print("STARTING MULTI-AGENT DQN TRAINING (OPTIMIZED FOR RTX 3060)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Number of discrete actions: {NUM_ACTIONS}")
    print(f"Action space: {NUM_FORWARD_SPEEDS} forward speeds × {NUM_TURN_RATES} turn rates")
    print()
    print("OPTIMIZATIONS ENABLED:")
    print(f"  - Batch size: {BATCH_SIZE} (was 96) - Utilizes GPU fully")
    print(f"  - Updates per step: {MAX_UPDATES_PER_STEP} - Multiple updates per trigger")
    print(f"  - Replay buffer: {REPLAY_CAPACITY:,} capacity - More diverse samples")
    print(f"  - Update frequency: Every {UPDATE_EVERY} step(s)")
    print(f"  - Mixed precision: Enabled (AMP)")
    if device == 'cuda' and hasattr(torch, 'compile'):
        print(f"  - torch.compile: Enabled (PyTorch 2.0+)")
    print(f"  - cuDNN benchmark: Enabled")
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
        lr=LR,
        hidden_dim=512,  # Optimized for best performance (GPU can handle it)
        buffer_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        update_every=UPDATE_EVERY,
        max_updates_per_step=MAX_UPDATES_PER_STEP,  # OPTIMIZED: Multiple updates per trigger
        target_update_every=TARGET_UPDATE_EVERY,
        learning_starts=WARMUP_STEPS,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        pen_vec_dim=2,
        num_actions=NUM_ACTIONS,
        use_torch_compile=True  # OPTIMIZED: Enable torch.compile for faster execution
    )

    if TRAIN_DOG or LOAD_DOG_CHECKPOINT:
        dog_agent = DQNAgent(agent_type="dog", **common_kwargs)
        if LOAD_DOG_CHECKPOINT:
            print(f"Loading dog checkpoint from: {DOG_CHECKPOINT_PATH}")
            dog_agent.load(DOG_CHECKPOINT_PATH)
            print("Dog checkpoint loaded successfully!")
    else:
        dog_agent = RandomAgent(agent_type="dog", max_value=0)

    if TRAIN_WOLF or LOAD_WOLF_CHECKPOINT:
        wolf_agent = DQNAgent(agent_type="wolf", **common_kwargs)
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
                f"Dog Updates: {dog_agent.update_count} | " if TRAIN_DOG else ""
                f"Wolf Updates: {wolf_agent.update_count} | " if TRAIN_WOLF else ""
                f"ε(dog): {dog_agent.epsilon:.3f} | " if TRAIN_DOG else ""
                f"ε(wolf): {wolf_agent.epsilon:.3f} | " if TRAIN_WOLF else ""
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
        headless=HEADLESS,          # Uses config from top (True = faster)
        render_every_n_steps=RENDER_EVERY,  # Uses config from top (0 = no render)
        render_fps=60,
        log_interval=1,
    )