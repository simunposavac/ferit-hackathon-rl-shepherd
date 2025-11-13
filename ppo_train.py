import os
# os.environ.pop("SDL_VIDEODRIVER", None)  # ensure windowed video driver
# -----------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pygame
from simulator import Simulator
from agents.base_agent import BaseAgent
from actions import DogAction, WolfAction
import config
from typing import Optional, Dict, Any, List
import torch.nn.functional as F
from torch.distributions import Beta
from datetime import datetime
from pathlib import Path
from agents.base_agent import BaseAgent
from actions import DogAction, WolfAction

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# TODO: consider implementing RNN as it gives better results
USE_RNN = True # use RNN networks (policy has memory, usually better decisions, slower)

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

LOAD_DOG_CHECKPOINT = False
DOG_CHECKPOINT_PATH = r"saves\ppo_rnn\run_2025-11-12_01-21-14\dog_episode_1000.pth"

LOAD_WOLF_CHECKPOINT = False
WOLF_CHECKPOINT_PATH = r"saves\ppo_rnn\run_2025-11-12_01-21-14\wolf_episode_1000.pth"

# ============================================================================
# NETWORKS
# ============================================================================

# This is the actor network that decides what action to take given the current observation
# Its "stateless" - no temporal memory - so each frame is treated independently
class BetaActorCNN(nn.Module):
    def __init__(self, observation_shape, action_dim, pen_vec_dim=2, hidden_dim=256):
        super().__init__()
        C, H, W = observation_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),              
            nn.Flatten()
        )
        conv_out = 64 * 4 * 4
        self.feature = nn.Sequential(
            nn.Linear(conv_out + pen_vec_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head  = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_grid, pen_vec):
        z = self.conv(obs_grid)
        z = torch.cat([z, pen_vec], dim=-1)
        z = self.feature(z)
        alpha = F.softplus(self.alpha_head(z)) + 1.0
        beta  = F.softplus(self.beta_head(z))  + 1.0
        return alpha, beta
    
    def get_dist(self, obs_grid, pen_vec):
        a, b = self.forward(obs_grid, pen_vec)
        return Beta(a, b)

    @torch.no_grad()
    def deterministic_act(self, obs_grid, pen_vec):
        a, b = self.forward(obs_grid, pen_vec)
        # mode = (alpha-1)/(alpha+beta-2), clipped for numeric safety
        mode = (a - 1.0) / (a + b - 2.0)
        return torch.clamp(mode, 1e-6, 1.0 - 1e-6)

# Estimates the state value V(s) the expected return from the current observation
class CriticCNN(nn.Module):
    def __init__(self, observation_shape, pen_vec_dim=2, hidden_dim=256):
        super().__init__()
        C, H, W = observation_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        conv_out = 64 * 4 * 4

        self.mlp = nn.Sequential(
            nn.Linear(conv_out + pen_vec_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_grid, pen_vec):
        """
        obs_grid: torch.Tensor (B,C,H,W)
        pen_vec:  torch.Tensor (B,2)
        returns:  torch.Tensor (B,)
        """
        z = self.conv(obs_grid)
        z = torch.cat([z, pen_vec], dim=-1)
        v = self.mlp(z).squeeze(-1)
        return v

# ============================================================================
# PPO Agents
# ============================================================================

# This class ties the actor and critic together and runs PPO updates
# Stores rollouts: (obs, action, reward, done, log_prob, value)
# Compute returns and advantages (using GAE, Generalized Advantage Estimation)
class PPOAgentCNN(BaseAgent):
    def __init__(self,
                 observation_shape,
                 action_dim,
                 agent_type="dog",
                 lr=3e-4,                     # [1e-4 - 5e-4] learning rate 
                 gamma=0.99,                  # discount factor (how much we care about future rewards)
                 eps_clip=0.2,                # PPO clipping range. Smaller = safer updates, slower learning
                 k_epochs=10,                 # [2 - 10] number of passes over the collected batch each update
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 a_optim_batch_size=2048,     # minibatch sizes for actor
                 c_optim_batch_size=2048,     # minibatch sizes for critic
                 entropy_coef=0.01,           # [0.001 - 0.02] higher - ecourages exploration (more random), lower - more conservative (policy can collapse to a single action)
                 entropy_coef_decay=0.9999,   # slowly reduce exploration over time (1.0 - no decay)
                 l2_reg=1e-4,
                 hidden_dim=256,              # [64–256]
                 amp=True                     # mixed precision acceleration on GPU (fast)
        ):
        self.device = device
        self.gamma = gamma
        self.lambd = 0.95
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.agent_type = agent_type
        self.a_bs = a_optim_batch_size
        self.c_bs = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_coef_decay
        self.l2_reg = l2_reg
        self.amp = bool(amp and (device == "cuda"))

        # Models
        self.actor = BetaActorCNN(observation_shape, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic = CriticCNN(observation_shape, hidden_dim=hidden_dim).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=l2_reg)

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.buffer = {
            'observations': [],
            'pen_vectors': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

    def episode_start(self):
        pass

    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        obs_grid = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        pen_vec  = torch.as_tensor(pen_vector,  dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.actor.get_dist(obs_grid, pen_vec)
            a01 = torch.clamp(dist.sample(), 1e-6, 1.0 - 1e-6)
            logprob = dist.log_prob(a01).cpu().numpy().flatten()   # per-dim
            value = self.critic(obs_grid, pen_vec).cpu().item()

        a01_np = a01.cpu().numpy()[0]
        self._last_log_prob_vec = logprob
        self._last_value = value

        speed = float(a01_np[0])                  # [0,1]
        turn  = float(2.0 * (a01_np[1] - 0.5))    # [-1,1]

        if self.agent_type == "dog":
            return DogAction(forward_speed=speed, turn_rate=turn)
        else:
            return WolfAction(forward_speed=speed, turn_rate=turn)

    def observe(self, observation: np.ndarray, pen_vector: np.ndarray,
                action, reward: float, next_observation: np.ndarray,
                next_pen_vector: np.ndarray, done: bool, info: dict):
        a01 = np.array([action.forward_speed, (action.turn_rate + 1.0) / 2.0], dtype=np.float32)
        a01 = np.clip(a01, 1e-6, 1.0 - 1e-6)

        self.buffer['observations'].append(observation)
        self.buffer['pen_vectors'].append(pen_vector)
        self.buffer['actions'].append(a01)
        self.buffer['log_probs'].append(self._last_log_prob_vec.copy())
        self.buffer['rewards'].append(float(reward))
        self.buffer['dones'].append(bool(done))
        self.buffer['values'].append(float(self._last_value))

    def episode_end(self, total_reward: float):
        if len(self.buffer['rewards']) == 0:
            return

        self.entropy_coef *= self.entropy_decay

        observations = torch.as_tensor(np.array(self.buffer['observations']), dtype=torch.float32, device=self.device)
        pen_vectors  = torch.as_tensor(np.array(self.buffer['pen_vectors']), dtype=torch.float32, device=self.device)
        actions01    = torch.as_tensor(np.array(self.buffer['actions']),     dtype=torch.float32, device=self.device)
        old_logp_vec = torch.as_tensor(np.array(self.buffer['log_probs']),   dtype=torch.float32, device=self.device)

        rewards = np.array(self.buffer['rewards'], dtype=np.float32)
        dones   = np.array(self.buffer['dones'],   dtype=np.bool_)
        values  = np.array(self.buffer['values'],  dtype=np.float32)

        T = len(rewards)
        adv_np = np.zeros((T,), dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            r_t = float(rewards[t])
            v_t = float(values[t])
            done_t = float(dones[t])
            delta = r_t + self.gamma * next_value * (1.0 - done_t) - v_t
            gae   = delta + self.gamma * 0.95 * (1.0 - done_t) * gae
            adv_np[t] = gae
            next_value = v_t

        adv = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = adv + torch.as_tensor(values, dtype=torch.float32, device=self.device)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        # Minibatch shuffling
        N = observations.shape[0]
        idx_all = torch.randperm(N, device=self.device)

        a_iters = int(np.ceil(N / self.a_bs))
        c_iters = int(np.ceil(N / self.c_bs))

        for _ in range(self.k_epochs):
            # ---------- Actor ----------
            for i in range(a_iters):
                sl = slice(i*self.a_bs, min((i+1)*self.a_bs, N))
                b = idx_all[sl]
                ob, pv, ac, ol, _, ad = (
                    observations[b], pen_vectors[b], actions01[b], old_logp_vec[b], returns[b], adv[b]
                )

                with torch.amp.autocast("cuda", enabled=self.amp):
                    dist = self.actor.get_dist(ob, pv)
                    ent  = dist.entropy().sum(1, keepdim=True)
                    logp = dist.log_prob(ac)
                    ratio = torch.exp(logp.sum(1, keepdim=True) - ol.sum(1, keepdim=True))
                    surr1 = ratio * ad.unsqueeze(1)
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * ad.unsqueeze(1)
                    a_loss = -torch.min(surr1, surr2) - self.entropy_coef * ent
                    a_loss = a_loss.mean()

                self.actor_opt.zero_grad(set_to_none=True)
                self.scaler.scale(a_loss).backward()
                # unscale before clipping
                self.scaler.unscale_(self.actor_opt)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40.0)
                self.scaler.step(self.actor_opt)
                self.scaler.update()

            # ---------- Critic ----------
            for i in range(c_iters):
                sl = slice(i*self.c_bs, min((i+1)*self.c_bs, N))
                b = idx_all[sl]
                ob, pv, rt = observations[b], pen_vectors[b], returns[b]

                with torch.amp.autocast("cuda", enabled=self.amp):
                    v_pred = self.critic(ob, pv)
                    c_loss = (v_pred - rt).pow(2).mean()

                self.critic_opt.zero_grad(set_to_none=True)
                self.scaler.scale(c_loss).backward()
                self.scaler.unscale_(self.critic_opt)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40.0)
                self.scaler.step(self.critic_opt)
                self.scaler.update()

        # clear buffer
        self.buffer = {k: [] for k in self.buffer.keys()}

    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str):
        chk = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(chk['actor_state_dict'])
        self.critic.load_state_dict(chk['critic_state_dict'])
        self.actor_opt.load_state_dict(chk['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(chk['critic_opt_state_dict'])

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

# from info["sheep_ids"]
def _get_sheep_with_id(id, ids, arr):
    elem = None
    if isinstance(ids, np.ndarray):
        if id in ids:
            idx = np.argmax(ids == id)
            if arr.size > idx:
                elem = arr[idx]
    else:
        if id in ids:
            idx = ids.index(id)
            if len(arr) > idx:
                elem = arr[idx]
    return elem

def dog_reward_fn(info: Dict[str, Any], prev_info: Optional[Dict[str, Any]] = None) -> float:
    reward = 0.0
    
    # Main objectives (sparse rewards)
    sheep_entered = info.get("sheep_entered_pen", 0)
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    
    # Big rewards/penalties for main events
    reward += float(sheep_entered) * 150.0  # Major reward for saving sheep
    reward -= float(sheep_eaten) * 120.0     # Major penalty when wolf eats sheep
    reward += float(wolf_killed) * 80.0      # Bonus for killing wolf
    
    # Dense shaping rewards (help learning when sparse rewards are rare)
    sheep_positions = info.get("sheep_positions", None)
    dog_pos = info.get("dog_position", None)
    wolf_pos = info.get("wolf_position", None)
    pen_center = info.get("pen_center", None)
    
    if sheep_positions is not None and len(sheep_positions) > 0 and dog_pos is not None and pen_center is not None:
        # Reward for being near sheep (herding behavior)
        sheep_dists = np.linalg.norm(sheep_positions - dog_pos, axis=1)
        avg_sheep_dist = float(np.mean(sheep_dists))
        min_sheep_dist = float(np.min(sheep_dists))
        
        # Small reward for staying close to sheep cluster
        reward += (1.0 - min(avg_sheep_dist / 400.0, 1.0)) * 0.15
        
        # Reward for sheep moving toward pen (progress-based shaping)
        if prev_info is not None:
            prev_sheep_pos = prev_info.get("sheep_positions", None)
            if prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                prev_dists_to_pen = np.linalg.norm(prev_sheep_pos - pen_center, axis=1)
                curr_dists_to_pen = np.linalg.norm(sheep_positions - pen_center, axis=1)
                progress = float(np.mean(prev_dists_to_pen - curr_dists_to_pen))
                reward += progress * 0.08  # Reward progress toward pen
        
        # Bonus for positioning between wolf and sheep (protective behavior)
        if wolf_pos is not None and not info.get("wolf_is_dead", False):
            wolf_to_dog = np.linalg.norm(dog_pos - wolf_pos)
            # Average distance from wolf to sheep
            wolf_to_sheep = np.mean(np.linalg.norm(sheep_positions - wolf_pos, axis=1))
            # Reward if dog is between wolf and sheep
            if wolf_to_dog < wolf_to_sheep:
                reward += 0.1
    
    # Small penalty for time (encourages faster completion)
    reward -= 0.01
    
    return reward

def wolf_reward_fn(info: Dict[str, Any], prev_info: Optional[Dict[str, Any]] = None) -> float:
    reward = 0.0

    # Main objectives (sparse rewards)
    sheep_eaten = info.get("sheep_eaten", 0)
    wolf_killed = info.get("wolf_killed", False)
    wolf_is_dead = info.get("wolf_is_dead", False)
    sheep_entered_pen = info.get("sheep_entered_pen", 0)
    
    # Big rewards/penalties for main events
    reward += float(sheep_eaten) * 150.0      # Major reward for eating sheep
    reward -= float(wolf_killed) * 100.0      # Major penalty for dying
    reward -= float(sheep_entered_pen) * 50.0 # Penalty when dog succeeds
    
    # Dense shaping rewards (only when wolf is alive)
    if not wolf_is_dead:
        sheep_positions = info.get("sheep_positions", None)
        wolf_pos = info.get("wolf_position", None)
        dog_pos = info.get("dog_position", None)
        
        if sheep_positions is not None and len(sheep_positions) > 0 and wolf_pos is not None:
            # Strong shaping: reward for getting close to nearest sheep
            sheep_dists = np.linalg.norm(sheep_positions - wolf_pos, axis=1)
            min_dist = float(np.min(sheep_dists))
            # Normalized distance reward (closer = better)
            reward += (1.0 - min(min_dist / 500.0, 1.0)) * 0.4
            
            # Bonus for being very close to sheep (about to eat)
            if min_dist < 50.0:
                reward += 0.3
            
            # Reward for moving toward sheep (progress-based)
            if prev_info is not None:
                prev_wolf_pos = prev_info.get("wolf_position", None)
                prev_sheep_pos = prev_info.get("sheep_positions", None)
                if prev_wolf_pos is not None and prev_sheep_pos is not None and len(prev_sheep_pos) == len(sheep_positions):
                    prev_min_dist = float(np.min(np.linalg.norm(prev_sheep_pos - prev_wolf_pos, axis=1)))
                    progress = prev_min_dist - min_dist
                    reward += progress * 0.05  # Reward progress toward sheep
            
            # Strategic: avoid dog when it's close (survival behavior)
            if dog_pos is not None:
                dog_dist = np.linalg.norm(wolf_pos - dog_pos)
                if dog_dist < 100.0:  # Dog is dangerous when close
                    # Penalty for being too close to dog
                    reward -= (1.0 - dog_dist / 100.0) * 0.25
                elif dog_dist < 150.0:  # Moderate danger zone
                    reward -= (1.0 - dog_dist / 150.0) * 0.1
    
    # Small time penalty to encourage action
    reward -= 0.01

    return reward

# ============================================================================
# TRAIN
# ============================================================================

def train(num_episodes=1000,      # how many episodes to run (Longer - more training, you can checkpoint and resume)
          max_steps=3600,         # max steps per episode
          device='cuda' if torch.cuda.is_available() else 'cpu',
          headless=False,         # hides the window for faster training
          render_every=1,         # draw environment image every N steps
          render_fps=60,          # cap rendering FPS, recommended <= 60 for stability
          save_checkpoints=True,  # whether to save model weights periodically
          save_every=10,          # save every N episodes
          checkpoint_save_dir="",
          train_dog_every=1,      # train dog every N episodes (0 - dont train, 1 - train every episode, k - train every k-th episode)
          train_wolf_every=1,     # train dog every N episodes (0 - dont train, 1 - train every episode, k - train every k-th episode)
          alternate_agents=False  # alternate between agents (first train dog <train_dog_every> times and then train wolf every <train_wolf_every> times)
    ):
    
    print("=" * 70)
    print(f"STARTING PPO {'RNN' if USE_RNN else 'CNN'} TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes} | Max steps: {max_steps}")
    print(f"AMP: {'on' if torch.cuda.is_available() else 'off'} | cuDNN benchmark: on")
    if not headless:
        print(f"Rendering: Every {render_every} steps (press H to toggle)")
        print(f"Controls: ESC to quit, H to toggle rendering, D to toggle debug visualizations")
    print()

    if train_dog_every == 0 or train_wolf_every == 0:
        alternate_agents = False

    observation_shape = (config.OBSERVATION_CHANNELS,
                         config.OBSERVATION_GRID_SIZE,
                         config.OBSERVATION_GRID_SIZE)
    action_dim = 2

    print(f"Observation shape: {observation_shape}")
    print(f"Observation range: {config.OBSERVATION_RANGE} pixels")
    print()

    
    dog_agent = PPOAgentCNN(observation_shape, action_dim, agent_type="dog", lr=3e-4, device=device)

    if LOAD_DOG_CHECKPOINT:
        try:
            dog_agent.load(DOG_CHECKPOINT_PATH)
            print(f"Loaded dog checkpoint from: {DOG_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Failed to load dog checkpoint: {e}")

    
    wolf_agent = PPOAgentCNN(observation_shape, action_dim, agent_type="wolf", lr=3e-4, device=device)

    if LOAD_WOLF_CHECKPOINT:
        try:
            wolf_agent.load(WOLF_CHECKPOINT_PATH)
            print(f"Loaded wolf checkpoint from: {WOLF_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Failed to load wolf checkpoint: {e}")

    simulator = Simulator(headless=headless, dog_reward_fn=dog_reward_fn, wolf_reward_fn=wolf_reward_fn)

    start_time = time.time()

    for episode in range(num_episodes):

        if train_dog_every == 0 and train_wolf_every == 0:
            train_dog = train_wolf = False
        elif not alternate_agents:
            train_dog  = (train_dog_every  > 0) and ((episode + 1) % train_dog_every  == 0)
            train_wolf = (train_wolf_every > 0) and ((episode + 1) % train_wolf_every == 0)
        else:
            d = max(1, train_dog_every)
            w = max(1, train_wolf_every)
            cycle = d + w
            pos = episode % cycle
            train_dog  = (train_dog_every  > 0) and (pos < d)
            train_wolf = (train_wolf_every > 0) and (not train_dog)

        if train_dog:
            print("Training dog...")
        if train_wolf:
            print("Training wolf...")
        if not train_dog and not train_wolf:
            print("Running without training...")

        (dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec) = simulator.reset()

        dog_agent.episode_start()
        wolf_agent.episode_start()

        prev_info = None

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
                            render_every = 0 if render_every > 0 else 5
                            status = "disabled" if render_every == 0 else "enabled"
                            print(f"\nRendering {status} (press H to toggle)")
                        elif event.key == pygame.K_d:
                            config.DEBUG_SHOW_OBSERVATIONS = not config.DEBUG_SHOW_OBSERVATIONS
                            config.DEBUG_SHOW_PEN_DIRECTION = not config.DEBUG_SHOW_PEN_DIRECTION
                            print(f"\nDEBUG_VISUALIZATION {config.DEBUG_SHOW_OBSERVATIONS} (press D to toggle)")

            dog_action = dog_agent.act(dog_obs, dog_pen_vec)
            wolf_action = wolf_agent.act(wolf_obs, wolf_pen_vec)

            next_dog_obs, next_dog_pen_vec, dog_reward, \
            next_wolf_obs, next_wolf_pen_vec, wolf_reward, info = simulator.step(dog_action=dog_action, wolf_action=wolf_action, episode=episode)

            if not simulator.headless and render_every > 0 and step % render_every == 0:
                simulator.render(fps=render_fps)

            done = info['done']

            # observe
            dog_agent.observe(dog_obs, dog_pen_vec, dog_action, dog_reward, next_dog_obs, next_dog_pen_vec, done, info)
            if prev_info is not None:
                if (prev_info.get("wolf_is_dead", False) and info.get("wolf_is_dead", False)) or \
                   (prev_info.get("wolf_in_cooldown", False) and info.get("wolf_in_cooldown", False)):
                    pass
                else:
                    wolf_agent.observe(wolf_obs, wolf_pen_vec, wolf_action, wolf_reward, next_wolf_obs, next_wolf_pen_vec, done, info)

            dog_episode_reward += dog_reward
            wolf_episode_reward += wolf_reward

            dog_obs = next_dog_obs
            dog_pen_vec = next_dog_pen_vec
            wolf_obs = next_wolf_obs
            wolf_pen_vec = next_wolf_pen_vec
            prev_info = info

            if done:
                break

        # PPO update at episode end
        if train_dog:
            dog_agent.episode_end(dog_episode_reward)
        if train_wolf:
            wolf_agent.episode_end(wolf_episode_reward)

        elapsed = time.time() - start_time

        print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Dog Reward: {dog_episode_reward:.2f} | "
                  f"Wolf Reward: {wolf_episode_reward:.2f} | "
                  f"Sheep Saved: {simulator.env.sheep_in_pen:.1f} | "
                  f"Steps: {step + 1} | "
                  f"Time: {elapsed:.1f}s")
        
        if save_checkpoints:
            if (episode + 1) % save_every == 0:
                if checkpoint_save_dir != "":
                    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
                    if train_dog:
                        dog_agent.save(str(checkpoint_save_dir / f"dog_episode_{episode + 1}.pth"))
                        print(f"Saved dog checkpoint at episode {episode + 1}")
                    if train_wolf:
                        wolf_agent.save(str(checkpoint_save_dir / f"wolf_episode_{episode + 1}.pth"))
                        print(f"Saved wolf checkpoint at episode {episode + 1}")
                
    print("\nTraining complete!")

    simulator.close()



def get_checkpoint_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_save_dir = Path(f"saves/{'ppo_rnn' if USE_RNN else 'ppo_cnn'}/run_{timestamp}")
    return checkpoint_save_dir

if __name__ == "__main__":
    train(
        num_episodes=1000,
        max_steps=3600,
        headless=False,          
        render_every=1,  
        render_fps=60,
        save_checkpoints=True,
        save_every=50,
        checkpoint_save_dir=get_checkpoint_dir(),
        train_dog_every=1,
        train_wolf_every=1,
        alternate_agents=False
    )
