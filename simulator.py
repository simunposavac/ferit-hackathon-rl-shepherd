"""
Simulator class that manages the entire sheep herding simulation.
Provides an easy-to-use API for controlling agents and getting observations/rewards.
"""

import pygame
import numpy as np
import os
from typing import Callable, Optional, Dict, Any, Tuple

from sim.environment import Environment
from sim.sheep import Sheep
from sim.dog import Dog
from sim.wolf import Wolf
from actions import DogAction, WolfAction
from utils.map_utils import create_green_background_image
from utils.observation_grid import (
    create_observation_grid, 
    create_global_observation_grid, 
    generate_wall_positions, 
    render_observation_grid,
    render_global_observation_grid
)
import config


class SimulationState:
    """Container for simulation state returned to agents."""
    
    def __init__(self, 
                 dog_position: np.ndarray,
                 dog_velocity: np.ndarray,
                 wolf_position: np.ndarray,
                 wolf_velocity: np.ndarray,
                 sheep_positions: np.ndarray,
                 sheep_velocities: np.ndarray,
                 sheep_ids: np.ndarray,
                 sheep_count: int,
                 sheep_in_pen: int,
                 pen_entrance_position: Tuple[float, float, float, float]):
        self.dog_position = dog_position
        self.dog_velocity = dog_velocity
        self.wolf_position = wolf_position
        self.wolf_velocity = wolf_velocity
        self.sheep_positions = sheep_positions
        self.sheep_velocities = sheep_velocities
        self.sheep_ids = sheep_ids
        self.sheep_count = sheep_count
        self.sheep_in_pen = sheep_in_pen
        self.pen_entrance_position = pen_entrance_position  # (x, top_y, bottom_y, width)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'dog_position': self.dog_position,
            'dog_velocity': self.dog_velocity,
            'wolf_position': self.wolf_position,
            'wolf_velocity': self.wolf_velocity,
            'sheep_positions': self.sheep_positions,
            'sheep_velocities': self.sheep_velocities,
            'sheep_ids': self.sheep_ids,
            'sheep_count': self.sheep_count,
            'sheep_in_pen': self.sheep_in_pen,
            'pen_entrance_position': self.pen_entrance_position
        }

def make_tiled_surface(tile, size, density=1.0, pixel_art=False):
    """
    Fill a new surface of `size` with `tile` scaled by `density`.
    Higher density = smaller tiles = more repeats.
    Lower density  = larger tiles  = fewer repeats.
    """
    W, H = size
    tw, th = tile.get_size()

    # Invert density so higher density means more tiles
    scale = 1.0 / max(0.0001, density)
    new_w = max(1, int(tw * scale))
    new_h = max(1, int(th * scale))

    if pixel_art:
        scaled_tile = pygame.transform.scale(tile, (new_w, new_h))
    else:
        scaled_tile = pygame.transform.smoothscale(tile, (new_w, new_h))

    # Match alpha mode
    has_alpha = scaled_tile.get_masks()[3] != 0 or scaled_tile.get_alpha() is not None
    surface = pygame.Surface(size, pygame.SRCALPHA if has_alpha else 0)
    surface = surface.convert_alpha() if has_alpha else surface.convert()

    for y in range(0, H, new_h):
        for x in range(0, W, new_w):
            surface.blit(scaled_tile, (x, y))
    return surface

class Simulator:
    """
    Main simulator class for the sheep herding environment.
    """
    
    def __init__(self,
                 dog_reward_fn: Optional[Callable] = None,
                 wolf_reward_fn: Optional[Callable] = None,
                 headless: bool = False):
        """
        Initialize the simulator.
        
        Args:
            dog_reward_fn: Callable that computes dog reward. 
                          Signature: fn(info: dict, prev_info: dict or None) -> float
            wolf_reward_fn: Callable that computes wolf reward.
                           Signature: fn(info: dict, prev_info: dict or None) -> float
            headless: If True, don't initialize pygame display (for training)
        """
        # Use config values
        self.width = config.SCREEN_WIDTH
        self.height = config.SCREEN_HEIGHT
        self.num_sheep = config.NUM_SHEEP
        self.headless = headless
        
        # Initialize pygame
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Sheep Herding Simulation")
            self.clock = pygame.time.Clock()
        else:
            # Initialize pygame without display for headless mode
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            self.screen = None
            self.clock = None
        
        # Create background
        background_pil = create_green_background_image(
            self.height, 
            self.width, 
            texture_strength=config.BACKGROUND_TEXTURE_STRENGTH
        )
        background_mode = background_pil.mode
        background_size = background_pil.size
        background_data = background_pil.tobytes()
        #self.background_surface = pygame.image.fromstring(background_data, background_size, background_mode)
        W, H = background_size
        tile = pygame.image.load("images/grass2.jpg").convert()  
        self.background_surface = make_tiled_surface(tile, (W, H), density=4.0)
        
        # Initialize environment
        self.env = Environment(self.width, self.height, self.background_surface)
        
        # Sprite paths
        self.sheep_sprite = os.path.join("images", "sheep.png")
        self.dog_sprite = os.path.join("images", "dog.png")
        self.wolf_sprite = os.path.join("images", "wolf.png")
        
        # Spawn entities
        self._spawn_entities()
        
        # Reward functions
        self.dog_reward_fn = dog_reward_fn or self._default_dog_reward
        self.wolf_reward_fn = wolf_reward_fn or self._default_wolf_reward
        
        # Simulation state
        self.step_count = 0
        self.total_dog_reward = 0.0
        self.total_wolf_reward = 0.0
        
        # Previous state for reward calculation
        self.prev_sheep_in_pen = 0
        self.prev_sheep_count = self.num_sheep
        
        # Previous info dict (for step-to-step delta rewards)
        self.prev_info = None
        
        # Wolf state tracking (step-based, not time-based)
        self.wolf_cooldown_steps = 0  # Steps remaining in eating cooldown
        self.wolf_is_in_cooldown = False
        self.wolf_respawn_cooldown_steps = 0  # Steps remaining until wolf respawns
        self.wolf_is_dead = False
        
        # Pre-generate wall positions for observation grids
        self.wall_positions = generate_wall_positions(
            self.width, 
            self.height, 
            config.BOUNDARY_OFFSET//2, 
            resolution=20  # Sample walls every 20 pixels
        )
    
    def _spawn_entities(self):
        """Spawn sheep, dog, and wolf in the environment."""
        # Spawn sheep in center area
        for i in range(self.num_sheep):
            x = np.random.uniform(config.SHEEP_SPAWN_X_MIN, config.SHEEP_SPAWN_X_MAX)
            y = np.random.uniform(config.SHEEP_SPAWN_Y_MIN, config.SHEEP_SPAWN_Y_MAX)
            sheep = Sheep((x, y), self.sheep_sprite, sheep_id=i)
            self.env.add_sheep(sheep)
        
        # Spawn dog (bottom left area)
        dog = Dog((config.DOG_SPAWN_X, config.DOG_SPAWN_Y), self.dog_sprite)
        self.env.set_dog(dog)
        
        # Spawn wolf (left side of map)
        self._spawn_wolf()
    
    def _spawn_wolf(self):
        """Spawn or respawn wolf on the left side of the map."""
        x = np.random.uniform(config.WOLF_SPAWN_X_MIN, config.WOLF_SPAWN_X_MAX)
        y = np.random.uniform(config.WOLF_SPAWN_Y_MIN, config.WOLF_SPAWN_Y_MAX)
        wolf = Wolf((x, y), self.wolf_sprite)
        self.env.set_wolf(wolf)
        # Reset wolf cooldown when spawning
        self.wolf_cooldown_steps = 0
        self.wolf_is_in_cooldown = False
    
    def _get_state(self) -> SimulationState:
        """Get current simulation state."""
        # Get sheep data
        if len(self.env.sheep_list) > 0:
            sheep_positions = np.array([sheep.position for sheep in self.env.sheep_list])
            sheep_velocities = np.array([sheep.velocity for sheep in self.env.sheep_list])
            sheep_ids = np.array([sheep.id for sheep in self.env.sheep_list])
        else:
            sheep_positions = np.array([]).reshape(0, 2)
            sheep_velocities = np.array([]).reshape(0, 2)
            sheep_ids = np.array([], dtype=int)
        
        # Get pen entrance info
        pen_info = (
            self.env.pen_entrance_x,
            self.env.pen_entrance_top,
            self.env.pen_entrance_bottom,
            self.env.pen_entrance_width
        )
        
        # Handle wolf position/velocity when dead
        if self.wolf_is_dead or self.env.wolf is None:
            # Use a placeholder position far off screen
            wolf_position = np.array([-1000.0, -1000.0])
            wolf_velocity = np.array([0.0, 0.0])
        else:
            wolf_position = self.env.wolf.position.copy()
            wolf_velocity = self.env.wolf.velocity.copy()
        
        return SimulationState(
            dog_position=self.env.dog.position.copy(),
            dog_velocity=self.env.dog.velocity.copy(),
            wolf_position=wolf_position,
            wolf_velocity=wolf_velocity,
            sheep_positions=sheep_positions,
            sheep_velocities=sheep_velocities,
            sheep_ids=sheep_ids,
            sheep_count=len(self.env.sheep_list),
            sheep_in_pen=self.env.sheep_in_pen,
            pen_entrance_position=pen_info
        )
    
    def _get_dog_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get dog's observation grid and metadata vector.
        
        Returns local or global observation based on config.USE_GLOBAL_STATE_OBSERVATION.
        
        Returns:
            Tuple of (obs_grid, metadata_vector)
            
            Local mode:
                obs_grid: (3, grid_size, grid_size) - [enemy, sheep, walls] in ego-centric frame
                metadata_vector: (2,) unit vector pointing to pen
            
            Global mode:
                obs_grid: (1, grid_size, grid_size) - [sheep] in world frame
                metadata_vector: (4,) [agent_x, agent_y, enemy_x, enemy_y] normalized
        """
        if config.USE_GLOBAL_STATE_OBSERVATION:
            return self._get_global_observation('dog')
        else:
            return self._get_local_observation('dog')
    
    def _get_wolf_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get wolf's observation grid and metadata vector.
        
        Returns local or global observation based on config.USE_GLOBAL_STATE_OBSERVATION.
        
        Returns:
            Tuple of (obs_grid, metadata_vector)
            
            Local mode:
                obs_grid: (3, grid_size, grid_size) - [enemy, sheep, walls] in ego-centric frame
                metadata_vector: (2,) unit vector pointing to pen
            
            Global mode:
                obs_grid: (1, grid_size, grid_size) - [sheep] in world frame
                metadata_vector: (4,) [agent_x, agent_y, enemy_x, enemy_y] normalized
        """
        # If wolf is dead, return empty observation
        if self.wolf_is_dead or self.env.wolf is None:
            if config.USE_GLOBAL_STATE_OBSERVATION:
                empty_obs = np.zeros((1, config.OBSERVATION_GRID_SIZE, config.OBSERVATION_GRID_SIZE), dtype=np.float32)
                zero_metadata = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # heading points right by default
            else:
                empty_obs = np.zeros((3, config.OBSERVATION_GRID_SIZE, config.OBSERVATION_GRID_SIZE), dtype=np.float32)
                zero_metadata = np.array([0.0, 0.0], dtype=np.float32)
            return empty_obs, zero_metadata
        
        if config.USE_GLOBAL_STATE_OBSERVATION:
            return self._get_global_observation('wolf')
        else:
            return self._get_local_observation('wolf')
    
    def _get_local_observation(self, agent_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get agent's local ego-centric observation (rotated by agent heading).
        
        Args:
            agent_type: 'dog' or 'wolf'
        
        Returns:
            Tuple of (obs_grid, pen_direction_vector)
        """
        # Collect entity positions
        sheep_positions = [sheep.position for sheep in self.env.sheep_list]
        
        if agent_type == 'dog':
            # For dog: wolf is enemy
            if self.wolf_is_dead or self.env.wolf is None:
                enemy_positions = []
            else:
                enemy_positions = [self.env.wolf.position]
            agent_position = self.env.dog.position
            agent_heading = self.env.dog.heading
        else:  # wolf
            # For wolf: dog is enemy
            enemy_positions = [self.env.dog.position]
            agent_position = self.env.wolf.position
            agent_heading = self.env.wolf.heading
        
        entities = {
            'enemy': enemy_positions,
            'sheep': sheep_positions,
            'walls': self.wall_positions
        }
        
        # Get pen entrance position
        pen_info = (
            self.env.pen_entrance_x,
            self.env.pen_entrance_top,
            self.env.pen_entrance_bottom,
            self.env.pen_entrance_width
        )
        
        return create_observation_grid(
            agent_position=agent_position,
            agent_heading=agent_heading,
            entities_by_type=entities,
            grid_size=config.OBSERVATION_GRID_SIZE,
            observation_range=config.OBSERVATION_RANGE,
            screen_width=self.width,
            screen_height=self.height,
            pen_entrance_pos=pen_info
        )
    
    def _get_global_observation(self, agent_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get agent's global observation (entire map, fixed reference frame).
        
        Args:
            agent_type: 'dog' or 'wolf'
        
        Returns:
            Tuple of (obs_grid, metadata_vector)
                obs_grid: (1, grid_size, grid_size) - sheep only in world frame
                metadata_vector: (6,) [agent_x, agent_y, heading_cos, heading_sin, enemy_x, enemy_y] normalized
        """
        # Collect entity positions
        sheep_positions = [sheep.position for sheep in self.env.sheep_list]
        
        if agent_type == 'dog':
            # For dog: wolf is enemy
            if self.wolf_is_dead or self.env.wolf is None:
                enemy_positions = []
            else:
                enemy_positions = [self.env.wolf.position]
            agent_position = self.env.dog.position
            agent_heading = self.env.dog.heading
        else:  # wolf
            # For wolf: dog is enemy
            enemy_positions = [self.env.dog.position]
            agent_position = self.env.wolf.position
            agent_heading = self.env.wolf.heading
        
        # For global observations:
        # - Sheep go into the grid (1 channel with Gaussian splatting)
        # - Enemy position goes into metadata vector (extracted by create_global_observation_grid)
        entities = {
            'sheep': sheep_positions,
            'enemy': enemy_positions,  # Used for metadata vector, not grid
        }
        
        # Get pen entrance position (not used in global mode, but needed for function signature)
        pen_info = (
            self.env.pen_entrance_x,
            self.env.pen_entrance_top,
            self.env.pen_entrance_bottom,
            self.env.pen_entrance_width
        )
        
        return create_global_observation_grid(
            agent_position=agent_position,
            agent_heading=agent_heading,
            entities_by_type=entities,
            grid_size=config.OBSERVATION_GRID_SIZE,
            screen_width=self.width,
            screen_height=self.height,
            pen_entrance_pos=pen_info
        )
    
    def _default_dog_reward(self, info: Dict[str, Any], prev_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Default reward function for the dog.
        Rewards getting sheep into the pen.
        
        Args:
            info: Current step info dict
            prev_info: Previous step info dict (can be None on first step)
        """
        reward = 0.0
        
        # Reward for each sheep that entered the pen this step
        sheep_entered = info.get('sheep_entered_pen', 0)
        reward += sheep_entered * 10.0
        
        # Small penalty for being far from sheep (encourage herding)
        sheep_remaining = info.get('sheep_remaining', 0)
        if sheep_remaining > 0:
            sheep_positions = info.get('sheep_positions')
            dog_position = info.get('dog_position')
            sheep_center = np.mean(sheep_positions, axis=0)
            distance_to_flock = np.linalg.norm(dog_position - sheep_center)
            reward -= distance_to_flock * 0.001
        
        return reward
    
    def _default_wolf_reward(self, info: Dict[str, Any], prev_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Default reward function for the wolf.
        Rewards staying close to sheep and preventing them from entering the pen.
        
        Args:
            info: Current step info dict
            prev_info: Previous step info dict (can be None on first step)
        """
        reward = 0.0
        
        # Penalty for each sheep that entered the pen
        sheep_entered = info.get('sheep_entered_pen', 0)
        reward -= sheep_entered * 5.0
        
        # Reward for being close to sheep
        sheep_remaining = info.get('sheep_remaining', 0)
        if sheep_remaining > 0:
            # Find closest sheep
            sheep_positions = info.get('sheep_positions')
            wolf_position = info.get('wolf_position')
            distances = np.linalg.norm(sheep_positions - wolf_position, axis=1)
            min_distance = np.min(distances)
            
            # Reward proximity (closer is better)
            proximity_reward = max(0, 100 - min_distance) * 0.01
            reward += proximity_reward
        
        return reward
    
    def step(self, 
             dog_action: DogAction, 
             wolf_action: WolfAction,
             dt: float = 1.0,
             episode: int = 0) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """
        Execute one simulation step.
        
        Args:
            dog_action: Action for the dog agent
            wolf_action: Action for the wolf agent
            dt: Time step (default 1.0)
        
        Returns:
            Tuple of (dog_obs, dog_pen_vec, dog_reward, wolf_obs, wolf_pen_vec, wolf_reward, info)
            - dog_obs: Observation grid for dog (3, grid_size, grid_size)
            - dog_pen_vec: Pen direction vector for dog (2,)
            - dog_reward: Reward for dog
            - wolf_obs: Observation grid for wolf (3, grid_size, grid_size)
            - wolf_pen_vec: Pen direction vector for wolf (2,)
            - wolf_reward: Reward for wolf
            - info: Dictionary with additional information
        """
        # Update wolf eating cooldown (step-based, deterministic)
        if self.wolf_is_in_cooldown:
            self.wolf_cooldown_steps -= 1
            if self.wolf_cooldown_steps <= 0:
                self.wolf_is_in_cooldown = False
                self.wolf_cooldown_steps = 0
        
        # Update wolf respawn cooldown
        if self.wolf_is_dead:
            self.wolf_respawn_cooldown_steps -= 1
            if self.wolf_respawn_cooldown_steps <= 0:
                # Respawn wolf
                self._spawn_wolf()
                self.wolf_is_dead = False
        
        # Store previous values for reward calculation
        prev_sheep_in_pen = self.env.sheep_in_pen
        prev_sheep_count = len(self.env.sheep_list)
        
        # ===================================================================
        # CRITICAL: Get CURRENT observation BEFORE any entities are removed
        # This ensures the agent sees the state that led to the reward
        # ===================================================================
        current_dog_obs, current_dog_metadata = self._get_dog_observation()
        current_wolf_obs, current_wolf_metadata = self._get_wolf_observation()
        
        # Apply actions to dog and wolf (now using forward speed and turn rate)
        dog_forward, dog_turn = dog_action.to_vector()
        
        # Wolf can only move if not in cooldown and not dead
        if self.wolf_is_in_cooldown or self.wolf_is_dead:
            wolf_forward, wolf_turn = 0.0, 0.0  # Zero action during cooldown or when dead
        else:
            wolf_forward, wolf_turn = wolf_action.to_vector()
        
        # Apply actions using the new apply_action method
        self.env.dog.apply_action(dog_forward, dog_turn, dt)
        
        # Wolf only updates if not in cooldown and not dead
        if not self.wolf_is_in_cooldown and not self.wolf_is_dead and self.env.wolf is not None:
            self.env.wolf.apply_action(wolf_forward, wolf_turn, dt)
        
        # Enforce boundaries for dog
        self.env.enforce_boundaries(self.env.dog)
        
        # Enforce boundaries for wolf (only if alive)
        if not self.wolf_is_dead and self.env.wolf is not None:
            self.env.enforce_boundaries(self.env.wolf)
        
        # Check for wolf eating sheep (only if alive and not in cooldown)
        sheep_eaten = 0
        sheep_eaten_ids = []
        wolf_just_ate = False  # Track if wolf ate THIS step (for transition storage)
        if not self.wolf_is_in_cooldown and not self.wolf_is_dead and len(self.env.sheep_list) > 0:
            sheep_to_remove = []
            for sheep in self.env.sheep_list:
                distance = np.linalg.norm(self.env.wolf.position - sheep.position)
                if distance <= config.WOLF_EAT_DISTANCE:
                    sheep_to_remove.append(sheep)
                    sheep_eaten_ids.append(sheep.id)
                    sheep_eaten += 1
                    wolf_just_ate = True  # Wolf ate on THIS step
                    # Wolf enters cooldown after eating (step-based)
                    self.wolf_is_in_cooldown = True
                    self.env.increment_eaten()
                    self.wolf_cooldown_steps = config.WOLF_EAT_COOLDOWN_STEPS
                    break  # Wolf can only eat one sheep at a time
            
            # Remove eaten sheep
            for sheep in sheep_to_remove:
                self.env.sheep_list.remove(sheep)
        
        # Check for dog killing wolf (only if alive)
        wolf_killed = False
        if not self.wolf_is_dead and self.env.wolf is not None:
            distance = np.linalg.norm(self.env.dog.position - self.env.wolf.position)
            if distance <= config.DOG_KILL_WOLF_DISTANCE:
                wolf_killed = True
                self.wolf_is_dead = True
                self.wolf_respawn_cooldown_steps = config.WOLF_RESPAWN_COOLDOWN_STEPS
                # Remove wolf from environment (will respawn after cooldown)
                self.env.wolf = None
        
        # Update environment (sheep movement, pen checks, etc.)
        sheep_entered_ids = self.env.update(dt)
        
        # Get current state
        state = self._get_state()
        
        # Calculate pen center (useful for reward functions)
        pen_center = np.array([
            self.env.pen_entrance_x,
            (self.env.pen_entrance_top + self.env.pen_entrance_bottom) / 2
        ])
        
        # Calculate info
        sheep_entered = self.env.sheep_in_pen - prev_sheep_in_pen
        info = {
            'step': self.step_count,
            'sheep_entered_pen': sheep_entered,
            'sheep_eaten': sheep_eaten,
            'wolf_killed': wolf_killed,
            'wolf_just_ate': wolf_just_ate,  # Flag for when wolf eats THIS step (for transition storage)
            'wolf_in_cooldown': self.wolf_is_in_cooldown,
            'wolf_cooldown_remaining': self.wolf_cooldown_steps,  # Now in steps
            'wolf_is_dead': self.wolf_is_dead,  # Flag for skipping wolf transitions
            'total_sheep_in_pen': self.env.sheep_in_pen,
            'sheep_remaining': len(self.env.sheep_list),
            'done': len(self.env.sheep_list) == 0,  # Episode done when all sheep are gone
            # Agent positions and velocities
            'dog_position': state.dog_position,
            'dog_velocity': state.dog_velocity,
            'wolf_position': state.wolf_position,
            'wolf_velocity': state.wolf_velocity,
            'sheep_positions': state.sheep_positions,
            'sheep_velocities': state.sheep_velocities,
            'sheep_ids': state.sheep_ids,
            # Pen center for distance calculations
            'pen_center': pen_center,
            # IDs of sheep that were affected this step
            'sheep_entered_ids': sheep_entered_ids,
            'sheep_eaten_ids': sheep_eaten_ids,
            'episode': episode
        }
        
        # Calculate rewards
        dog_reward = self.dog_reward_fn(info, self.prev_info)
        wolf_reward = self.wolf_reward_fn(info, self.prev_info)
        
        # Store current info as previous for next step
        self.prev_info = info.copy()
        
        # Update totals
        self.total_dog_reward += dog_reward
        self.total_wolf_reward += wolf_reward
        self.step_count += 1
        
        next_dog_obs, next_dog_metadata = self._get_dog_observation()
        next_wolf_obs, next_wolf_metadata = self._get_wolf_observation()
        
        return next_dog_obs, next_dog_metadata, dog_reward, next_wolf_obs, next_wolf_metadata, wolf_reward, info
    
    def render(self, fps: int = 60) -> Optional[pygame.Surface]:
        """
        Render the current simulation state.
        
        Args:
            fps: Frames per second (only used if not headless)
        
        Returns:
            The rendered surface (or None if headless)
        """
        if self.headless:
            return None
        
        # Calculate cooldown ratio for visual feedback (step-based)
        cooldown_ratio = 0.0
        if self.wolf_is_in_cooldown and config.WOLF_EAT_COOLDOWN_STEPS > 0:
            cooldown_ratio = self.wolf_cooldown_steps / config.WOLF_EAT_COOLDOWN_STEPS
        
        # Render environment with wolf cooldown info
        self.env.render(self.screen, self.wolf_is_in_cooldown, cooldown_ratio)
        
        # Render observation grids if debug mode is enabled
        if config.DEBUG_SHOW_OBSERVATIONS:
            # Get current observations
            dog_obs, dog_metadata = self._get_dog_observation()
            wolf_obs, wolf_metadata = self._get_wolf_observation()
            
            if config.USE_GLOBAL_STATE_OBSERVATION:
                # Render global observation (covers entire screen)
                # Show dog's observation (what the dog agent sees)
                render_global_observation_grid(
                    self.screen,
                    dog_obs,
                    dog_metadata,
                    self.width,
                    self.height,
                    alpha=70,
                    label="Dog's Global Observation"
                )
            else:
                # Render local ego-centric observations
                # Render dog's observation grid
                render_observation_grid(
                    self.screen,
                    dog_obs,
                    self.env.dog.position,
                    self.env.dog.heading,
                    config.OBSERVATION_RANGE,
                    alpha=100
                )
                
                # Render wolf's observation grid (if wolf is alive)
                if not self.wolf_is_dead and self.env.wolf is not None:
                    render_observation_grid(
                        self.screen,
                        wolf_obs,
                        self.env.wolf.position,
                        self.env.wolf.heading,
                        config.OBSERVATION_RANGE,
                        alpha=100
                    )
        
        # Render pen direction vectors if debug mode is enabled (only for local mode)
        if config.DEBUG_SHOW_PEN_DIRECTION and not config.USE_GLOBAL_STATE_OBSERVATION:
            # Get current pen vectors (only meaningful in local mode)
            dog_obs, dog_pen_vec = self._get_dog_observation()
            
            # Draw dog's pen direction arrow (in world frame, not rotated)
            # Convert from agent frame back to world frame
            dog_heading = self.env.dog.heading
            cos_angle = np.cos(dog_heading)
            sin_angle = np.sin(dog_heading)
            world_pen_x = dog_pen_vec[0] * cos_angle - dog_pen_vec[1] * sin_angle
            world_pen_y = dog_pen_vec[0] * sin_angle + dog_pen_vec[1] * cos_angle
            
            arrow_length = 50
            end_x = self.env.dog.position[0] + world_pen_x * arrow_length
            end_y = self.env.dog.position[1] + world_pen_y * arrow_length
            
            # Draw arrow for dog (blue)
            pygame.draw.line(self.screen, (0, 100, 255), 
                           (int(self.env.dog.position[0]), int(self.env.dog.position[1])),
                           (int(end_x), int(end_y)), 3)
            # Draw arrowhead
            arrow_size = 10
            angle = np.arctan2(world_pen_y, world_pen_x)
            pygame.draw.polygon(self.screen, (0, 100, 255), [
                (int(end_x), int(end_y)),
                (int(end_x - arrow_size * np.cos(angle - np.pi/6)), 
                 int(end_y - arrow_size * np.sin(angle - np.pi/6))),
                (int(end_x - arrow_size * np.cos(angle + np.pi/6)), 
                 int(end_y - arrow_size * np.sin(angle + np.pi/6)))
            ])
            
            # Draw wolf's pen direction arrow (if alive)
            if not self.wolf_is_dead and self.env.wolf is not None:
                wolf_obs, wolf_pen_vec = self._get_wolf_observation()
                
                # Convert from agent frame back to world frame
                wolf_heading = self.env.wolf.heading
                cos_angle = np.cos(wolf_heading)
                sin_angle = np.sin(wolf_heading)
                world_pen_x = wolf_pen_vec[0] * cos_angle - wolf_pen_vec[1] * sin_angle
                world_pen_y = wolf_pen_vec[0] * sin_angle + wolf_pen_vec[1] * cos_angle
                
                end_x = self.env.wolf.position[0] + world_pen_x * arrow_length
                end_y = self.env.wolf.position[1] + world_pen_y * arrow_length
                
                # Draw arrow for wolf (red)
                pygame.draw.line(self.screen, (255, 50, 50), 
                               (int(self.env.wolf.position[0]), int(self.env.wolf.position[1])),
                               (int(end_x), int(end_y)), 3)
                # Draw arrowhead
                angle = np.arctan2(world_pen_y, world_pen_x)
                pygame.draw.polygon(self.screen, (255, 50, 50), [
                    (int(end_x), int(end_y)),
                    (int(end_x - arrow_size * np.cos(angle - np.pi/6)), 
                     int(end_y - arrow_size * np.sin(angle - np.pi/6))),
                    (int(end_x - arrow_size * np.cos(angle + np.pi/6)), 
                     int(end_y - arrow_size * np.sin(angle + np.pi/6)))
                ])
        
        # Update display
        pygame.display.flip()
        
        if self.clock is not None:
            self.clock.tick(fps)
        
        return self.screen
    
    def reset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Reset the simulation to initial state.
        
        Returns:
            Tuple of ((dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec))
                dog_obs: observation grid for dog
                dog_pen_vec: pen direction vector for dog
                wolf_obs: observation grid for wolf
                wolf_pen_vec: pen direction vector for wolf
        """
        # Clear existing entities
        self.env.sheep_list.clear()
        self.env.sheep_in_pen = 0
        
        # Respawn entities
        self._spawn_entities()
        
        # Reset counters
        self.step_count = 0
        self.total_dog_reward = 0.0
        self.total_wolf_reward = 0.0
        self.prev_sheep_in_pen = 0
        self.prev_sheep_count = self.num_sheep
        
        # Reset previous info
        self.prev_info = None
        
        # Reset wolf state (step-based)
        self.wolf_cooldown_steps = 0
        self.wolf_is_in_cooldown = False
        self.wolf_respawn_cooldown_steps = 0
        self.wolf_is_dead = False
        
        # Return observation grids and pen vectors
        dog_obs, dog_pen_vec = self._get_dog_observation()
        wolf_obs, wolf_pen_vec = self._get_wolf_observation()

        self.env.reset_eaten()

        return (dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec)
    
    def close(self):
        """Clean up resources."""
        if not self.headless:
            pygame.quit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            'step_count': self.step_count,
            'sheep_remaining': len(self.env.sheep_list),
            'sheep_in_pen': self.env.sheep_in_pen,
            'total_dog_reward': self.total_dog_reward,
            'total_wolf_reward': self.total_wolf_reward
        }
