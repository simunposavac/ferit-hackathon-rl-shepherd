"""
Base agent class that all RL agents must inherit from.
Provides a consistent interface for the training loop.
"""

from abc import ABC, abstractmethod
import numpy as np
from actions import DogAction, WolfAction


class BaseAgent(ABC):
    """
    Abstract base class for all agents (dog and wolf).
    
    Subclasses must implement:
    - act(): Select action given observation
    - observe(): Process a transition
    """
    
    @abstractmethod
    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        """
        Select an action given the current observation.
        
        Args:
            observation: (3, grid_size, grid_size) observation grid
                Channel 0: Enemy agent
                Channel 1: Sheep
                Channel 2: Walls
            pen_vector: (2,) unit vector pointing to pen entrance in agent's frame
        
        Returns:
            Action object (DogAction or WolfAction)
        """
        pass
    
    @abstractmethod
    def observe(self, observation: np.ndarray, pen_vector: np.ndarray, 
                action, reward: float, next_observation: np.ndarray, 
                next_pen_vector: np.ndarray, done: bool, info: dict):
        """
        Process a transition from the environment.
        
        This is where the agent should:
        - Store the transition in a replay buffer
        - Perform learning updates
        - Update internal state
        
        Args:
            observation: Current observation
            pen_vector: Current pen direction vector
            action: Action that was taken
            reward: Reward received
            next_observation: Next observation
            next_pen_vector: Next pen direction vector
            done: Whether episode ended
            info: Additional info from environment
        """
        pass
    
    def episode_start(self):
        """
        Called at the start of each episode.
        Override this if you need to reset episode-specific state.
        """
        pass
    
    def episode_end(self, total_reward: float):
        """
        Called at the end of each episode.
        Override this for episode-level updates or logging.
        
        Args:
            total_reward: Total reward accumulated during episode
        """
        pass
    
    def save(self, path: str):
        """
        Save agent's state to disk.
        Override this to implement checkpointing.
        
        Args:
            path: File path to save to
        """
        raise NotImplementedError("Save method not implemented for this agent")
    
    def load(self, path: str):
        """
        Load agent's state from disk.
        Override this to implement checkpoint loading.
        
        Args:
            path: File path to load from
        """
        raise NotImplementedError("Load method not implemented for this agent")


class RandomAgent(BaseAgent):
    """
    Simple random agent for testing.
    Takes random actions regardless of observations.
    """
    
    def __init__(self, agent_type: str = "dog", max_value: float = 1.0):
        """
        Args:
            agent_type: "dog" or "wolf"
            max_value: Maximum value for random actions (forward speed and turn rate)
        """
        self.agent_type = agent_type
        self.max_value = max_value
    
    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        """Return random action."""
        if self.agent_type == "dog":
            return DogAction.random(self.max_value)
        else:
            return WolfAction.random(self.max_value)
    
    def observe(self, observation: np.ndarray, pen_vector: np.ndarray,
                action, reward: float, next_observation: np.ndarray,
                next_pen_vector: np.ndarray, done: bool, info: dict):
        """Random agent doesn't learn, so this is a no-op."""
        pass
