"""
Action classes for controlling dog and wolf agents in the simulation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DogAction:
    """
    Action for the dog agent.
    
    The action represents forward speed and turn rate in the agent's local frame.
    """
    forward_speed: float  # Desired forward speed (0 to 1, scaled by max speed)
    turn_rate: float      # Desired turn rate (-1 to 1, scaled by max turn rate)

    
    def __post_init__(self):
        """Validate action values."""
        if not isinstance(self.forward_speed, (int, float)):
            raise TypeError(f"forward_speed must be a number, got {type(self.forward_speed)}")
        if not isinstance(self.turn_rate, (int, float)):
            raise TypeError(f"turn_rate must be a number, got {type(self.turn_rate)}")
    
    def to_vector(self):
        """Convert action to numpy vector."""
        return np.array([self.forward_speed, self.turn_rate], dtype=float)
    
    @classmethod
    def from_vector(cls, vector):
        """Create action from numpy vector or list."""
        if len(vector) != 2:
            raise ValueError(f"Vector must have length 2, got {len(vector)}")
        return cls(float(vector[0]), float(vector[1]))
    
    @classmethod
    def random(cls, max_value=1.0):
        """Generate random action."""
        forward_speed = np.random.uniform(0.0, max_value)  # Only forward, no backward
        turn_rate = np.random.uniform(-max_value, max_value)
        return cls(forward_speed, turn_rate)
    
    @classmethod
    def zero(cls):
        """Create zero action (no movement)."""
        return cls(0.0, 0.0)


@dataclass
class WolfAction:
    """
    Action for the wolf agent.
    
    The action represents forward speed and turn rate in the agent's local frame.
    """
    forward_speed: float  # Desired forward speed (0 to 1, scaled by max speed)
    turn_rate: float      # Desired turn rate (-1 to 1, scaled by max turn rate)
    
    def __post_init__(self):
        """Validate action values."""
        if not isinstance(self.forward_speed, (int, float)):
            raise TypeError(f"forward_speed must be a number, got {type(self.forward_speed)}")
        if not isinstance(self.turn_rate, (int, float)):
            raise TypeError(f"turn_rate must be a number, got {type(self.turn_rate)}")
    
    def to_vector(self):
        """Convert action to numpy vector."""
        return np.array([self.forward_speed, self.turn_rate], dtype=float)
    
    @classmethod
    def from_vector(cls, vector):
        """Create action from numpy vector or list."""
        if len(vector) != 2:
            raise ValueError(f"Vector must have length 2, got {len(vector)}")
        return cls(float(vector[0]), float(vector[1]))
    
    @classmethod
    def random(cls, max_value=1.0):
        """Generate random action."""
        forward_speed = np.random.uniform(0.0, max_value)  # Only forward, no backward
        turn_rate = np.random.uniform(-max_value, max_value)
        return cls(forward_speed, turn_rate)
    
    @classmethod
    def zero(cls):
        """Create zero action (no movement)."""
        return cls(0.0, 0.0)
