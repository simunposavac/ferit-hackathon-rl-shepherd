import numpy as np
import pygame
import config


class Dog:
    """
    Sheepdog entity - controlled by RL agent in the future.
    """
    
    def __init__(self, position, sprite_path):
        """
        Initialize a dog.
        
        Args:
            position: tuple (x, y) initial position
            sprite_path: str, path to dog sprite image
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        
        # Load and store original sprite (facing up)
        original_sprite = pygame.image.load(sprite_path)
        
        # Scale sprite using config values
        width, height = original_sprite.get_size()
        max_dimension = max(width, height)
        scale_factor = config.SPRITE_MAX_DIMENSION / max_dimension
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        self.original_sprite = pygame.transform.scale(original_sprite, (new_width, new_height))
        self.sprite = self.original_sprite
        self.rect = self.sprite.get_rect(center=self.position)
        
        # Dog parameters from config
        self.max_speed = config.DOG_MAX_SPEED
        self.threat_weight = config.DOG_THREAT_WEIGHT
        
        # Movement smoothing parameters from config
        self.max_turn_rate = config.DOG_MAX_TURN_RATE
        self.velocity_smoothing = config.DOG_VELOCITY_SMOOTHING
        
        # Heading angle (derived from velocity, used for local frame actions)
        self.heading = 0.0
    
    def apply_action(self, forward_speed, turn_rate, dt=1.0):
        """
        Apply action in local frame (forward speed and turn rate).
        
        Args:
            forward_speed: Desired forward speed (0 to 1, scaled internally)
            turn_rate: Desired turn rate (-1 to 1, scaled internally)
            dt: Time step
        """
        # Clamp inputs to valid ranges
        forward_speed = np.clip(forward_speed, 0.0, 1.0)  # Only forward movement
        turn_rate = np.clip(turn_rate, -1.0, 1.0)
        
        # Scale to actual values
        desired_speed = forward_speed * self.max_speed
        delta_heading = turn_rate * self.max_turn_rate
        
        # Update heading purely from turn command
        self.heading += delta_heading
        # Keep heading within [-pi, pi] for numerical stability
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi
        
        # Compute desired velocity from heading and speed
        desired_velocity = np.array([
            desired_speed * np.cos(self.heading),
            desired_speed * np.sin(self.heading)
        ])
        
        # Apply smoothing
        smoothed_velocity = (1 - self.velocity_smoothing) * self.velocity + self.velocity_smoothing * desired_velocity
        
        # Update velocity and position
        self.velocity = smoothed_velocity
        self.position += self.velocity * dt
        # Align heading with motion when position is set directly
        current_speed = np.linalg.norm(self.velocity)
        if current_speed > 1e-3:
            self.heading = np.arctan2(self.velocity[1], self.velocity[0])
        
    def update_position(self, new_velocity, dt=1.0):
        """
        Update dog position based on new velocity with smoothing and turn rate limiting.
        
        Args:
            new_velocity: np.array, new velocity vector
            dt: float, time step
        """
        # Limit to max speed
        speed = np.linalg.norm(new_velocity)
        if speed > self.max_speed:
            new_velocity = new_velocity / speed * self.max_speed
        
        # Apply velocity smoothing (moving average)
        # New velocity is a blend of old and new
        smoothed_velocity = (1 - self.velocity_smoothing) * self.velocity + self.velocity_smoothing * new_velocity
        
        # Limit turn rate (maximum change in direction)
        current_speed = np.linalg.norm(self.velocity)
        new_speed = np.linalg.norm(smoothed_velocity)
        
        if current_speed > 0.01 and new_speed > 0.01:
            # Calculate current and new directions
            current_direction = self.velocity / current_speed
            new_direction = smoothed_velocity / new_speed
            
            # Calculate the angle difference
            dot_product = np.dot(current_direction, new_direction)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_diff = np.arccos(dot_product)
            
            # If turn is too sharp, limit it
            if angle_diff > self.max_turn_rate:
                # Interpolate between current and new direction
                interpolation_factor = self.max_turn_rate / angle_diff
                limited_direction = current_direction * (1 - interpolation_factor) + new_direction * interpolation_factor
                
                # Normalize and apply speed
                limited_direction = limited_direction / np.linalg.norm(limited_direction)
                smoothed_velocity = limited_direction * new_speed
        
        self.velocity = smoothed_velocity
        self.position += self.velocity * dt
    
    def render(self, screen):
        """
        Render the dog on the screen with rotation based on heading.
        
        Args:
            screen: pygame surface
        """
        # Use heading for rotation
        # Convert heading (in radians) to degrees and negate for pygame's clockwise rotation
        angle = -np.degrees(self.heading) - 90
        self.sprite = pygame.transform.rotate(self.original_sprite, angle)
        
        # Update rect for centered rendering
        self.rect = self.sprite.get_rect(center=self.position)
        screen.blit(self.sprite, self.rect)
