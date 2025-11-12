import numpy as np
import pygame
import config


class Sheep:
    """
    Sheep entity that follows boids flocking behavior with predator avoidance.
    """
    
    def __init__(self, position, sprite_path, sheep_id=None):
        """
        Initialize a sheep.
        
        Args:
            position: tuple (x, y) initial position
            sprite_path: str, path to sheep sprite image
            sheep_id: int, unique identifier for this sheep (optional)
        """
        self.id = sheep_id  # Unique identifier
        self.position = np.array(position, dtype=float)
        self.velocity = np.random.uniform(-1, 1, 2)  # Random initial velocity
        
        # Normalize initial velocity
        speed = np.linalg.norm(self.velocity)
        if speed > 0:
            self.velocity = self.velocity / speed * 0.5
        
        # Load and store original sprite (facing up)
        original_sprite = pygame.image.load(sprite_path)
        
        # Scale sprite using config values
        width, height = original_sprite.get_size()
        max_dimension = max(width, height)
        scale_factor = (config.SPRITE_MAX_DIMENSION / max_dimension) * config.SHEEP_SPRITE_SCALE
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        self.original_sprite = pygame.transform.scale(original_sprite, (new_width, new_height))
        self.sprite = self.original_sprite
        self.rect = self.sprite.get_rect(center=self.position)
        
        # Sheep parameters from config
        self.max_speed = config.SHEEP_MAX_SPEED
        self.separation_radius = config.SHEEP_SEPARATION_RADIUS
        self.alignment_radius = config.SHEEP_ALIGNMENT_RADIUS
        self.cohesion_radius = config.SHEEP_COHESION_RADIUS
        self.detection_radius = config.SHEEP_DETECTION_RADIUS
        self.panic_radius_wolf = config.SHEEP_PANIC_RADIUS_WOLF
        self.panic_radius_dog = config.SHEEP_PANIC_RADIUS_DOG
        self.field_of_view = config.SHEEP_FIELD_OF_VIEW
        
        # Movement smoothing parameters from config
        self.max_turn_rate = config.SHEEP_MAX_TURN_RATE
        self.velocity_smoothing = config.SHEEP_VELOCITY_SMOOTHING
        
        # State
        self.panic_mode = False
        
    def update_position(self, new_velocity, dt=1.0):
        """
        Update sheep position based on new velocity with smoothing and turn rate limiting.
        
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
        
    def get_neighbors(self, all_sheep, radius):
        """
        Get sheep within a certain radius.
        
        Args:
            all_sheep: list of Sheep objects
            radius: float, detection radius
            
        Returns:
            list of Sheep objects within radius
        """
        neighbors = []
        for sheep in all_sheep:
            if sheep is not self:
                distance = np.linalg.norm(sheep.position - self.position)
                if distance < radius:
                    neighbors.append(sheep)
        return neighbors
    
    def can_detect_predator(self, predator_position):
        """
        Check if sheep can detect a predator based on vision cone.
        
        Args:
            predator_position: np.array, position of predator
            
        Returns:
            bool, True if predator is detected
        """
        # Check distance
        to_predator = predator_position - self.position
        distance = np.linalg.norm(to_predator)
        
        if distance > self.detection_radius:
            return False
        
        # Check angle (field of view)
        speed = np.linalg.norm(self.velocity)
        if speed < 0.1:  # If nearly stationary, can see all directions
            return True
        
        # Calculate angle between heading and direction to predator
        velocity_normalized = self.velocity / speed
        to_predator_normalized = to_predator / distance
        
        dot_product = np.dot(velocity_normalized, to_predator_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        angle_degrees = np.degrees(angle)
        
        return angle_degrees <= self.field_of_view / 2
    
    def render(self, screen):
        """
        Render the sheep on the screen with rotation based on velocity.
        
        Args:
            screen: pygame surface
        """
        # Calculate rotation angle from velocity
        if np.linalg.norm(self.velocity) > 0.1:
            # Angle in degrees, pygame rotates counter-clockwise
            # Sprite initially faces up (negative y direction)
            angle = np.degrees(np.arctan2(-self.velocity[0], -self.velocity[1]))
            self.sprite = pygame.transform.rotate(self.original_sprite, angle)
        else:
            self.sprite = self.original_sprite
        
        # Update rect for centered rendering
        self.rect = self.sprite.get_rect(center=self.position)
        screen.blit(self.sprite, self.rect)
