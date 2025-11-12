"""
Manual control script for testing sheep herding dynamics.
Dog is controlled by WASD keys, Wolf is controlled by arrow keys.
No training - just manual testing of the environment.
"""

import pygame
import numpy as np
import math

from simulator import Simulator
from agents.base_agent import BaseAgent
from actions import DogAction, WolfAction
import config


class HumanAgent(BaseAgent):
    """
    Human-controlled agent that reads keyboard input.
    """
    
    def __init__(self, agent_type="dog", control_keys="wasd", acceleration=0.5):
        """
        Args:
            agent_type: "dog" or "wolf"
            control_keys: "wasd" or "arrows"
            acceleration: Movement acceleration
        """
        self.agent_type = agent_type
        self.control_keys = control_keys
        self.acceleration = acceleration
    
    def act(self, observation: np.ndarray, pen_vector: np.ndarray):
        """Read keyboard input and return action."""
        keys = pygame.key.get_pressed()
        
        forward_speed = 0.0
        turn_rate = 0.0
        
        if self.control_keys == "wasd":
            # WASD controls - W for forward only, A/D for turning
            if keys[pygame.K_w]:
                forward_speed = self.acceleration
            # S key removed - no backward movement
            if keys[pygame.K_a]:
                turn_rate = -self.acceleration  # Turn left
            if keys[pygame.K_d]:
                turn_rate = self.acceleration   # Turn right
        else:
            # Arrow key controls - UP for forward only, LEFT/RIGHT for turning
            if keys[pygame.K_UP]:
                forward_speed = self.acceleration
            # DOWN key removed - no backward movement
            if keys[pygame.K_LEFT]:
                turn_rate = -self.acceleration  # Turn left
            if keys[pygame.K_RIGHT]:
                turn_rate = self.acceleration   # Turn right
        
        # Create action object with forward speed and turn rate
        if self.agent_type == "dog":
            return DogAction(forward_speed=forward_speed, turn_rate=turn_rate)
        else:
            return WolfAction(forward_speed=forward_speed, turn_rate=turn_rate)
    
    def observe(self, observation: np.ndarray, pen_vector: np.ndarray,
                action, reward: float, next_observation: np.ndarray,
                next_pen_vector: np.ndarray, done: bool, info: dict):
        """Human agent doesn't learn, so this is a no-op."""
        pass


def vector_to_degrees(vec_x, vec_y):
    """
    Convert a 2D vector to degrees (0-360).
    0° is right (positive x-axis), 90° is down (positive y-axis in pygame coords).
    """
    angle_rad = math.atan2(vec_y, vec_x)
    angle_deg = math.degrees(angle_rad)
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


def render_vector_info(screen, dog_pen_vec, wolf_pen_vec, wolf_is_dead):
    """Render pen vector information on screen."""
    # Initialize font if needed
    font = pygame.font.Font(None, 24)
    
    # Calculate angles
    dog_angle = vector_to_degrees(dog_pen_vec[0], dog_pen_vec[1])
    
    # Render dog info
    dog_text = f"Dog Pen Vec: ({dog_pen_vec[0]:.3f}, {dog_pen_vec[1]:.3f}) | {dog_angle:.1f}°"
    dog_surface = font.render(dog_text, True, (0, 150, 255))
    
    # Background for better readability
    bg_rect = dog_surface.get_rect()
    bg_rect.topleft = (10, 80)
    bg_rect.inflate_ip(10, 4)
    pygame.draw.rect(screen, (0, 0, 0), bg_rect)
    pygame.draw.rect(screen, (255, 255, 255), bg_rect, 1)

    screen.blit(dog_surface, (15, 82))
    
    # Render wolf info (if alive)
    if not wolf_is_dead:
        wolf_angle = vector_to_degrees(wolf_pen_vec[0], wolf_pen_vec[1])
        wolf_text = f"Wolf Pen Vec: ({wolf_pen_vec[0]:.3f}, {wolf_pen_vec[1]:.3f}) | {wolf_angle:.1f}°"
        wolf_surface = font.render(wolf_text, True, (255, 100, 100))
        
        bg_rect = wolf_surface.get_rect()
        bg_rect.topleft = (10, 110)
        bg_rect.inflate_ip(10, 4)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect)
        pygame.draw.rect(screen, (255, 255, 255), bg_rect, 1)
        
        screen.blit(wolf_surface, (15, 112))


def main():
    """Run manual control mode."""
    print("=" * 70)
    print("MANUAL CONTROL MODE")
    print("=" * 70)
    print()
    print("Controls:")
    print("  Dog:  W for forward, A/D for turn left/right")
    print("  Wolf: Arrow UP for forward, LEFT/RIGHT for turn")
    print("  ESC:  Quit")
    print("  R:    Reset episode")
    print()
    print("Try to herd the sheep into the pen on the right!")
    print("The wolf tries to eat the sheep.")
    print("=" * 70)
    print()
    
    # Create human-controlled agents
    dog_agent = HumanAgent(agent_type="dog", control_keys="wasd", acceleration=0.5)
    wolf_agent = HumanAgent(agent_type="wolf", control_keys="arrows", acceleration=0.5)
    
    # Create simulator with visualization
    simulator = Simulator(headless=False)
    
    # Reset environment
    (dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec) = simulator.reset()
    
    running = True
    step_count = 0
    episode_count = 1
    
    print(f"Episode {episode_count} started!")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset episode
                    (dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec) = simulator.reset()
                    step_count = 0
                    episode_count += 1
                    print(f"\nEpisode {episode_count} started!")
        
        if not running:
            break
        
        # Get actions from human input
        dog_action = dog_agent.act(dog_obs, dog_pen_vec)
        wolf_action = wolf_agent.act(wolf_obs, wolf_pen_vec)
        
        # Execute step
        next_dog_obs, next_dog_pen_vec, dog_reward, next_wolf_obs, next_wolf_pen_vec, wolf_reward, info = simulator.step(
            dog_action, wolf_action
        )
        
        # Render
        simulator.render(fps=60)
        
        # Render vector info overlay
        render_vector_info(simulator.screen, next_dog_pen_vec, next_wolf_pen_vec, simulator.wolf_is_dead)
        pygame.display.flip()  # Update display after adding overlay
        
        # Update observations
        dog_obs = next_dog_obs
        dog_pen_vec = next_dog_pen_vec
        wolf_obs = next_wolf_obs
        wolf_pen_vec = next_wolf_pen_vec
        
        step_count += 1
        
        # Print info periodically
        if step_count % 100 == 0:
            print(f"Step {step_count}: Sheep in pen: {simulator.env.sheep_in_pen}/{config.NUM_SHEEP}, "
                  f"Sheep remaining: {len(simulator.env.sheep_list)}")
        
        # Check if episode done
        if info['done']:
            print(f"\nEpisode {episode_count} complete!")
            print(f"  Steps: {step_count}")
            print(f"  Sheep saved: {simulator.env.sheep_in_pen}")
            print(f"  Sheep eaten: {config.NUM_SHEEP - simulator.env.sheep_in_pen - len(simulator.env.sheep_list)}")
            print(f"  Press R to restart or ESC to quit")
            
            # Wait for user input
            waiting = True
            while waiting and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False
                        elif event.key == pygame.K_r:
                            (dog_obs, dog_pen_vec), (wolf_obs, wolf_pen_vec) = simulator.reset()
                            step_count = 0
                            episode_count += 1
                            print(f"\nEpisode {episode_count} started!")
                            waiting = False
                
                # Keep rendering while waiting
                simulator.render(fps=60)
                render_vector_info(simulator.screen, dog_pen_vec, wolf_pen_vec, simulator.wolf_is_dead)
                pygame.display.flip()
    
    print("\nManual control session ended!")
    print(f"Total episodes: {episode_count}")
    simulator.close()


if __name__ == "__main__":
    main()
