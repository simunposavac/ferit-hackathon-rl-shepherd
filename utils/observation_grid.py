"""
Observation grid utility for generating agent-centric local observations.
Creates a grid-based representation of the environment from each agent's perspective.
Supports Gaussian and bilinear splatting for smooth entity representation.
"""

import numpy as np
import pygame
import config


def gaussian_kernel_2d(sigma, radius):
    """
    Create a 2D Gaussian kernel for splatting.
    
    Args:
        sigma: standard deviation in grid cells
        radius: truncation radius in grid cells
        
    Returns:
        2D numpy array with Gaussian weights
    """
    size = 2 * radius + 1
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel


def splat_entity_gaussian(grid, grid_x_float, grid_y_float, value, sigma, radius):
    """
    Splat an entity onto the grid using Gaussian weighting (vectorized).
    
    Args:
        grid: 2D numpy array to splat onto
        grid_x_float: float x position in grid coordinates
        grid_y_float: float y position in grid coordinates
        value: value to add (weight for this entity)
        sigma: Gaussian standard deviation
        radius: truncation radius
    """
    grid_size = grid.shape[0]
    
    # Get integer center position
    center_x = int(np.round(grid_x_float))
    center_y = int(np.round(grid_y_float))
    
    # Calculate sub-pixel offset
    offset_x = grid_x_float - center_x
    offset_y = grid_y_float - center_y
    
    # Generate Gaussian kernel
    kernel = gaussian_kernel_2d(sigma, radius)
    
    # Determine bounds
    y_start = max(0, center_y - radius)
    y_end = min(grid_size, center_y + radius + 1)
    x_start = max(0, center_x - radius)
    x_end = min(grid_size, center_x + radius + 1)
    
    # Determine kernel bounds
    ky_start = max(0, radius - center_y)
    ky_end = ky_start + (y_end - y_start)
    kx_start = max(0, radius - center_x)
    kx_end = kx_start + (x_end - x_start)
    
    # Apply kernel (vectorized)
    if y_end > y_start and x_end > x_start:
        grid[y_start:y_end, x_start:x_end] += kernel[ky_start:ky_end, kx_start:kx_end] * value


def splat_entity_bilinear(grid, grid_x_float, grid_y_float, value):
    """
    Splat an entity onto the grid using bilinear interpolation (vectorized).
    
    Args:
        grid: 2D numpy array to splat onto
        grid_x_float: float x position in grid coordinates
        grid_y_float: float y position in grid coordinates
        value: value to add (weight for this entity)
    """
    grid_size = grid.shape[0]
    
    # Get integer bounds
    x0 = int(np.floor(grid_x_float))
    x1 = x0 + 1
    y0 = int(np.floor(grid_y_float))
    y1 = y0 + 1
    
    # Calculate fractional parts
    fx = grid_x_float - x0
    fy = grid_y_float - y0
    
    # Bilinear weights
    weights = [
        (1 - fx) * (1 - fy),  # (x0, y0)
        fx * (1 - fy),         # (x1, y0)
        (1 - fx) * fy,         # (x0, y1)
        fx * fy                # (x1, y1)
    ]
    
    positions = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    
    # Apply weights (vectorized check and assignment)
    for (x, y), weight in zip(positions, weights):
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y, x] += weight * value


def create_observation_grid(agent_position, agent_heading, entities_by_type, grid_size, observation_range, screen_width, screen_height, pen_entrance_pos):
    """
    Create a grid-based observation centered on an agent with rotation.
    
    Args:
        agent_position: (x, y) position of the observing agent
    agent_heading: heading angle (radians) describing agent orientation
        entities_by_type: dict with keys 'enemy', 'sheep', 'walls' containing entity positions
        grid_size: NxN size of the observation grid
        observation_range: radius of observation in pixels
        screen_width: width of the screen
        screen_height: height of the screen
        pen_entrance_pos: (x, top_y, bottom_y, width) pen entrance position
    
    Returns:
        tuple: (obs_grid, pen_direction_vector)
            obs_grid: (3, grid_size, grid_size) observation grid
            pen_direction_vector: (2,) unit vector pointing to nearest pen point
    """
    # Initialize observation grid (3 channels)
    obs_grid = np.zeros((3, grid_size, grid_size), dtype=np.float32)

    # Use provided heading angle
    heading_angle = float(agent_heading)
    
    # Precompute rotation matrices
    cos_angle = np.cos(-heading_angle)
    sin_angle = np.sin(-heading_angle)
    
    # Cell size in world coordinates
    cell_size = (2 * observation_range) / grid_size
    
    # Determine splatting method
    splatting_method = config.SPLATTING_METHOD
    
    # Process each entity type
    for channel_idx, entity_type in enumerate(['enemy', 'sheep', 'walls']):
        if entity_type not in entities_by_type:
            continue
        
        entities = entities_by_type[entity_type]
        if entities is None or len(entities) == 0:
            continue
        
        # Convert to numpy array
        if not isinstance(entities, np.ndarray):
            entities = np.array(entities, dtype=np.float32)
        
        if len(entities.shape) == 1:
            entities = entities.reshape(-1, 2)
        
        # Vectorized transformation
        relative_pos = entities - agent_position
        
        # Rotate all positions at once
        rotated_x = relative_pos[:, 0] * cos_angle - relative_pos[:, 1] * sin_angle
        rotated_y = relative_pos[:, 0] * sin_angle + relative_pos[:, 1] * cos_angle
        
        # Filter by observation range
        in_range = (np.abs(rotated_x) <= observation_range) & (np.abs(rotated_y) <= observation_range)
        
        if not np.any(in_range):
            continue
        
        rotated_x = rotated_x[in_range]
        rotated_y = rotated_y[in_range]
        
        # Convert to grid coordinates (float for splatting)
        grid_x_float = (rotated_x + observation_range) / cell_size
        grid_y_float = (rotated_y + observation_range) / cell_size
        
        # Apply splatting or discrete placement
        if splatting_method == 'gaussian' and entity_type == 'sheep':
            # Use Gaussian splatting for sheep
            for gx, gy in zip(grid_x_float, grid_y_float):
                splat_entity_gaussian(
                    obs_grid[channel_idx], gx, gy, 1.0,
                    config.GAUSSIAN_SIGMA, config.GAUSSIAN_RADIUS
                )
            # Normalize after splatting
            obs_grid[channel_idx] = np.clip(obs_grid[channel_idx] / 5.0, 0.0, 1.0)
        elif splatting_method == 'bilinear' and entity_type == 'sheep':
            # Use bilinear splatting for sheep
            for gx, gy in zip(grid_x_float, grid_y_float):
                splat_entity_bilinear(obs_grid[channel_idx], gx, gy, 1.0)
            # Normalize after splatting
            obs_grid[channel_idx] = np.clip(obs_grid[channel_idx] / 5.0, 0.0, 1.0)
        else:
            # Discrete placement (original method)
            grid_x_int = np.clip(grid_x_float.astype(int), 0, grid_size - 1)
            grid_y_int = np.clip(grid_y_float.astype(int), 0, grid_size - 1)
            
            if entity_type == 'sheep':
                # Count sheep per cell, normalized by reasonable max density
                for gx, gy in zip(grid_x_int, grid_y_int):
                    obs_grid[channel_idx, gy, gx] += 1.0
                # Normalize sheep channel to ~[0, 1] assuming max ~5 sheep per cell
                obs_grid[channel_idx] = np.clip(obs_grid[channel_idx] / 5.0, 0.0, 1.0)
            else:
                # Binary for enemy/walls
                obs_grid[channel_idx, grid_y_int, grid_x_int] = 1.0
    
    # Mark everything outside walls as walls
    boundary_offset = config.BOUNDARY_OFFSET
    
    # For each grid cell, transform its center position to world coordinates
    # and check if it's outside the playable area
    for gy in range(grid_size):
        for gx in range(grid_size):
            # Grid cell center in agent's rotated frame (relative to agent)
            # Grid is centered at (grid_size/2, grid_size/2), each cell is cell_size wide
            relative_x = (gx + 0.5) * cell_size - observation_range
            relative_y = (gy + 0.5) * cell_size - observation_range
            
            # Rotate from agent frame back to world frame (inverse rotation)
            # Since we rotated by -heading_angle, inverse is +heading_angle
            world_relative_x = relative_x * cos_angle + relative_y * sin_angle
            world_relative_y = -relative_x * sin_angle + relative_y * cos_angle
            
            # Convert to absolute world coordinates
            world_x = agent_position[0] + world_relative_x
            world_y = agent_position[1] + world_relative_y
            
            # Check if outside playable area
            if (world_x < boundary_offset or world_x > screen_width - boundary_offset or
                world_y < boundary_offset or world_y > screen_height - boundary_offset):
                obs_grid[2, gy, gx] = 1  # Mark as wall
    
    # Calculate pen direction vector
    pen_x = pen_entrance_pos[0]
    pen_center_y = (pen_entrance_pos[1] + pen_entrance_pos[2]) / 2
    
    # Find nearest point on pen entrance line segment
    # Pen is a vertical line at x = pen_x, from y = pen_top to y = pen_bottom
    nearest_pen_x = pen_x
    nearest_pen_y = np.clip(agent_position[1], pen_entrance_pos[1], pen_entrance_pos[2])
    
    # Vector from agent to nearest pen point
    to_pen = np.array([nearest_pen_x - agent_position[0], nearest_pen_y - agent_position[1]])
    pen_distance = np.linalg.norm(to_pen)
    
    if pen_distance > 0.001:
        pen_unit_vector = to_pen / pen_distance
    else:
        pen_unit_vector = np.array([1.0, 0.0])
    
    # Rotate pen vector to agent's frame of reference
    # Use -heading_angle rotation (same as entity positions)
    cos_rot = np.cos(-heading_angle)
    sin_rot = np.sin(-heading_angle)
    rotated_pen_x = pen_unit_vector[0] * cos_rot - pen_unit_vector[1] * sin_rot
    rotated_pen_y = pen_unit_vector[0] * sin_rot + pen_unit_vector[1] * cos_rot
    pen_direction_vector = np.array([rotated_pen_x, rotated_pen_y], dtype=np.float32)
    
    return obs_grid, pen_direction_vector


def create_global_observation_grid(agent_position, agent_heading, entities_by_type, grid_size, screen_width, screen_height, pen_entrance_pos):
    """
    Create a global observation grid showing the entire map from a fixed reference frame.
    
    For global observations, we use a simplified representation:
    - 2 channel grid: 
        - Channel 0: sheep positions
        - Channel 1: pen entrance (1s at entrance, 0s elsewhere)
    - 6D metadata vector: agent position, heading, and enemy position
    
    Args:
        agent_position: (x, y) position of the observing agent
        agent_heading: heading angle in radians
        entities_by_type: dict with keys 'enemy', 'sheep', 'walls' containing entity positions
        grid_size: NxN size of the observation grid
        screen_width: width of the screen
        screen_height: height of the screen
        pen_entrance_pos: (x, top_y, bottom_y, width) pen entrance position
    
    Returns:
        tuple: (obs_grid, metadata_vector)
            obs_grid: (2, grid_size, grid_size) observation grid with sheep and pen entrance
            metadata_vector: (6,) [agent_x, agent_y, heading_cos, heading_sin, enemy_x, enemy_y] normalized
    """
    # Initialize observation grid (2 channels: sheep, pen entrance)
    obs_grid = np.zeros((2, grid_size, grid_size), dtype=np.float32)
    
    # Cell size: map entire screen onto grid
    cell_width = screen_width / grid_size
    cell_height = screen_height / grid_size
    
    # Determine splatting method
    splatting_method = config.SPLATTING_METHOD
    
    # Process sheep positions
    if 'sheep' in entities_by_type:
        entities = entities_by_type['sheep']
        if entities is not None and len(entities) > 0:
            # Convert to numpy array
            if not isinstance(entities, np.ndarray):
                entities = np.array(entities, dtype=np.float32)
            
            if len(entities.shape) == 1:
                entities = entities.reshape(-1, 2)
            
            # Convert world coordinates to grid coordinates
            grid_x_float = entities[:, 0] / cell_width
            grid_y_float = entities[:, 1] / cell_height
            
            # Filter entities within bounds
            in_bounds = (grid_x_float >= 0) & (grid_x_float < grid_size) & \
                        (grid_y_float >= 0) & (grid_y_float < grid_size)
            
            if np.any(in_bounds):
                grid_x_float = grid_x_float[in_bounds]
                grid_y_float = grid_y_float[in_bounds]
                
                # Splat sheep onto grid
                if splatting_method == 'gaussian':
                    # Smooth splatting for sheep flock
                    sigma = config.GAUSSIAN_SIGMA
                    radius = config.GAUSSIAN_RADIUS
                    for gx, gy in zip(grid_x_float, grid_y_float):
                        splat_entity_gaussian(obs_grid[0], gx, gy, 1.0, sigma, radius)
                elif splatting_method == 'bilinear':
                    for gx, gy in zip(grid_x_float, grid_y_float):
                        splat_entity_bilinear(obs_grid[0], gx, gy, 1.0)
                else:
                    # Discrete grid (no splatting)
                    grid_x_int = np.clip(grid_x_float.astype(int), 0, grid_size - 1)
                    grid_y_int = np.clip(grid_y_float.astype(int), 0, grid_size - 1)
                    for gx, gy in zip(grid_x_int, grid_y_int):
                        obs_grid[0, gy, gx] += 1.0
    
    # Normalize sheep channel (assuming max ~5 sheep per cell)
    obs_grid[0] = np.clip(obs_grid[0] / 5.0, 0.0, 1.0)
    
    # Create pen entrance channel (channel 1)
    # Mark the pen entrance area with 1s
    pen_x, pen_top_y, pen_bottom_y, pen_width = pen_entrance_pos
    
    # Convert pen entrance to grid coordinates
    pen_grid_x = int(pen_x / (screen_width / grid_size))
    pen_grid_top_y = int(pen_top_y / (screen_height / grid_size))
    pen_grid_bottom_y = int(pen_bottom_y / (screen_height / grid_size))
    
    # Clip to grid bounds
    pen_grid_x = np.clip(pen_grid_x, 0, grid_size - 1)
    pen_grid_top_y = np.clip(pen_grid_top_y, 0, grid_size - 1)
    pen_grid_bottom_y = np.clip(pen_grid_bottom_y, 0, grid_size - 1)
    
    # Mark the pen entrance in the grid (vertical line at pen_grid_x)
    for y in range(pen_grid_top_y, pen_grid_bottom_y + 1):
        obs_grid[1, y, pen_grid_x] = 1.0
    
    # Create metadata vector: [agent_x, agent_y, heading_cos, heading_sin, enemy_x, enemy_y]
    metadata = np.zeros(6, dtype=np.float32)
    
    # Agent position (normalized to [0, 1])
    metadata[0] = agent_position[0] / screen_width
    metadata[1] = agent_position[1] / screen_height
    
    # Agent heading (as cos/sin to avoid angle wrapping issues)
    metadata[2] = np.cos(agent_heading)
    metadata[3] = np.sin(agent_heading)
    
    # Enemy position (normalized to [0, 1])
    if 'enemy' in entities_by_type:
        enemies = entities_by_type['enemy']
        if enemies is not None and len(enemies) > 0:
            enemy_pos = enemies[0] if isinstance(enemies, list) else enemies
            metadata[4] = enemy_pos[0] / screen_width
            metadata[5] = enemy_pos[1] / screen_height
        # If no enemy, leave as [0, 0] (dead/not present)
    
    return obs_grid, metadata


def generate_wall_positions(screen_width, screen_height, boundary_offset, resolution=10):
    """
    Generate positions representing walls (boundaries) of the map.
    
    Args:
        screen_width: width of the screen
        screen_height: height of the screen
        boundary_offset: offset from screen edges
        resolution: spacing between wall points
    
    Returns:
        list: List of (x, y) positions representing walls
    """
    walls = []
    
    # Top wall
    for x in range(boundary_offset, screen_width - boundary_offset, resolution):
        walls.append([x, boundary_offset])
    
    # Bottom wall
    for x in range(boundary_offset, screen_width - boundary_offset, resolution):
        walls.append([x, screen_height - boundary_offset])
    
    # Left wall
    for y in range(boundary_offset, screen_height - boundary_offset, resolution):
        walls.append([boundary_offset, y])
    
    # Right wall
    for y in range(boundary_offset, screen_height - boundary_offset, resolution):
        walls.append([screen_width - boundary_offset, y])
    
    return walls


def render_observation_grid(screen, obs_grid, agent_position, agent_heading, observation_range, alpha=128):
    """
    Render the observation grid as a semi-transparent overlay on the screen.
    
    Args:
        screen: pygame surface to render on
        obs_grid: (3, grid_size, grid_size) observation array
        agent_position: (x, y) position of the agent
        agent_heading: heading angle (radians) describing agent orientation
        observation_range: radius of observation in pixels
        alpha: transparency (0-255)
    """
    grid_size = obs_grid.shape[1]
    cell_size_pixels = (2 * observation_range) / grid_size
    
    heading_angle = float(agent_heading)
    
    # Create surface for the grid
    grid_surface = pygame.Surface((int(2 * observation_range), int(2 * observation_range)), pygame.SRCALPHA)
    
    # Draw grid cells
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_size_pixels
            y = row * cell_size_pixels
            
            # Determine color based on channel values
            enemy_val = obs_grid[0, row, col]
            sheep_val = obs_grid[1, row, col]
            wall_val = obs_grid[2, row, col]
            
            color = None
            if enemy_val > 0:
                color = (255, 0, 0, alpha)  # Red for enemy
            elif sheep_val > 0:
                # Green for sheep, darker with more sheep
                intensity = min(255, int(50 + sheep_val * 500))
                color = (0, intensity, intensity, alpha)
            elif wall_val > 0:
                color = (128, 128, 128, alpha)  # Gray for walls
            
            if color:
                pygame.draw.rect(grid_surface, color, (x, y, cell_size_pixels, cell_size_pixels))
            
            # Draw grid lines
            pygame.draw.rect(grid_surface, (255, 255, 255, 30), (x, y, cell_size_pixels, cell_size_pixels), 1)
    
    # Rotate the grid surface according to agent heading
    # Convert angle from radians to degrees (pygame uses degrees)
    angle_degrees = -np.degrees(heading_angle)
    rotated_surface = pygame.transform.rotate(grid_surface, angle_degrees)
    
    # Calculate position to center the rotated surface on the agent
    rotated_rect = rotated_surface.get_rect(center=(int(agent_position[0]), int(agent_position[1])))
    
    # Blit to screen
    screen.blit(rotated_surface, rotated_rect.topleft)
    
    # Draw a line indicating agent's forward direction
    forward_length = 30
    end_x = agent_position[0] + forward_length * np.cos(heading_angle)
    end_y = agent_position[1] + forward_length * np.sin(heading_angle)
    pygame.draw.line(screen, (255, 255, 0), agent_position, (end_x, end_y), 3)


def render_global_observation_grid(screen, obs_grid, metadata_vector, screen_width, screen_height, alpha=128, label=None):
    """
    Render the global observation grid as a semi-transparent overlay covering the entire screen.
    
    Unlike local observations, this shows the entire map in a fixed world frame.
    For global observations: obs_grid is (2, grid_size, grid_size) with sheep and pen entrance,
    and metadata_vector is (6,) with [agent_x, agent_y, heading_cos, heading_sin, enemy_x, enemy_y].
    
    Args:
        screen: pygame surface to render on
        obs_grid: (2, grid_size, grid_size) global observation array
                  - Channel 0: Sheep positions
                  - Channel 1: Pen entrance markers
        metadata_vector: (6,) [agent_x, agent_y, heading_cos, heading_sin, enemy_x, enemy_y] normalized
        screen_width: width of the screen
        screen_height: height of the screen
        alpha: transparency (0-255)
        label: optional text label to show which agent's observation this is
    """
    grid_size = obs_grid.shape[1]
    cell_width = screen_width / grid_size
    cell_height = screen_height / grid_size
    
    # Create surface for the grid overlay
    grid_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    
    # Draw grid cells
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_width
            y = row * cell_height
            
            # Sheep channel (channel 0)
            sheep_val = obs_grid[0, row, col]
            
            # Pen entrance channel (channel 1)
            pen_val = obs_grid[1, row, col] if obs_grid.shape[0] > 1 else 0
            
            if sheep_val > 0:
                # Green for sheep, intensity based on density
                # sheep_val is normalized (0-1)
                intensity = min(255, int(50 + sheep_val * 500))
                color = (0, intensity, intensity, alpha)
                pygame.draw.rect(grid_surface, color, (x, y, cell_width, cell_height))
            
            if pen_val > 0:
                # Brown for pen entrance (139, 69, 19) with transparency
                # Intensity based on pen_val (0-1)
                intensity = min(1.0, pen_val)
                brown_color = (int(139 * intensity), int(69 * intensity), int(19 * intensity), alpha)
                pygame.draw.rect(grid_surface, brown_color, (x, y, cell_width, cell_height))
            
            # Draw grid lines (very subtle)
            pygame.draw.rect(grid_surface, (255, 255, 255, 15), (x, y, cell_width, cell_height), 1)
    
    # Draw agent and enemy positions from metadata vector
    agent_x_norm, agent_y_norm = metadata_vector[0], metadata_vector[1]
    heading_cos, heading_sin = metadata_vector[2], metadata_vector[3]
    enemy_x_norm, enemy_y_norm = metadata_vector[4], metadata_vector[5]
    
    # Convert normalized positions to screen pixels
    agent_pixel_x = int(agent_x_norm * screen_width)
    agent_pixel_y = int(agent_y_norm * screen_height)
    enemy_pixel_x = int(enemy_x_norm * screen_width)
    enemy_pixel_y = int(enemy_y_norm * screen_height)
    
    # Draw agent position (bright green circle with outline)
    pygame.draw.circle(grid_surface, (0, 255, 0, alpha + 60), (agent_pixel_x, agent_pixel_y), 8)
    pygame.draw.circle(grid_surface, (255, 255, 255, alpha + 60), (agent_pixel_x, agent_pixel_y), 8, 2)
    
    # Draw heading direction arrow
    # Reconstruct heading angle from cos/sin
    heading_angle = np.arctan2(heading_sin, heading_cos)
    arrow_length = 25
    arrow_end_x = int(agent_pixel_x + arrow_length * np.cos(heading_angle))
    arrow_end_y = int(agent_pixel_y + arrow_length * np.sin(heading_angle))
    
    # Draw arrow line
    pygame.draw.line(grid_surface, (255, 255, 0, alpha + 60), 
                     (agent_pixel_x, agent_pixel_y), (arrow_end_x, arrow_end_y), 3)
    
    # Draw arrowhead
    arrow_head_size = 8
    pygame.draw.polygon(grid_surface, (255, 255, 0, alpha + 60), [
        (arrow_end_x, arrow_end_y),
        (int(arrow_end_x - arrow_head_size * np.cos(heading_angle - np.pi/6)),
         int(arrow_end_y - arrow_head_size * np.sin(heading_angle - np.pi/6))),
        (int(arrow_end_x - arrow_head_size * np.cos(heading_angle + np.pi/6)),
         int(arrow_end_y - arrow_head_size * np.sin(heading_angle + np.pi/6)))
    ])
    
    # Draw enemy position (red circle with outline) if present
    if enemy_x_norm > 0.01 or enemy_y_norm > 0.01:  # Assume enemy at origin means dead/absent
        pygame.draw.circle(grid_surface, (255, 0, 0, alpha + 60), (enemy_pixel_x, enemy_pixel_y), 8)
        pygame.draw.circle(grid_surface, (255, 255, 255, alpha + 60), (enemy_pixel_x, enemy_pixel_y), 8, 2)
    
    # Blit to screen
    screen.blit(grid_surface, (0, 0))
    
    # Draw label if provided
    if label:
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, (255, 255, 0))
        text_rect = text.get_rect(topleft=(10, 10))
        # Draw background for readability
        bg_rect = text_rect.inflate(10, 5)
        pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)
        screen.blit(text, text_rect)
