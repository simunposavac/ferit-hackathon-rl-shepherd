import numpy as np
import pygame
from sim.sheep import Sheep
from sim.dog import Dog
from sim.wolf import Wolf
import config

def draw_tiled_texture(screen, texture, rect, density=1.0, angle=0.0, pixel_art=False, offset=(0, 0)):
    """
    Draw a tiled texture inside rect=(x, y, w, h), scaled by density and rotated by angle (degrees).
    - Higher density = smaller tiles (more repeats)
    - Lower density  = larger tiles (fewer repeats)
    """
    x, y, w, h = rect

    # Clip to keep drawing inside the fence area
    prev_clip = screen.get_clip()
    screen.set_clip(pygame.Rect(x, y, w, h))

    # Scale + rotate the texture
    tw, th = texture.get_size()
    scale = 1.0 / max(density, 1e-6)
    tw, th = max(1, int(tw * scale)), max(1, int(th * scale))
    transform = pygame.transform.scale if pixel_art else pygame.transform.smoothscale
    tile = transform(texture, (tw, th))
    if angle:
        tile = pygame.transform.rotate(tile, angle % 360)
    tw, th = tile.get_size()

    # Tile it across the rect
    ox, oy = offset
    start_x = x - ((x + int(ox)) % tw)
    start_y = y - ((y + int(oy)) % th)
    bottom, right = y + h, x + w

    ty = start_y
    while ty < bottom:
        tx = start_x
        while tx < right:
            screen.blit(tile, (tx, ty))
            tx += tw
        ty += th

    # Restore clip
    screen.set_clip(prev_clip)

def get_centered_rect(x, y, width, height):
    """
    Return a pygame.Rect whose center is (x, y)
    and whose size is (width, height).
    """
    left = x - width / 2
    top = y - height / 2
    return pygame.Rect(left, top, width, height)

class Environment:
    """
    Main environment class that manages the simulation and implements
    the boids flocking algorithm with predator extension.
    """
    
    def __init__(self, width=None, height=None, background_surface=None):
        """
        Initialize the environment.
        
        Args:
            width: int, width of the simulation area (uses config if None)
            height: int, height of the simulation area (uses config if None)
            background_surface: pygame.Surface, background image
        """
        self.width = width if width is not None else config.SCREEN_WIDTH
        self.height = height if height is not None else config.SCREEN_HEIGHT
        self.background = background_surface
        
        # Entity containers
        self.sheep_list = []
        self.dog = None
        self.wolf = None
        
        # Boids weight parameters from config
        self.w_separation = config.BOIDS_W_SEPARATION
        self.w_alignment = config.BOIDS_W_ALIGNMENT
        self.w_cohesion = config.BOIDS_W_COHESION
        self.w_predator = config.BOIDS_W_PREDATOR
        self.w_wolf_threat = config.BOIDS_W_WOLF_THREAT
        self.w_dog_threat = config.BOIDS_W_DOG_THREAT
        self.compression_factor = config.BOIDS_COMPRESSION_FACTOR
        
        # Boundary repulsion parameters from config
        self.boundary_margin = config.BOUNDARY_MARGIN
        self.boundary_strength = config.BOUNDARY_STRENGTH
        self.boundary_offset = config.BOUNDARY_OFFSET
        
        # Pen parameters from config
        self.fence_thickness = config.FENCE_THICKNESS
        self.fence_color = config.FENCE_COLOR
        self.pen_entrance_width = int(height * config.PEN_ENTRANCE_WIDTH_RATIO)
        pen_entrance_center_y = height * config.PEN_ENTRANCE_CENTER_RATIO
        self.pen_entrance_top = int(pen_entrance_center_y - self.pen_entrance_width // 2)
        self.pen_entrance_bottom = int(pen_entrance_center_y + self.pen_entrance_width // 2)
        self.pen_entrance_x = width - 20  # Right edge
        self.pen_detection_distance = config.PEN_DETECTION_DISTANCE
        
        # Stats
        self.sheep_in_pen = 0
        self.sheep_eaten = 0
    
    def increment_eaten(self):
        if self.sheep_eaten is not None:
            self.sheep_eaten += 1

    def reset_eaten(self):
        self.sheep_eaten = 0
        
    def add_sheep(self, sheep):
        """Add a sheep to the environment."""
        self.sheep_list.append(sheep)
    
    def set_dog(self, dog):
        """Set the dog entity."""
        self.dog = dog
    
    def set_wolf(self, wolf):
        """Set the wolf entity."""
        self.wolf = wolf
    
    def calculate_separation(self, sheep):
        """
        Calculate separation force for a sheep (avoid crowding).
        
        Args:
            sheep: Sheep object
            
        Returns:
            np.array, separation velocity vector
        """
        separation_force = np.array([0.0, 0.0])
        neighbors = sheep.get_neighbors(self.sheep_list, sheep.separation_radius)
        
        for neighbor in neighbors:
            to_neighbor = neighbor.position - sheep.position
            distance = np.linalg.norm(to_neighbor)
            
            if distance > 0:
                # Repulsion weighted by inverse distance
                repulsion = -to_neighbor / distance
                separation_force += repulsion
        
        return separation_force
    
    def calculate_alignment(self, sheep):
        """
        Calculate alignment force for a sheep (match velocity).
        
        Args:
            sheep: Sheep object
            
        Returns:
            np.array, alignment velocity vector
        """
        neighbors = sheep.get_neighbors(self.sheep_list, sheep.alignment_radius)
        
        if len(neighbors) == 0:
            return np.array([0.0, 0.0])
        
        # Average velocity of neighbors
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        alignment_force = avg_velocity - sheep.velocity
        
        return alignment_force
    
    def calculate_cohesion(self, sheep):
        """
        Calculate cohesion force for a sheep (move toward flock center).
        
        Args:
            sheep: Sheep object
            
        Returns:
            np.array, cohesion velocity vector
        """
        neighbors = sheep.get_neighbors(self.sheep_list, sheep.cohesion_radius)
        
        if len(neighbors) == 0:
            return np.array([0.0, 0.0])
        
        # Center of mass of neighbors
        center_of_mass = np.mean([n.position for n in neighbors], axis=0)
        cohesion_force = center_of_mass - sheep.position
        
        return cohesion_force
    
    def calculate_predator_avoidance(self, sheep):
        """
        Calculate predator avoidance force for a sheep.
        
        Args:
            sheep: Sheep object
            
        Returns:
            np.array, predator avoidance velocity vector
        """
        avoidance_force = np.array([0.0, 0.0])
        detected_predators = []
        
        # Check dog
        if self.dog is not None:
            if sheep.can_detect_predator(self.dog.position):
                detected_predators.append((self.dog, self.w_dog_threat))
        
        # Check wolf
        if self.wolf is not None:
            if sheep.can_detect_predator(self.wolf.position):
                detected_predators.append((self.wolf, self.w_wolf_threat))
        
        # Calculate avoidance for each detected predator
        for predator, threat_weight in detected_predators:
            to_predator = predator.position - sheep.position
            distance = np.linalg.norm(to_predator)
            
            if distance > 0:
                # Escape vector weighted by inverse square distance and threat level
                escape = -to_predator / (distance ** 2)
                avoidance_force += threat_weight * escape
        
        return avoidance_force
    
    def check_panic_mode(self, sheep):
        """
        Check if sheep should enter panic mode with separate panic distances for dog and wolf.
        
        Args:
            sheep: Sheep object
            
        Returns:
            tuple: (bool panic_mode, np.array panic_direction or None)
        """
        min_distance = float('inf')
        nearest_predator = None
        panic_threshold = float('inf')
        
        # Check dog with dog-specific panic radius
        if self.dog is not None:
            distance = np.linalg.norm(self.dog.position - sheep.position)
            if distance < sheep.panic_radius_dog and distance < min_distance:
                min_distance = distance
                nearest_predator = self.dog
                panic_threshold = sheep.panic_radius_dog
        
        # Check wolf with wolf-specific panic radius
        if self.wolf is not None:
            distance = np.linalg.norm(self.wolf.position - sheep.position)
            if distance < sheep.panic_radius_wolf and distance < min_distance:
                min_distance = distance
                nearest_predator = self.wolf
                panic_threshold = sheep.panic_radius_wolf
        
        # Enter panic if predator is within its specific panic radius
        if nearest_predator is not None and min_distance < panic_threshold:
            to_predator = nearest_predator.position - sheep.position
            distance = np.linalg.norm(to_predator)
            if distance > 0:
                panic_direction = -to_predator / distance
                return True, panic_direction
        
        return False, None
    
    def calculate_modified_cohesion_weight(self, sheep):
        """
        Calculate modified cohesion weight when predators are present.
        Flock compresses when under threat.
        
        Args:
            sheep: Sheep object
            
        Returns:
            float, modified cohesion weight
        """
        num_predators = 0
        min_predator_distance = float('inf')
        
        if self.dog is not None:
            num_predators += 1
            distance = np.linalg.norm(self.dog.position - sheep.position)
            min_predator_distance = min(min_predator_distance, distance)
        
        if self.wolf is not None:
            num_predators += 1
            distance = np.linalg.norm(self.wolf.position - sheep.position)
            min_predator_distance = min(min_predator_distance, distance)
        
        if num_predators == 0:
            return self.w_cohesion
        
        # Modified cohesion formula
        compression = 1 + self.compression_factor * num_predators / (1 + min_predator_distance)
        return self.w_cohesion * compression
    
    def calculate_boundary_force(self, position):
        """
        Calculate force to keep entities away from boundaries.
        
        Args:
            position: np.array, current position
            
        Returns:
            np.array, boundary repulsion force
        """
        force = np.array([0.0, 0.0])
        
        # Left boundary
        if position[0] < self.boundary_margin:
            force[0] += self.boundary_strength * (self.boundary_margin - position[0])
        
        # Right boundary
        if position[0] > self.width - self.boundary_margin:
            force[0] -= self.boundary_strength * (position[0] - (self.width - self.boundary_margin))
        
        # Top boundary
        if position[1] < self.boundary_margin:
            force[1] += self.boundary_strength * (self.boundary_margin - position[1])
        
        # Bottom boundary
        if position[1] > self.height - self.boundary_margin:
            force[1] -= self.boundary_strength * (position[1] - (self.height - self.boundary_margin))
        
        return force
    
    def update_sheep(self, dt=1.0):
        """
        Update all sheep positions using boids algorithm.
        
        Args:
            dt: float, time step
        """
        for sheep in self.sheep_list:
            # Check for panic mode
            panic_mode, panic_direction = self.check_panic_mode(sheep)
            sheep.panic_mode = panic_mode
            
            if panic_mode:
                # Panic: flee at max speed directly away from nearest predator
                new_velocity = panic_direction * sheep.max_speed
            else:
                # Normal boids behavior with predator extension
                
                # Calculate individual forces
                separation = self.calculate_separation(sheep)
                alignment = self.calculate_alignment(sheep)
                cohesion = self.calculate_cohesion(sheep)
                predator_avoidance = self.calculate_predator_avoidance(sheep)
                boundary = self.calculate_boundary_force(sheep.position)
                
                # Get modified cohesion weight
                w_cohesion_mod = self.calculate_modified_cohesion_weight(sheep)
                
                # Combine forces with weights
                new_velocity = (
                    sheep.velocity +
                    self.w_separation * separation +
                    self.w_alignment * alignment +
                    w_cohesion_mod * cohesion +
                    self.w_predator * predator_avoidance +
                    boundary
                )
            
            # Update position with velocity limiting
            sheep.update_position(new_velocity, dt)
            
            # Enforce boundary constraints
            self.enforce_boundaries(sheep)
    
    def enforce_boundaries(self, entity):
        """
        Enforce hard boundaries - prevent entities from going outside map edges.
        Entities collide with edges and their velocity is reflected.
        Uses BOUNDARY_OFFSET to keep entities fully inside the visible area.
        
        Args:
            entity: Any entity with position and velocity attributes
        """
        # Left boundary (with offset)
        if entity.position[0] < self.boundary_offset:
            entity.position[0] = self.boundary_offset
            entity.velocity[0] = abs(entity.velocity[0]) * 0.5  # Bounce back with damping
        
        # Top boundary (with offset)
        if entity.position[1] < self.boundary_offset:
            entity.position[1] = self.boundary_offset
            entity.velocity[1] = abs(entity.velocity[1]) * 0.5  # Bounce back with damping
        
        # Bottom boundary (with offset)
        if entity.position[1] > self.height - self.boundary_offset:
            entity.position[1] = self.height - self.boundary_offset
            entity.velocity[1] = -abs(entity.velocity[1]) * 0.5  # Bounce back with damping

        # Right boundary (with offset)
        if type(entity) is Dog or type(entity) is Wolf:
            if entity.position[0] > self.width - self.boundary_offset - 2 * config.PADDING:
                entity.position[0] = self.width - self.boundary_offset - 2 * config.PADDING
                entity.velocity[0] = -abs(entity.velocity[0]) * 0.5  # Bounce back with damping
        else:
            if entity.position[0] > self.width - self.boundary_offset:
                entity.position[0] = self.width - self.boundary_offset
                entity.velocity[0] = -abs(entity.velocity[0]) * 0.5  # Bounce back with damping
    
    def update(self, dt=1.0):
        """
        Update the entire simulation.
        
        Args:
            dt: float, time step
            
        Returns:
            list: IDs of sheep that entered the pen this step
        """
        # Update sheep with boids algorithm
        self.update_sheep(dt)
        
        # Check for sheep entering the pen
        sheep_entered_ids = self.check_sheep_in_pen()
        
        return sheep_entered_ids
    
    def check_sheep_in_pen(self):
        """
        Check if any sheep have entered the pen and remove them.
        
        Returns:
            list: IDs of sheep that entered the pen this step
        """
        sheep_to_remove = []
        sheep_entered_ids = []
        
        for sheep in self.sheep_list:
            # Check if sheep is near the pen entrance on the right side
            if sheep.position[0] >= self.pen_entrance_x - self.pen_detection_distance:
                # Check if within vertical range of entrance
                if self.pen_entrance_top <= sheep.position[1] <= self.pen_entrance_bottom:
                    sheep_to_remove.append(sheep)
                    sheep_entered_ids.append(sheep.id)
                    self.sheep_in_pen += 1
        
        # Remove sheep that entered the pen
        for sheep in sheep_to_remove:
            self.sheep_list.remove(sheep)
        
        return sheep_entered_ids
    
    def render(self, screen, wolf_in_cooldown=False, wolf_cooldown_ratio=0.0):
        """
        Render all entities in the environment.
        
        Args:
            screen: pygame.Surface
            wolf_in_cooldown: bool, whether wolf is in cooldown
            wolf_cooldown_ratio: float (0-1), cooldown remaining ratio
        """
        # Draw background
        if self.background is not None:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill((81, 115, 49))  # Green grass color
        
        # Draw fence around the edges (except pen entrance)
        self.draw_fence(screen)
        
        # Render all entities
        for sheep in self.sheep_list:
            sheep.render(screen)
        
        if self.dog is not None:
            self.dog.render(screen)
        
        if self.wolf is not None:
            self.wolf.render(screen, wolf_in_cooldown, wolf_cooldown_ratio)
        
        # Draw stats
        self.draw_stats(screen)
    
    def draw_fence(self, screen):
        """
        Draw fence around the map edges with a pen entrance on the right side.
        
        Args:
            screen: pygame.Surface
        """

        self.fence_texture = pygame.image.load("images/wood.jpg").convert_alpha()

        padding = config.PADDING

        # Top fence
        top_rect = (self.width - padding, padding, self.width, self.fence_thickness)
        draw_tiled_texture(screen, self.fence_texture, top_rect, density=5)

        # Bottom fence
        bottom_rect = (self.width - padding, self.height - self.fence_thickness - padding, self.width, self.fence_thickness)
        draw_tiled_texture(screen, self.fence_texture, bottom_rect, density=5)
        
        # Right fence - split into two parts with gap for pen entrance
        # Upper part
        right_upper_rect = (self.width - self.fence_thickness - padding, 0 + padding, self.fence_thickness, self.pen_entrance_top)
        draw_tiled_texture(screen, self.fence_texture, right_upper_rect, density=5, angle=90)
        # Lower part
        right_lower_rect = (self.width - self.fence_thickness - padding , self.pen_entrance_bottom - padding, self.fence_thickness, self.height - self.pen_entrance_bottom)
        draw_tiled_texture(screen, self.fence_texture, right_lower_rect, density=5, angle=90)
        
        # Draw pen entrance markers (posts)
        small_post_width = 15
        post_color = (80, 50, 20)  # Darker brown for posts
        
        # Top fence posts
        top_post_rect1 = get_centered_rect(self.width - padding - self.fence_thickness // 2, self.pen_entrance_top + padding + self.fence_thickness // 2, small_post_width, small_post_width)
        draw_tiled_texture(screen, self.fence_texture, top_post_rect1, density=5)
        top_post_rect2 = get_centered_rect(self.width - self.fence_thickness//2 - padding, (padding + self.pen_entrance_top)//2 + self.fence_thickness, small_post_width, small_post_width)
        draw_tiled_texture(screen, self.fence_texture, top_post_rect2, density=5)
        top_post_rect3 = get_centered_rect(self.width - self.fence_thickness//2 - padding, 0 + self.fence_thickness//2 + padding, small_post_width, small_post_width)
        draw_tiled_texture(screen, self.fence_texture, top_post_rect3, density=5)
        
        # Bottom fence posts
        bottom_post_rect1 = get_centered_rect(self.width - padding - self.fence_thickness // 2, self.pen_entrance_bottom - padding - self.fence_thickness // 2, small_post_width, small_post_width)
        draw_tiled_texture(screen, self.fence_texture, bottom_post_rect1, density=5)
        bottom_post_rect2 = get_centered_rect(self.width - self.fence_thickness//2 - padding, self.height - (padding + self.pen_entrance_top)//2 - self.fence_thickness, small_post_width, small_post_width)
        draw_tiled_texture(screen, self.fence_texture, bottom_post_rect2, density=5)
        bottom_post_rect3 = get_centered_rect(self.width - self.fence_thickness//2 - padding, self.height - self.fence_thickness//2 - padding, small_post_width, small_post_width)
        draw_tiled_texture(screen, self.fence_texture, bottom_post_rect3, density=5)
        

    def draw_stats(self, screen):
        """
        Draw statistics on the screen.
        
        Args:
            screen: pygame.Surface
        """
        font = pygame.font.Font(None, 25)
        
        # Draw sheep count
        sheep_text = font.render(f"Total: {len(self.sheep_list)}", True, (255, 255, 255))
        screen.blit(sheep_text, (30, 30))
        
        # Draw sheep in pen count
        pen_text = font.render(f"In Pen: {self.sheep_in_pen}", True, (255, 255, 255))
        screen.blit(pen_text, (30, 60))

        # Draw sheep in pen count
        pen_text = font.render(f"Eaten: {self.sheep_eaten}", True, (255, 255, 255))
        screen.blit(pen_text, (30, 90))
