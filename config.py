"""
Configuration file for the sheep herding simulation.
All simulation parameters are defined here for easy iteration and testing.
"""

# ============================================================================
# DEBUG AND VISUALIZATION
# ============================================================================

# Debug mode - show agent observation grids
DEBUG_SHOW_OBSERVATIONS = False  # Set to True to visualize agent observation grids

# Show pen direction vectors on agents
DEBUG_SHOW_PEN_DIRECTION = False  # Set to True to draw pen direction arrows on agents

# ============================================================================
# OBSERVATION GRID PARAMETERS
# ============================================================================

# Observation mode: 'local' for ego-centric grid, 'global' for full map view
USE_GLOBAL_STATE_OBSERVATION = False  # Set to True for global observation mode

# Observation grid resolution (NxN grid)
OBSERVATION_GRID_SIZE = 20  # 20x20 grid for both local and global modes

# Observation range in pixels (radius around agent) - only used in local mode
OBSERVATION_RANGE = 300  # How far the agent can see (in pixels)

# Number of observation channels (depends on mode)
# Local mode: 3 channels [enemy, sheep, walls] + 2D pen_vector
# Global mode: 2 channels [sheep, pen_entrance] + 6D metadata [agent_x, agent_y, heading_cos, heading_sin, enemy_x, enemy_y]
OBSERVATION_CHANNELS = 2 if USE_GLOBAL_STATE_OBSERVATION else 3
METADATA_DIM = 6 if USE_GLOBAL_STATE_OBSERVATION else 2  # metadata vector size

# Splatting method for rendering entities in observation grid
# Options: None, 'gaussian', 'bilinear'
SPLATTING_METHOD = 'gaussian'  # Use None for discrete grid (no splatting)

# Gaussian splatting parameters (only used if SPLATTING_METHOD = 'gaussian')
GAUSSIAN_SIGMA = 0.5  # Standard deviation in grid cells
GAUSSIAN_RADIUS = 2  # Truncation radius in grid cells (how far to splat)

# ============================================================================
# SCREEN AND MAP PARAMETERS
# ============================================================================

# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

PADDING = 15

# Map boundary offset (distance from screen edge to actual playable boundary)
# This prevents entities from appearing half outside the screen
BOUNDARY_OFFSET = 50  # pixels from edge

# ============================================================================
# ENTITY SPAWN PARAMETERS
# ============================================================================

# Number of sheep to spawn
NUM_SHEEP = 25

# Sheep spawn area (center of map)
SHEEP_SPAWN_X_MIN = 300
SHEEP_SPAWN_X_MAX = 700
SHEEP_SPAWN_Y_MIN = 300
SHEEP_SPAWN_Y_MAX = 700

# Dog spawn position (bottom left area)
DOG_SPAWN_X = 200
DOG_SPAWN_Y = 800

# Wolf spawn position (left side of map)
# Wolf always spawns on the left side, both initially and after respawn
WOLF_SPAWN_X_MIN = 50
WOLF_SPAWN_X_MAX = 200
WOLF_SPAWN_Y_MIN = 200
WOLF_SPAWN_Y_MAX = 800

# ============================================================================
# ENTITY MOVEMENT PARAMETERS
# ============================================================================

# Sheep parameters
SHEEP_MAX_SPEED = 1.0
SHEEP_MAX_TURN_RATE = 0.2
SHEEP_VELOCITY_SMOOTHING = 0.4

# Dog parameters
DOG_MAX_SPEED = 4.0
DOG_MAX_TURN_RATE = 0.2
DOG_VELOCITY_SMOOTHING = 0.4
DOG_THREAT_WEIGHT = 1.0

# Wolf parameters
WOLF_MAX_SPEED = 4.5
WOLF_MAX_TURN_RATE = 0.2
WOLF_VELOCITY_SMOOTHING = 0.4
WOLF_THREAT_WEIGHT = 3.0

# ============================================================================
# BOIDS ALGORITHM PARAMETERS
# ============================================================================

# Boids weight parameters
BOIDS_W_SEPARATION = 4.0
BOIDS_W_ALIGNMENT = 0.005
BOIDS_W_COHESION = 0.0001
BOIDS_W_PREDATOR = 2.5
BOIDS_W_WOLF_THREAT = 2.0
BOIDS_W_DOG_THREAT = 3.0
BOIDS_COMPRESSION_FACTOR = 5.0

# Boids radius parameters[]
SHEEP_SEPARATION_RADIUS = 30
SHEEP_ALIGNMENT_RADIUS = 250
SHEEP_COHESION_RADIUS = 200
SHEEP_DETECTION_RADIUS = 150

# Sheep panic parameters
SHEEP_PANIC_RADIUS_WOLF = 150  # Panic distance for wolf
SHEEP_PANIC_RADIUS_DOG = 200   # Panic distance for dog
SHEEP_FIELD_OF_VIEW = 270  # degrees

# ============================================================================
# BOUNDARY PARAMETERS
# ============================================================================

# Boundary repulsion force parameters
BOUNDARY_MARGIN = 50
BOUNDARY_STRENGTH = 0.5

# ============================================================================
# PEN PARAMETERS
# ============================================================================

# Fence visual parameters
FENCE_THICKNESS = 8
FENCE_COLOR = (101, 67, 33)  # Brown fence color

# Pen entrance parameters (on the right side of the map)
PEN_ENTRANCE_WIDTH_RATIO = 0.5  # Ratio of screen height (0.5 = half of screen height)
PEN_ENTRANCE_CENTER_RATIO = 0.5  # Center position ratio (0.5 = centered vertically)
PEN_DETECTION_DISTANCE = 30  # Distance from entrance to trigger sheep disappearance

# ============================================================================
# COMBAT PARAMETERS
# ============================================================================

# Wolf eating sheep parameters
WOLF_EAT_DISTANCE = 30  # Distance at which wolf can eat sheep (pixels)
WOLF_EAT_COOLDOWN_STEPS = 180  # Cooldown after eating sheep (steps, ~3 seconds at 60 FPS)

# Dog killing wolf parameters
DOG_KILL_WOLF_DISTANCE = 40  # Distance at which dog can kill wolf (pixels)
WOLF_RESPAWN_COOLDOWN_STEPS = 150  # Cooldown before wolf respawns after being killed (steps, ~2 seconds at 60 FPS)

# ============================================================================
# RENDERING PARAMETERS
# ============================================================================

# Sprite size parameters
SPRITE_MAX_DIMENSION = 100  # Max dimension for sprites in pixels
SHEEP_SPRITE_SCALE = 0.6  # Additional scale factor for sheep (smaller than dog/wolf)

# Background texture strength
BACKGROUND_TEXTURE_STRENGTH = 3

# Default FPS
DEFAULT_FPS = 60
