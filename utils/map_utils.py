import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def create_green_background_image(height, width, texture_strength=3, seed=42):
    """
    Create a green background image with patchy texture, similar to the grass in create_stylized_track_image.
    
    Parameters:
    - height: int, image height in pixels
    - width: int, image width in pixels
    - texture_strength: int, strength of texture noise (default: 3)
    - seed: int, random seed for reproducible textures (default: 42)
    
    Returns:
    - PIL Image in RGB format
    """
    
    # Colors (same as in create_stylized_track_image)
    grass = np.array([81, 115, 49], dtype=np.uint8)    # green field
    
    # Create base canvas filled with grass color
    canvas = np.tile(grass, (height, width, 1))
    
    # Add patchy texture for realism (same as grass texture in original function)
    def generate_patchy_noise(shape, strength, patch_size=8, contrast=2.0, seed=42):
        """Generate patchy, sharp-looking noise with clusters"""
        rng = np.random.default_rng(seed)
        H, W = shape[:2]
        
        # Create base noise at lower resolution for patches
        low_res_h, low_res_w = H // patch_size, W // patch_size
        if low_res_h == 0 or low_res_w == 0:
            low_res_h = max(1, low_res_h)
            low_res_w = max(1, low_res_w)
        base_noise = rng.normal(0, strength, (low_res_h, low_res_w))
        
        # Upsample using nearest neighbor to create sharp patches
        patchy_noise = zoom(base_noise, patch_size, order=0)  # order=0 = nearest neighbor
        
        # Crop or pad to exact size
        if patchy_noise.shape[0] > H:
            patchy_noise = patchy_noise[:H, :]
        if patchy_noise.shape[1] > W:
            patchy_noise = patchy_noise[:, :W]
        if patchy_noise.shape[0] < H or patchy_noise.shape[1] < W:
            padded = np.zeros((H, W))
            padded[:patchy_noise.shape[0], :patchy_noise.shape[1]] = patchy_noise
            patchy_noise = padded
        
        # Add very subtle high-frequency detail noise
        detail_noise = rng.integers(-strength//4, strength//4 + 1, size=(H, W), dtype=np.int16)
        
        # Combine for final patchy texture
        final_noise = (patchy_noise * contrast + detail_noise).astype(np.int16)
        
        # Expand to 3 channels
        return final_noise[..., np.newaxis]
    
    # Generate grass noise (same parameters as in original)
    grass_noise = generate_patchy_noise((height, width), texture_strength * 1.5, patch_size=2, contrast=1.0, seed=seed)
    
    # Apply noise to the entire canvas
    canvas = canvas.astype(np.int16)
    canvas = np.clip(canvas + grass_noise, 0, 255)
    canvas = canvas.astype(np.uint8)
    
    # Convert to PIL Image
    background_image = Image.fromarray(canvas, mode="RGB")
    
    return background_image