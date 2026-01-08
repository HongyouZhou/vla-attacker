import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import math
from typing import Optional, Tuple


# Helper function: Generate uniformly distributed random numbers within a range
def uniform_range(minval: float, maxval: float, shape: Tuple = (), epsilon: float = 1e-8):
    """Generate uniformly distributed random numbers within a specified range.

    Args:
        minval: The minimum value.
        maxval: The maximum value.
        shape: The output shape.
        epsilon: A numerical stability constant to prevent division by zero.
    """
    safe_range = max(maxval - minval, epsilon)
    adjusted_maxval = minval + safe_range

    if shape == ():
        return minval + (adjusted_maxval - minval) * torch.rand(1).item()
    else:
        return minval + (adjusted_maxval - minval) * torch.rand(shape)


# Helper function: Generate integer random numbers within a range
def randint_range(minval: int, maxval: int, shape: Tuple = ()):
    """Generate integer random numbers within a specified range."""
    if shape == ():
        return torch.randint(minval, maxval, (1,)).item()
    else:
        return torch.randint(minval, maxval, shape)


# Custom RGB to HSV function
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB colors to the HSV color space."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Calculate the maximum and minimum values for each pixel
    max_val, _ = torch.max(rgb, dim=-1)
    min_val, _ = torch.min(rgb, dim=-1)
    delta = max_val - min_val

    # Initialize hue H
    h = torch.zeros_like(max_val)

    delta_safe = torch.where(delta < 1e-8, torch.ones_like(delta), delta)

    # Calculate hue based on the max channel
    cond_r = (max_val == r) & (delta > 1e-8)
    cond_g = (max_val == g) & (delta > 1e-8)
    cond_b = (max_val == b) & (delta > 1e-8)

    h = torch.where(
        cond_r,
        ((g - b) / delta_safe) % 6,
        h
    )

    h = torch.where(
        cond_g,
        2.0 + (b - r) / delta_safe,
        h
    )

    h = torch.where(
        cond_b,
        4.0 + (r - g) / delta_safe,
        h
    )

    # Normalize hue to the 0-360 degree range
    h = (h * 60) % 360

    # Calculate saturation
    s = torch.where(max_val > 1e-8, delta_safe / max_val, torch.zeros_like(max_val))

    # Value is the maximum value
    v = max_val

    return torch.stack([h, s, v], dim=-1)


# Custom HSV to RGB function
def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV colors to the RGB color space."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Map hue to the 0-360 range
    h = h % 360

    # Calculate intermediate variables
    c = v * s
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v - c

    # Determine RGB values based on the hue range
    cond0 = h < 60
    cond60 = (h >= 60) & (h < 120)
    cond120 = (h >= 120) & (h < 180)
    cond180 = (h >= 180) & (h < 240)
    cond240 = (h >= 240) & (h < 300)
    cond300 = h >= 300

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    r = torch.where(cond0 | cond300, c, r)
    r = torch.where(cond60 | cond180, x, r)
    r = torch.where(cond120 | cond240, torch.zeros_like(r), r)

    g = torch.where(cond0 | cond180, x, g)
    g = torch.where(cond60 | cond120, c, g)
    g = torch.where(cond240 | cond300, torch.zeros_like(g), g)

    b = torch.where(cond0 | cond120, torch.zeros_like(b), b)
    b = torch.where(cond60 | cond240, x, b)
    b = torch.where(cond180 | cond300, c, b)

    # Add the value adjustment
    r += m
    g += m
    b += m

    # Clamp to the 0-1 range
    return torch.clamp(torch.stack([r, g, b], dim=-1), 0, 1)


# Image augmentation functions - all functions correctly handle *b h w c shape
def color_jittering(image: torch.Tensor) -> torch.Tensor:
    """Randomly adjust the brightness, contrast, saturation, and hue of the image."""

    # Brightness adjustment
    brightness_factor = uniform_range(0.8, 1.2)
    image = image * brightness_factor

    # Contrast adjustment
    contrast_factor = uniform_range(0.5, 1.5)
    # Calculate the mean of each image (keeping the batch dimension)
    mean = torch.mean(image, dim=tuple(range(image.ndim - 3)), keepdim=True)
    image = contrast_factor * (image - mean) + mean

    # Convert to HSV space
    hsv_img = rgb_to_hsv(image)

    # Saturation adjustment
    saturation_factor = uniform_range(0.7, 1.3)
    hsv_img = hsv_img.clone()
    hsv_img[..., 1] = torch.clamp(hsv_img[..., 1] * saturation_factor, 0, 1)

    # Hue adjustment
    hue_factor = uniform_range(-0.1, 0.1)
    hsv_img[..., 0] = hsv_img[..., 0] + hue_factor * 180

    # Convert back to RGB space
    return torch.clamp(hsv_to_rgb(hsv_img), 0, 1)


def light_intensity_adjustment(image: torch.Tensor) -> torch.Tensor:
    """Randomly adjust the light intensity."""
    epsilon = 1e-8
    factor = uniform_range(0.5, 2.0, epsilon=epsilon)

    if math.isnan(factor) or math.isinf(factor):
        factor = 1.0  # Use default value if factor is abnormal

    result = torch.clamp(image * factor, 0, 1)

    if torch.isnan(result).any() or torch.isinf(result).any():
        return image

    return result


def simulate_light_direction(image: torch.Tensor) -> torch.Tensor:
    """Simulate light direction - compatible with *b h w c format."""
    # Get spatial dimensions (height and width)
    height, width = image.shape[-3], image.shape[-2]
    device = image.device
    epsilon = 1e-8

    # Create grid coordinates
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )

    # Random light source position
    center_x = uniform_range(width//4, width*3//4, epsilon=epsilon)
    center_y = uniform_range(height//4, height*3//4, epsilon=epsilon)

    # Calculate distance matrix
    distances = torch.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2 + epsilon)

    # Create light mask
    max_dist = torch.max(distances) + epsilon
    light_mask = 1.0 - torch.clamp(distances / max_dist, 0, 1)

    if torch.isnan(light_mask).any() or torch.isinf(light_mask).any():
        light_mask = torch.ones_like(light_mask) * 0.8

    # Adjust light mask shape to match the image
    light_mask = light_mask.unsqueeze(-1)  # Add channel dimension -> (height, width, 1)

    # Add dimensions for batch
    for _ in range(image.ndim - 3):
        light_mask = light_mask.unsqueeze(0)  # Add batch dimension -> (*1, height, width, 1)

    # Apply light effect
    result = torch.clamp(image * light_mask, 0, 1)

    if torch.isnan(result).any() or torch.isinf(result).any():
        return image

    return result


def add_shadow(image: torch.Tensor) -> torch.Tensor:
    """Add a shadow to the image - compatible with *b h w c format."""
    # Get spatial dimensions
    height, width = image.shape[-3], image.shape[-2]
    device = image.device

    # Random shadow position and size
    shadow_x = uniform_range(0, width//2)
    shadow_y = uniform_range(0, height//2)
    shadow_width = uniform_range(width//4, width//2)
    shadow_height = uniform_range(height//4, height//2)

    # Shadow intensity
    shadow_intensity = uniform_range(0.3, 0.7)

    # Create shadow mask
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    in_shadow = (x_indices >= shadow_x) & (x_indices < shadow_x + shadow_width) & \
                (y_indices >= shadow_y) & (y_indices < shadow_y + shadow_height)

    # Adjust shadow mask shape to match the image
    shadow_mask = torch.where(in_shadow, shadow_intensity, 1.0)
    shadow_mask = shadow_mask.unsqueeze(-1)  # Add channel dimension -> (height, width, 1)

    # Add dimensions for batch
    for _ in range(image.ndim - 3):
        shadow_mask = shadow_mask.unsqueeze(0)  # Add batch dimension -> (*1, height, width, 1)

    return torch.clamp(image * shadow_mask, 0, 1)


def random_occlusion(image: torch.Tensor) -> torch.Tensor:
    """Randomly occlude a part of the image - compatible with *b h w c format."""
    # Get spatial dimensions
    height, width = image.shape[-3], image.shape[-2]
    device = image.device

    # Random occlusion position and size
    occlusion_width = uniform_range(width//10, width//4)
    occlusion_height = uniform_range(height//10, height//4)
    x = uniform_range(0, float(width - occlusion_width))
    y = uniform_range(0, float(height - occlusion_height))

    # Random occlusion color
    occlusion_color = torch.rand(3, device=device)

    # Create occlusion mask
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    occluded = (x_indices >= x) & (x_indices < x + occlusion_width) & \
               (y_indices >= y) & (y_indices < y + occlusion_height)

    # Adjust occlusion mask shape to match the image
    occluded = occluded.unsqueeze(-1)  # Add channel dimension -> (height, width, 1)

    # Add dimensions for batch
    for _ in range(image.ndim - 3):
        occluded = occluded.unsqueeze(0)  # Add batch dimension -> (*1, height, width, 1)

    # Adjust occlusion color shape to match the image
    occlusion_color_full = torch.zeros_like(image)
    occlusion_color_full[..., 0] = occlusion_color[0]
    occlusion_color_full[..., 1] = occlusion_color[1]
    occlusion_color_full[..., 2] = occlusion_color[2]

    return torch.where(occluded, occlusion_color_full, image)


def object_based_occlusion(image: torch.Tensor) -> torch.Tensor:
    """Object-based occlusion - compatible with *b h w c format."""
    # Get spatial dimensions
    height, width = image.shape[-3], image.shape[-2]
    device = image.device

    # Random occlusion position and size
    object_x = uniform_range(width//4, width*3//4)
    object_y = uniform_range(height//4, height*3//4)
    object_width = uniform_range(width//8, width//4)
    object_height = uniform_range(height//8, height//4)

    # Random occlusion color
    occlusion_color = torch.rand(3, device=device)

    # Create occlusion mask
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    occluded = (x_indices >= object_x) & (x_indices < object_x + object_width) & \
               (y_indices >= object_y) & (y_indices < object_y + object_height)

    # Adjust occlusion mask shape to match the image
    occluded = occluded.unsqueeze(-1)  # Add channel dimension -> (height, width, 1)

    # Add dimensions for batch
    for _ in range(image.ndim - 3):
        occluded = occluded.unsqueeze(0)  # Add batch dimension -> (*1, height, width, 1)

    # Adjust occlusion color shape to match the image
    occlusion_color_full = torch.zeros_like(image)
    occlusion_color_full[..., 0] = occlusion_color[0]
    occlusion_color_full[..., 1] = occlusion_color[1]
    occlusion_color_full[..., 2] = occlusion_color[2]

    return torch.where(occluded, occlusion_color_full, image)


def add_noise(image: torch.Tensor) -> torch.Tensor:
    """Add Gaussian and salt-and-pepper noise - compatible with *b h w c format."""
    shape = image.shape
    device = image.device

    # Gaussian noise
    sigma = uniform_range(0.01, 0.1)
    gaussian_noise = sigma * torch.randn(shape, device=device)
    image = image + gaussian_noise

    # Salt-and-pepper noise
    salt_prob = uniform_range(0.001, 0.01)
    pepper_prob = uniform_range(0.001, 0.01)

    # Create salt noise mask (for spatial dimensions)
    salt_mask_shape = shape[:-1]  # Remove channel dimension
    salt_mask = torch.rand(salt_mask_shape, device=device) < salt_prob

    # Create pepper noise mask (for spatial dimensions)
    pepper_mask = torch.rand(salt_mask_shape, device=device) < pepper_prob

    # Apply salt noise (white)
    if len(salt_mask_shape) < len(shape):
        salt_mask = salt_mask.unsqueeze(-1)
    image = torch.where(salt_mask, torch.ones_like(image), image)

    # Apply pepper noise (black)
    if len(pepper_mask.shape) < len(shape):
        pepper_mask = pepper_mask.unsqueeze(-1)
    image = torch.where(pepper_mask, torch.zeros_like(image), image)

    return torch.clamp(image, 0, 1)


def create_gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a Gaussian kernel."""
    epsilon = 1e-8
    safe_sigma = max(sigma, epsilon)

    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel_1d = torch.exp(-0.5 * (x / safe_sigma) ** 2)
    kernel_sum = torch.sum(kernel_1d)

    if kernel_sum < epsilon:
        kernel_1d = torch.ones(size, dtype=torch.float32, device=device) / size
    else:
        kernel_1d = kernel_1d / kernel_sum

    kernel_2d = torch.outer(kernel_1d, kernel_1d)

    if torch.isnan(kernel_2d).any() or torch.isinf(kernel_2d).any():
        kernel_2d = torch.ones((size, size), dtype=torch.float32, device=device) / (size * size)

    return kernel_2d


def gaussian_blur_conv(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Implement Gaussian blur using convolution."""
    device = image.device
    epsilon = 1e-8

    safe_sigma = max(sigma, epsilon)

    kernel_size = int(6 * safe_sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = create_gaussian_kernel(kernel_size, safe_sigma, device)

    # Reshape image to batch format (N, C, H, W)
    original_shape = image.shape
    if len(original_shape) == 4:  # (B, H, W, C)
        image = image.permute(0, 3, 1, 2)  # -> (B, C, H, W)
    elif len(original_shape) == 3:  # (H, W, C)
        image = image.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
    else:
        # Handle more complex batch dimensions
        batch_dims = original_shape[:-3]
        h, w, c = original_shape[-3:]
        image = image.view(-1, h, w, c).permute(0, 3, 1, 2)  # -> (B', C, H, W)

    # Prepare convolution kernel
    num_channels = image.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)

    # Apply convolution
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=num_channels)

    if torch.isnan(blurred).any() or torch.isinf(blurred).any():
        if len(original_shape) == 4:
            return image.permute(0, 2, 3, 1)
        elif len(original_shape) == 3:
            return image.squeeze(0).permute(1, 2, 0)
        else:
            return image.permute(0, 2, 3, 1).view(original_shape)

    # Restore original shape
    if len(original_shape) == 4:  # (B, H, W, C)
        blurred = blurred.permute(0, 2, 3, 1)
    elif len(original_shape) == 3:  # (H, W, C)
        blurred = blurred.squeeze(0).permute(1, 2, 0)
    else:
        # Restore complex batch dimensions
        blurred = blurred.permute(0, 2, 3, 1).view(original_shape)

    return blurred


def apply_motion_blur_kernel(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply motion blur to the image using a convolution kernel."""
    device = image.device
    epsilon = 1e-8

    # Reshape image to batch format (N, C, H, W)
    original_shape = image.shape
    if len(original_shape) == 4:  # (B, H, W, C)
        image = image.permute(0, 3, 1, 2)  # -> (B, C, H, W)
    elif len(original_shape) == 3:  # (H, W, C)
        image = image.permute(2, 0, 1).unsqueeze(0)  # -> (1, C, H, W)
    else:
        # Handle more complex batch dimensions
        batch_dims = original_shape[:-3]
        h, w, c = original_shape[-3:]
        image = image.view(-1, h, w, c).permute(0, 3, 1, 2)  # -> (B', C, H, W)

    # Prepare convolution kernel
    num_channels = image.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(num_channels, 1, -1, -1)

    # Ensure kernel is normalized
    kernel_sum = torch.sum(kernel)
    if kernel_sum < epsilon:
        kernel = torch.ones_like(kernel) / kernel.numel()
    else:
        kernel = kernel / kernel_sum

    # Apply convolution
    padding = kernel.shape[-1] // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=num_channels)

    # Restore original shape
    if len(original_shape) == 4:  # (B, H, W, C)
        blurred = blurred.permute(0, 2, 3, 1)
    elif len(original_shape) == 3:  # (H, W, C)
        blurred = blurred.squeeze(0).permute(1, 2, 0)
    else:
        # Restore complex batch dimensions
        blurred = blurred.permute(0, 2, 3, 1).view(original_shape)

    # NaN check
    if torch.isnan(blurred).any() or torch.isinf(blurred).any():
        return image

    return blurred


def simulate_blur(image: torch.Tensor) -> torch.Tensor:
    """Simulate Gaussian and motion blur."""
    epsilon = 1e-8

    # Randomly choose blur type
    blur_type = randint_range(0, 2)

    if blur_type == 0:
        # Gaussian blur
        sigma_choice = randint_range(0, 3)
        if sigma_choice == 0:
            sigma = 1.0 + epsilon
        elif sigma_choice == 1:
            sigma = 5.0/3 + epsilon
        else:
            sigma = 7.0/3 + epsilon

        result = gaussian_blur_conv(image, sigma)
    else:
        # Motion blur
        angle = uniform_range(0, 180, epsilon=epsilon)

        # Create motion blur kernel
        kernel_size = 7
        center = kernel_size // 2
        kernel = torch.zeros((kernel_size, kernel_size), device=image.device)

        # Choose direction based on angle - simplified rotation implementation
        if 45 <= angle < 135:  # Vertical direction
            kernel[:, center] = 1.0 / kernel_size
        elif 135 <= angle < 225:  # Horizontal direction
            kernel[center, :] = 1.0 / kernel_size
        else:  # Default to horizontal direction
            kernel[center, :] = 1.0 / kernel_size

        # Apply motion blur convolution
        result = apply_motion_blur_kernel(image, kernel)

    if torch.isnan(result).any() or torch.isinf(result).any():
        return image

    return result


def add_scratch_and_stain(image: torch.Tensor) -> torch.Tensor:
    """Add scratches and stains - compatible with *b h w c format."""
    # Get spatial dimensions
    height, width = image.shape[-3], image.shape[-2]
    device = image.device
    epsilon = 1e-8

    def add_single_scratch(img: torch.Tensor) -> torch.Tensor:
        # Random scratch position and direction
        x = uniform_range(0, width, epsilon=epsilon)
        y = uniform_range(0, height, epsilon=epsilon)
        angle = uniform_range(0, 180, epsilon=epsilon)
        length = uniform_range(20, 100, epsilon=epsilon)

        # Create scratch color
        scratch_color = torch.rand(3, device=device)

        # Create grid
        y_indices, x_indices = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing='ij'
        )
        dx = math.cos(math.radians(angle)) * length
        dy = math.sin(math.radians(angle)) * length

        # Calculate distance from point to line
        vx = x_indices - x
        vy = y_indices - y
        cross = torch.abs(dx * vy - dy * vx)
        length_line = math.sqrt(dx**2 + dy**2) + epsilon
        distance = cross / length_line

        # Apply scratch
        scratch_width = 2.0
        scratched = distance < scratch_width  # Shape: (height, width)

        # Adjust scratch mask shape to match the image
        scratched = scratched.unsqueeze(-1)  # Add channel dimension -> (height, width, 1)

        # Add dimensions for batch
        for _ in range(img.ndim - 3):
            scratched = scratched.unsqueeze(0)  # Add batch dimension

        # Adjust scratch color shape to match the image
        scratch_color_full = torch.zeros_like(img)
        scratch_color_full[..., 0] = scratch_color[0]
        scratch_color_full[..., 1] = scratch_color[1]
        scratch_color_full[..., 2] = scratch_color[2]

        return torch.where(scratched, scratch_color_full, img)

    def add_single_stain(img: torch.Tensor) -> torch.Tensor:
        # Random stain position and size
        x = uniform_range(0, width, epsilon=epsilon)
        y = uniform_range(0, height, epsilon=epsilon)
        radius = uniform_range(5, 20, epsilon=epsilon)

        # Create stain color
        stain_color = torch.rand(3, device=device)

        # Create grid
        y_indices, x_indices = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing='ij'
        )
        distance = torch.sqrt((x_indices - x)**2 + (y_indices - y)**2 + epsilon)
        stained = distance < radius  # Shape: (height, width)

        # Adjust stain mask shape to match the image
        stained = stained.unsqueeze(-1)  # Add channel dimension -> (height, width, 1)

        # Add dimensions for batch
        for _ in range(img.ndim - 3):
            stained = stained.unsqueeze(0)  # Add batch dimension

        # Adjust stain color shape to match the image
        stain_color_full = torch.zeros_like(img)
        stain_color_full[..., 0] = stain_color[0]
        stain_color_full[..., 1] = stain_color[1]
        stain_color_full[..., 2] = stain_color[2]

        return torch.where(stained, stain_color_full, img)

    # Add scratch first, then stain
    result = add_single_scratch(image)
    result = add_single_stain(result)
    result = torch.clamp(result, 0, 1)

    if torch.isnan(result).any() or torch.isinf(result).any():
        return image

    return result


def image_shift(image: torch.Tensor) -> torch.Tensor:
    """Randomly shift the image - compatible with *b h w c format."""
    # Get spatial dimensions
    height, width = image.shape[-3], image.shape[-2]
    device = image.device

    # Randomly generate shift amounts (max shift ratio is 10%)
    max_shift_ratio = 0.1
    max_shift_x = int(width * max_shift_ratio)
    max_shift_y = int(height * max_shift_ratio)

    shift_x = int(randint_range(-max_shift_x, max_shift_x + 1))
    shift_y = int(randint_range(-max_shift_y, max_shift_y + 1))

    # Use PyTorch's roll function for shifting
    shifted_image = torch.roll(image, shifts=shift_x, dims=-2)  # Horizontal shift
    shifted_image = torch.roll(shifted_image, shifts=shift_y, dims=-3)  # Vertical shift

    # Handle boundaries: set exposed boundaries to black
    if shift_x > 0:
        shifted_image[..., :shift_x, :] = 0
    elif shift_x < 0:
        shifted_image[..., shift_x:, :] = 0

    if shift_y > 0:
        shifted_image[..., :shift_y, :, :] = 0
    elif shift_y < 0:
        shifted_image[..., shift_y:, :, :] = 0

    return torch.clamp(shifted_image, 0, 1)


def image_rotation(image: torch.Tensor) -> torch.Tensor:
    """Randomly rotate the image - compatible with *b h w c format, simplified version."""
    # Get spatial dimensions
    height, width = image.shape[-3], image.shape[-2]
    device = image.device

    # Randomly generate rotation angle (max 30 degrees)
    max_angle = 30.0
    angle = uniform_range(-max_angle, max_angle)

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Calculate rotation center
    center_x = width / 2.0
    center_y = height / 2.0

    # Create coordinate grid
    y_indices, x_indices = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Convert coordinates to be relative to the center
    x_centered = x_indices - center_x
    y_centered = y_indices - center_y

    # Apply rotation transformation
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    x_rotated = x_centered * cos_angle - y_centered * sin_angle + center_x
    y_rotated = x_centered * sin_angle + y_centered * cos_angle + center_y

    # Use nearest-neighbor interpolation for image rotation
    x_rotated = torch.round(x_rotated).long()
    y_rotated = torch.round(y_rotated).long()

    # Create a mask for valid coordinates
    valid_mask = (x_rotated >= 0) & (x_rotated < width) & (y_rotated >= 0) & (y_rotated < height)

    # Initialize rotated image (black background)
    rotated_image = torch.zeros_like(image)

    # For valid coordinates, sample from the original image
    # Use conditional indexing to avoid out-of-bounds
    x_safe = torch.clamp(x_rotated, 0, width - 1)
    y_safe = torch.clamp(y_rotated, 0, height - 1)

    # Sample pixel values from the original image
    sampled_values = image[..., y_safe, x_safe, :]

    # Apply sampled values only at valid positions
    valid_mask_expanded = valid_mask.unsqueeze(-1)  # Expand to channel dimension

    # Add dimensions for batch
    for _ in range(image.ndim - 3):
        valid_mask_expanded = valid_mask_expanded.unsqueeze(0)

    rotated_image = torch.where(valid_mask_expanded, sampled_values, rotated_image)

    return torch.clamp(rotated_image, 0, 1)


def apply_random_augmentation(image: torch.Tensor, augmentation_type: Optional[str] = None) -> torch.Tensor:
    """Apply random image augmentation - fully compatible with *b h w c format.

    Args:
        image: Input image, with shape (*b, h, w, c).
        augmentation_type: Optional, specify the type of augmentation to apply.

    Returns:
        The augmented image.
    """

    # Mapping of available augmentation functions
    augmentation_functions = {
        'color_jittering': color_jittering,
        'light_intensity_adjustment': light_intensity_adjustment,
        'simulate_light_direction': simulate_light_direction,
        'add_shadow': add_shadow,
        'random_occlusion': random_occlusion,
        'object_based_occlusion': object_based_occlusion,
        'simulate_blur': simulate_blur,
        'add_scratch_and_stain': add_scratch_and_stain,
        'add_noise': add_noise,
        'image_shift': image_shift,
        'image_rotation': image_rotation
    }

    # If no augmentation type is specified, choose one randomly
    if augmentation_type is None:
        augmentation_idx = randint_range(0, len(augmentation_functions))
        augmentation_keys = list(augmentation_functions.keys())
        augmentation_type = augmentation_keys[int(augmentation_idx)]

    # If an augmentation type is specified, call the corresponding function directly
    if augmentation_type in augmentation_functions:
        return augmentation_functions[augmentation_type](image)
    else:
        # Unknown type, do not augment
        return image

def apply_visual_augmentation(image: torch.Tensor, augmentation_name: str) -> torch.Tensor:
    """
    Visual augmentation interface for the OpenVLA UCB augmentation balancer.

    Args:
        image: Input image tensor, with shape (*batch_dims, height, width, channels).
        augmentation_name: The name of the augmentation type.

    Returns:
        The augmented image tensor.
    """
    return apply_random_augmentation(image, augmentation_name)
