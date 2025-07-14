from src.data.config import DEFAULT_CONFIG
from src.logger import logger


async def colorize_param_change(denoise_sigma: float, image_tile_size: int):
    """
    Endpoint to change the colorization parameters.
    This is a placeholder for future implementation.
    """
    # Update the default configuration with new parameters
    DEFAULT_CONFIG["denoise_sigma"] = denoise_sigma
    DEFAULT_CONFIG["image_tile_size"] = image_tile_size

    # Log the changes
    logger.info(f"Colorization parameters updated: denoise_sigma={denoise_sigma}, image_tile_size={image_tile_size}")

    return {"message": "Colorization parameters updated successfully"}
