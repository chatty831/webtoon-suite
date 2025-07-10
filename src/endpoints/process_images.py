from typing import Any, Dict, List, Optional, Tuple, Union

from src.cache import translate_dc_cache
from src.logger import logger
from src.utils import generate_cache_key


def check_cache_and_prepare_processing(images: List[str], translate: bool, colorize: bool, upscale: bool):
    """
    Check cache for processed images and prepare uncached ones for processing.
    Returns cached results, images to process, and metadata.
    """
    processed_base64_images = []
    images_to_process = []
    process_indices = []
    cache_keys = []

    for i, b64_img in enumerate(images):
        if not b64_img:
            continue
        cache_key = generate_cache_key(b64_img, translate, colorize, upscale)
        cache_keys.append(cache_key)

        if cache_key in translate_dc_cache:
            logger.info(f"Cache hit for image {i}")
            processed_base64_images.append(translate_dc_cache[cache_key])
        else:
            logger.info(f"Cache miss for image {i}")
            processed_base64_images.append(None)  # Placeholder
            process_indices.append(i)
            images_to_process.append(b64_img)

    return processed_base64_images, images_to_process, process_indices, cache_keys
