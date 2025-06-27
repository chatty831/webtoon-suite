import base64
import hashlib
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from src.colorize.colorizer import colorize_batch
from src.constants import DEVICE, DTYPE
from src.enhance.upscale import upscale_image
from src.logger import logger
from src.ml_models import COLORIZER, DENOISER


def is_url(path: str) -> bool:
    """
    Helper function to determine if the provided string is a URL or a file path.

    Args:
        path: String to check

    Returns:
        True if path is a URL, False otherwise
    """
    parsed = urlparse(path)
    return bool(parsed.scheme and parsed.netloc)


def generate_image_hash(b64_img: str) -> str:
    """Generate a hash for the image content to use as a cache key.
    Using MD5 on the binary image data gives us a compact unique identifier.
    """
    # Hash the raw image data rather than the entire base64 string
    img_data = base64.b64decode(b64_img)
    return hashlib.md5(img_data).hexdigest()


def generate_cache_key(base64_image: str, translate: bool, colorize: bool, upscale: bool) -> str:
    """Generate parameter-specific cache key for an image."""
    img_hash = generate_image_hash(base64_image)
    param_key = f"_t{int(translate)}_c{int(colorize)}_u{int(upscale)}"
    return f"{img_hash}{param_key}"


def preprocess_images(base64_images: List[str]) -> List[np.ndarray]:
    """Decode base64 images and convert to RGB format."""
    rgb_images = []

    for b64_img in base64_images:
        try:
            img_data = base64.b64decode(b64_img)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Convert BGR to RGB (critical step)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_images.append(img_rgb)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    return rgb_images


def colorize_images(images: List[np.ndarray]) -> List[np.ndarray]:
    """Apply colorization to a batch of images."""
    return colorize_batch(COLORIZER, DENOISER, images, DEVICE, DTYPE)


def upscale_images(images: List[np.ndarray], scale_factor: int = 3) -> List[np.ndarray]:
    """Upscale images using the specified scale factor."""
    upscaled_images = []
    for img in images:
        upscaled_img = upscale_image(img, outscale=scale_factor)
        upscaled_images.append(upscaled_img)
    return upscaled_images


def convert_np_images_to_base64(images: List[np.ndarray]) -> List[str]:
    """Convert processed RGB images to base64 strings."""
    base64_images = []

    for img in images:
        if img is None:
            base64_images.append(None)
            continue
        # Convert the image to a PIL Image
        pil_img = Image.fromarray(img)
        # Save the image to a buffer
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        # Convert buffer to base64 string
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_str)

    return base64_images


def convert_base64_to_np_images(base64_images: List[str]) -> List[np.ndarray]:
    """Convert a list of base64 encoded strings to numpy images."""
    np_images = []

    for base64_str in base64_images:
        if base64_str is None:
            np_images.append(None)
            continue
        try:
            # Decode the base64 string to bytes
            image_data = base64.b64decode(base64_str)
            # Convert bytes to a numpy array
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            # Decode the numpy array to an image
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            np_images.append(img_rgb)
        except Exception as e:
            logger.error(f"Error converting base64 to numpy image: {str(e)}")
            raise

    return np_images


def convert_pil_images_to_base64(pil_images: List[Image.Image]) -> List[str]:
    """Convert a list of PIL images to base64 encoded strings."""
    base64_images = []

    for pil_img in pil_images:
        if pil_img is None:
            base64_images.append(None)
            continue
        # Save the image to a buffer
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        # Convert buffer to base64 string
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_str)

    return base64_images


def convert_base64_to_pil_images(base64_images: List[str]) -> List[Image.Image]:
    """Convert a list of base64 encoded strings to PIL images."""
    pil_images = []

    for base64_str in base64_images:
        if base64_str is None:
            pil_images.append(None)
            continue
        # Decode the base64 string to bytes
        image_data = base64.b64decode(base64_str)
        # Convert bytes to a numpy array
        img_array = np.frombuffer(image_data, dtype=np.uint8)
        # Decode the numpy array to an image using OpenCV
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert the numpy array to a PIL Image
            pil_img = Image.fromarray(img_rgb)
            pil_images.append(pil_img)
        else:
            logger.error("Error decoding image from base64 string.")
            raise ValueError("Error decoding image from base64 string.")

    return pil_images
