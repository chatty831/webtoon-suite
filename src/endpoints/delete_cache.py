
import asyncio
import base64
import hashlib
import os
import re
import traceback
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import diskcache as dc
import fastapi
import numpy as np
from cachetools import TTLCache, cached
from fastapi import Body, HTTPException, status
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from sahi.predict import get_sliced_prediction
import torch
from fastapi.responses import JSONResponse

from src.constants import DOMAIN_MAPS, DEVICE, DTYPE
from src.ocr.gpt_ocr import gpt_ocr
from src.image_utils import postprocess_bboxes
from src.logger import logger
from src.ml_models import YOLO_MODEL, COLORIZER, DENOISER
from src.colorize.colorizer import colorize_batch
from src.enhance.upscale import upscale_image
from src.utils import is_url, generate_image_hash, generate_cache_key
from src.cache import translate_dc_cache, image_ttl_cache


def delete_images_cache(images: list) -> dict:
    """
    Process cache deletion for multiple images with all parameter permutations
    
    Args:
        images: List of base64 encoded images
        
    Returns:
        dict: Result summary with deletion counts and any errors
    """
    if not images:
        return {"deleted_count": 0, "cache_misses": 0, "message": "No images provided"}

    # Define all permutations of the processing flags
    permutations = [
        (t, c, u) for t in [0, 1] for c in [0, 1] for u in [0, 1]
    ]

    deleted_count = 0
    cache_misses = 0

    for i, b64_img in enumerate(images):
        if b64_img is None:  # Skip None entries
            continue

        img_hash = generate_image_hash(b64_img)

        # Check and delete cache for each permutation of parameters
        image_cache_found = False
        for t, c, u in permutations:
            param_key = f"_t{t}_c{c}_u{u}"
            cache_key = f"{img_hash}{param_key}"
            if cache_key in translate_dc_cache:
                del translate_dc_cache[cache_key]
                logger.info(f"Deleted cache for image hash {img_hash[:8]} with params {param_key}...")
                deleted_count += 1
                image_cache_found = True

        if not image_cache_found:
            logger.info(f"Cache miss for image {i}")
            cache_misses += 1

    return {
        "deleted_count": deleted_count,
        "cache_misses": cache_misses,
        "message": "Cache deletion completed successfully"
    }
