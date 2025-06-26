

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