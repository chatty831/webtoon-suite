
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
from src.utils import is_url, generate_cache_key
from src.cache import translate_dc_cache, image_ttl_cache
from src.endpoints.delete_cache import delete_images_cache


def detect_text_regions(images: List[np.ndarray]) -> List[List]:
    """
    Detect text regions in images using YOLO model.
    Returns processed bounding boxes for each image.
    """
    # Run object detection on all images
    results = [
        get_sliced_prediction(
            image=image,
            detection_model=YOLO_MODEL,
            slice_height=720,
            slice_width=720,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        for image in images
    ]

    # Process detection results
    processed_results = []
    for result in results:
        bboxes = []
        for detection in result.object_prediction_list:
            bbox_xywh = detection.bbox.to_xywh()
            bboxes.append(bbox_xywh)
        
        # Post-process bounding boxes
        final_bboxes = postprocess_bboxes(bboxes)
        processed_results.append(final_bboxes)

    return processed_results