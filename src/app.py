import re
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import fastapi
from fastapi import Body, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.cache import update_cache_and_results
from src.constants import DOMAIN_MAPS
from src.endpoints.delete_cache import delete_images_cache
from src.endpoints.process_images import check_cache_and_prepare_processing
from src.logger import logger
from src.ocr.bbox_identification import detect_text_regions
from src.ocr.translate_bbox_cuts import translate_text_regions
from src.utils import colorize_images, convert_np_images_to_base64, preprocess_images, upscale_images

app = fastapi.FastAPI()


@app.get("/liveness-probe")
async def liveness_probe():
    return {"status": "ok"}


class ImagesRequest(BaseModel):
    images: List[str | None]


@app.get("/scrape-images")
async def scrape_images(url: str):
    domain = re.search(r"^(?:https?://)?(?:www\.)?([^/:]+)", url, re.IGNORECASE).group(1)
    scrape_func = DOMAIN_MAPS[domain]
    results = await scrape_func(url)
    return {"scraped_images": results}


@app.post("/delete-cache")
async def delete_cache(images_request: ImagesRequest):
    """
    Endpoint to delete cache for specific images
    """
    try:
        result = delete_images_cache(images_request.images)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error deleting cache: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/process-images")
async def process_images(images_request: ImagesRequest, translate: bool, colorize: bool, upscale: bool):
    """
    Endpoint to process images with efficient disk caching and proper RGB format handling.
    Cache is now parameter-aware, storing different versions based on processing flags.
    """
    try:
        if not images_request.images:
            return {"translated_images": []}

        logger.info(f"Processing with flags - translate: {translate}, colorize: {colorize}, upscale: {upscale}")

        # Check cache and prepare images for processing
        final_results, images_to_process, process_indices, cache_keys = check_cache_and_prepare_processing(
            images_request.images, translate, colorize, upscale
        )

        # If all images were in cache, return immediately
        if not images_to_process:
            logger.info("All images found in cache")
            return {"translated_images": final_results}

        # Preprocess images (decode and convert to RGB)
        processed_images = preprocess_images(images_to_process)

        # Apply translation processing if requested
        if translate:
            # Step 1: Detect text regions (bounding boxes)
            bbox_results = detect_text_regions(processed_images)

            # Step 2: Translate text regions and apply to images
            processed_images = await translate_text_regions(processed_images, bbox_results)

        # Apply colorization if requested
        if colorize:
            processed_images = colorize_images(processed_images)

        # Apply upscaling if requested
        if upscale:
            processed_images = upscale_images(processed_images, scale_factor=3)

        # Convert processed images back to base64
        base64_results = convert_np_images_to_base64(processed_images)

        # Update cache and merge results
        final_results = update_cache_and_results(base64_results, process_indices, cache_keys, final_results)

        return {"translated_images": final_results}

    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing images")
