import json
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import fastapi
from fastapi import Body, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.cache import update_cache_and_results
from src.constants import DEVICE, DOMAIN_MAPS
from src.endpoints.colorize_model_change import colorize_model_change
from src.endpoints.colorize_params_change import colorize_param_change
from src.endpoints.delete_cache import delete_images_cache
from src.endpoints.process_images import check_cache_and_prepare_processing
from src.logger import logger
from src.ml_models import COLORIZER
from src.ocr.bbox_identification import detect_text_regions
from src.ocr.translate_bbox_cuts import translate_text_regions
from src.utils import colorize_images, colorize_images_comfyui, convert_np_images_to_base64, preprocess_images, upscale_images

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
async def process_images(
    images_request: ImagesRequest,
    translate: bool,
    colorize: bool,
    upscale: bool,
    colorize_method: str = "gan",
    comfyui_url: str = "http://localhost:9010",
):
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
            if colorize_method == "comfyui":
                processed_images = await colorize_images_comfyui(processed_images, comfyui_url)
            else:
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


@app.post("/process-images-stream")
async def process_images_stream(
    request: Request,
    images_request: ImagesRequest,
    translate: bool,
    colorize: bool,
    upscale: bool,
    colorize_method: str = "gan",
    comfyui_url: str = "http://localhost:9010",
):
    """
    Streaming endpoint that processes images one at a time and yields each
    result immediately as NDJSON (newline-delimited JSON).
    Format per line: {"index": <int>, "image": <base64_str>}

    Uses a prefetch strategy: keeps 2 requests active at once (1 processing
    + 1 queued ahead) to eliminate idle time between images. On client
    disconnect, the prefetched task is cancelled so at most 1 extra image
    is wasted on the ComfyUI server.
    """

    async def process_single_image(i: int, b64_img: str) -> str:
        """Process a single image through the full pipeline. Returns NDJSON line."""
        from src.utils import generate_cache_key
        from src.cache import translate_dc_cache

        try:
            cache_key = generate_cache_key(b64_img, translate, colorize, upscale)
            if cache_key in translate_dc_cache:
                logger.info(f"[Stream] Cache hit for image {i}")
                result_b64 = translate_dc_cache[cache_key]
                return json.dumps({"index": i, "image": result_b64}) + "\n"

            processed = preprocess_images([b64_img])
            img = processed[0]

            if translate:
                from src.ocr.bbox_identification import detect_text_regions
                from src.ocr.translate_bbox_cuts import translate_text_regions

                bbox_results = detect_text_regions([img])
                [img] = await translate_text_regions([img], bbox_results)

            if colorize:
                if colorize_method == "comfyui":
                    [img] = await colorize_images_comfyui([img], comfyui_url)
                else:
                    [img] = colorize_images([img])

            if upscale:
                [img] = upscale_images([img], scale_factor=3)

            [result_b64] = convert_np_images_to_base64([img])

            translate_dc_cache[cache_key] = result_b64
            logger.info(f"[Stream] Processed and cached image {i}")

            return json.dumps({"index": i, "image": result_b64}) + "\n"

        except Exception as e:
            logger.error(f"[Stream] Error processing image {i}: {str(e)}")
            traceback.print_exc()
            return json.dumps({"index": i, "image": None, "error": str(e)}) + "\n"

    async def generate():
        import asyncio

        images = images_request.images
        if not images:
            return

        # Build work list of valid (index, image) pairs
        work_items = [(i, img) for i, img in enumerate(images) if img]
        if not work_items:
            return

        work_iter = iter(work_items)
        # Start first task
        i, img = next(work_iter)
        current_task = asyncio.ensure_future(process_single_image(i, img))

        # Start prefetch (1 ahead) if more work exists
        prefetch_task = None
        next_item = next(work_iter, None)
        if next_item is not None:
            pi, pimg = next_item
            prefetch_task = asyncio.ensure_future(process_single_image(pi, pimg))

        while current_task is not None:
            # Check for client disconnect before awaiting
            if await request.is_disconnected():
                logger.info("[Stream] Client disconnected, cancelling remaining work")
                current_task.cancel()
                if prefetch_task is not None:
                    prefetch_task.cancel()
                return

            # Wait for the current task to finish
            result = await current_task
            yield result

            # Advance: current becomes the prefetch, start a new prefetch
            current_task = prefetch_task
            prefetch_task = None

            if current_task is not None:
                next_item = next(work_iter, None)
                if next_item is not None:
                    pi, pimg = next_item
                    prefetch_task = asyncio.ensure_future(process_single_image(pi, pimg))

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/colorize-model-change")
async def colorize_model_change_endpoint(model_path: str):
    """
    Endpoint to change the colorization model.
    This is a placeholder for future implementation.
    """
    return await colorize_model_change(model_path)


@app.get("/colorize-param-change")
async def colorize_param_change_endpoint(denoise_sigma: float, image_tile_size: int):
    """
    Endpoint to change the colorization parameters.
    This is a placeholder for future implementation.
    """
    return await colorize_param_change(denoise_sigma=denoise_sigma, image_tile_size=image_tile_size)
