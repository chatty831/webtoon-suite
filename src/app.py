import asyncio
import json
import re
import traceback
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import fastapi
from fastapi import Body, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.cache import scrape_lru_cache, update_cache_and_results
from src.constants import DEVICE, resolve_scraper
from src.endpoints.colorize_model_change import colorize_model_change
from src.endpoints.colorize_params_change import colorize_param_change
from src.endpoints.delete_cache import delete_images_cache
from src.endpoints.process_images import check_cache_and_prepare_processing
from src.logger import logger
from src.ml_models import COLORIZER
from src.ocr.bbox_identification import detect_text_regions
from src.ocr.translate_bbox_cuts import ocr_page, translate_and_render_page, translate_text_regions
from src.scheduler import process_scheduler, scrape_scheduler
from src.utils import colorize_images, colorize_images_comfyui, convert_np_images_to_base64, preprocess_images, upscale_images


# ---------------------------------------------------------------------------
# App lifecycle — start / stop scheduler workers
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    scrape_scheduler.start()
    process_scheduler.start()
    yield
    await process_scheduler.stop()
    await scrape_scheduler.stop()


app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/liveness-probe")
async def liveness_probe():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@app.post("/session/{session_id}/cancel")
async def cancel_session_endpoint(session_id: str):
    """Cancel all pending/in-flight tasks for a session across all schedulers."""
    scrape_scheduler.cancel_session(session_id)
    process_scheduler.cancel_session(session_id)
    return {"status": "cancelled", "session_id": session_id}


# ---------------------------------------------------------------------------
# Scrape
# ---------------------------------------------------------------------------

class ImagesRequest(BaseModel):
    images: List[str | None]


@app.get("/scrape-images")
async def scrape_images(url: str, session_id: str = ""):
    # Check scrape cache first
    if url in scrape_lru_cache:
        logger.info(f"[Scrape] Cache hit for {url}")
        return {"scraped_images": scrape_lru_cache[url]}

    if not session_id:
        session_id = str(_uuid.uuid4())

    domain = re.search(r"^(?:https?://)?(?:www\.)?([^/:]+)", url, re.IGNORECASE).group(1)
    scrape_func = resolve_scraper(domain)
    if scrape_func is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported domain: {domain}",
        )

    async def scrape_task():
        results = await scrape_func(url)
        scrape_lru_cache[url] = results
        logger.info(f"[Scrape] Cached {len(results)} images for {url}")
        return results

    session = await scrape_scheduler.submit_batch(session_id, [scrape_task])

    try:
        result = await session.result_queue.get()
    finally:
        scrape_scheduler.cleanup_session(session_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scrape was cancelled or failed",
        )

    return {"scraped_images": result}


# ---------------------------------------------------------------------------
# Delete cache
# ---------------------------------------------------------------------------

@app.post("/delete-cache")
async def delete_cache(images_request: ImagesRequest):
    """Endpoint to delete cache for specific images."""
    try:
        result = delete_images_cache(images_request.images)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error deleting cache: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ---------------------------------------------------------------------------
# Process images (batch — not routed through scheduler)
# ---------------------------------------------------------------------------

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
            processed_images, _ = await translate_text_regions(processed_images, bbox_results)

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


# ---------------------------------------------------------------------------
# Process images (streaming — routed through process_scheduler)
# ---------------------------------------------------------------------------

@app.post("/process-images-stream")
async def process_images_stream(
    request: Request,
    images_request: ImagesRequest,
    translate: bool,
    colorize: bool,
    upscale: bool,
    session_id: str = "",
    colorize_method: str = "gan",
    comfyui_url: str = "http://localhost:9010",
):
    """
    Streaming endpoint that yields NDJSON as each image finishes.
    Format per line: {"index": <int>, "image": <base64_str>}

    Pipeline per page (within the process_scheduler):
      Phase 1 — YOLO + batch OCR (unbounded, no rate limit)
        → signals page_ocr_events[i] when done
      Phase 2 — Contextual translation (rate-limited by LLM semaphore)
        → waits for page_ocr_events[i-1] so previous page's raw OCR
          is available as narrative context

    All pages' OCR runs in parallel.  Since OCR is fast and unbounded,
    by the time any page reaches Phase 2 the previous page's OCR is
    almost always already complete, so translations also run in parallel.
    """
    if not session_id:
        session_id = str(_uuid.uuid4())

    # Shared state across all page tasks in this session
    page_ocr_events: dict[int, asyncio.Event] = {}   # signaled when OCR finishes
    page_ocr_texts: dict[int, list[str]] = {}         # raw OCR results per page

    # Build work list
    images = images_request.images or []
    work_items = [(i, img) for i, img in enumerate(images) if img]

    if not work_items:
        return StreamingResponse(iter([]), media_type="application/x-ndjson")

    # Pre-create OCR events for all pages (so any page can wait on its predecessor)
    if translate:
        for i, _ in work_items:
            page_ocr_events[i] = asyncio.Event()

    def make_process_factory(i: int, b64_img: str):
        """Build a closure that processes one image through the full pipeline."""

        async def factory():
            from src.utils import generate_cache_key
            from src.cache import translate_dc_cache

            try:
                cache_key = generate_cache_key(b64_img, translate, colorize, upscale)
                if cache_key in translate_dc_cache:
                    logger.info(f"[Stream] Cache hit for image {i}")
                    # Signal OCR done even for cached pages so next page doesn't hang
                    if i in page_ocr_events:
                        page_ocr_texts[i] = []
                        page_ocr_events[i].set()
                    result_b64 = translate_dc_cache[cache_key]
                    return json.dumps({"index": i, "image": result_b64}) + "\n"

                processed = preprocess_images([b64_img])
                img = processed[0]

                if translate:
                    # Phase 1: YOLO + batch OCR (unbounded parallelism)
                    bbox_results = detect_text_regions([img])
                    bboxes = bbox_results[0] if bbox_results else []

                    raw_texts = await ocr_page(img, bboxes) if bboxes else []
                    page_ocr_texts[i] = raw_texts
                    page_ocr_events[i].set()
                    logger.info(f"[Stream] Page {i} OCR done ({len(raw_texts)} texts)")

                    # Phase 2: Wait for previous page's OCR, then translate
                    if bboxes and raw_texts:
                        prev_event = page_ocr_events.get(i - 1)
                        if prev_event:
                            await prev_event.wait()
                            logger.info(f"[Stream] Page {i} using OCR context from page {i - 1}")
                        else:
                            logger.info(f"[Stream] Page {i} proceeding without previous context")

                        prev_raw = page_ocr_texts.get(i - 1)
                        img = await translate_and_render_page(
                            img, bboxes, raw_texts, prev_raw
                        )

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
                # Signal OCR done even on error so next page doesn't hang
                if i in page_ocr_events and not page_ocr_events[i].is_set():
                    page_ocr_texts[i] = []
                    page_ocr_events[i].set()
                return json.dumps({"index": i, "image": None, "error": str(e)}) + "\n"

        return factory

    # Submit all image tasks to the scheduler
    factories = [make_process_factory(i, img) for i, img in work_items]
    session = await process_scheduler.submit_batch(session_id, factories)

    async def generate():
        remaining = len(factories)
        try:
            while remaining > 0:
                # Check for client disconnect periodically
                if await request.is_disconnected():
                    logger.info("[Stream] Client disconnected, cancelling session")
                    process_scheduler.cancel_session(session_id)
                    return

                try:
                    result = await asyncio.wait_for(
                        session.result_queue.get(), timeout=2.0
                    )
                    remaining -= 1
                    if result is not None:  # None = cancelled/skipped task
                        yield result
                except asyncio.TimeoutError:
                    # No result yet — loop back to check disconnect
                    continue
        finally:
            process_scheduler.cleanup_session(session_id)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Model configuration endpoints
# ---------------------------------------------------------------------------

@app.get("/list-colorize-models")
async def list_colorize_models():
    """Return available colorization model filenames."""
    import os
    model_dir = "models/colorize_models"
    if os.path.isdir(model_dir):
        return {"models": os.listdir(model_dir)}
    return {"models": []}


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
