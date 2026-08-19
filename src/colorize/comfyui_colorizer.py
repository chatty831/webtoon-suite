import os
import json
import asyncio
import aiohttp
import time
import uuid
import tempfile
import random
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

from src.logger import logger


async def upload_image_to_comfyui(
    session: aiohttp.ClientSession,
    comfyui_url: str,
    image: np.ndarray,
    image_index: int,
) -> Dict:
    """
    Upload a numpy RGB image to a ComfyUI instance.
    Saves to a temp file, uploads, then cleans up.
    """
    upload_url = f"{comfyui_url}/upload/image"
    temp_path = None

    try:
        pil_img = PILImage.fromarray(image)

        unique_id = str(uuid.uuid4())[:8]
        unique_filename = f"manga_img_{image_index}_{unique_id}.png"

        temp_path = os.path.join(tempfile.gettempdir(), unique_filename)
        pil_img.save(temp_path, format="PNG")

        with open(temp_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field(
                "image",
                f,
                filename=unique_filename,
                content_type="image/png",
            )

            async with session.post(upload_url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "status": "success",
                        "filename": result.get("name", unique_filename),
                        "image_index": image_index,
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "message": f"Upload failed: {response.status} - {error_text}",
                        "image_index": image_index,
                    }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "image_index": image_index,
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def create_comfyui_colorize_workflow(
    image_name: str,
    output_prefix: str = "colorized",
    positive_prompt: Optional[str] = None,
) -> Dict:
    """
    Create a ComfyUI API prompt for FLUX-based manga colorization.
    """
    if positive_prompt is None:
        positive_prompt = "colorize this manga image"

    prompt = {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "fused_kontext_v2.safetensors",
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "flux",
                "device": "default",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"},
        },
        "142": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name, "upload": "image"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["38", 0], "text": positive_prompt},
        },
        "135": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "124": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["142", 0], "vae": ["39", 0]},
        },
        "177": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["6", 0], "latent": ["124", 0]},
        },
        "35": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["177", 0], "guidance": 4.0},
        },
        "31": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["37", 0],
                "positive": ["35", 0],
                "negative": ["135", 0],
                "latent_image": ["124", 0],
                "seed": 42,
                "steps": 15,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["31", 0], "vae": ["39", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": output_prefix},
        },
    }

    return prompt


async def send_inference_request(
    session: aiohttp.ClientSession,
    comfyui_url: str,
    workflow: Dict,
    client_id: str,
) -> Dict:
    """Send inference request to a ComfyUI instance."""
    prompt_url = f"{comfyui_url}/prompt"
    payload = {"prompt": workflow, "client_id": client_id}

    try:
        async with session.post(prompt_url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "status": "success",
                    "prompt_id": result.get("prompt_id"),
                    "client_id": client_id,
                }
            else:
                error_text = await response.text()
                return {
                    "status": "error",
                    "message": f"Request failed: {response.status} - {error_text}",
                }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def wait_for_completion(
    session: aiohttp.ClientSession,
    comfyui_url: str,
    prompt_id: str,
    check_interval: float = 1.0,
    save_node_id: str = "9",
    timeout: float = 300.0,
) -> Dict:
    """Wait for a prompt to complete on a ComfyUI instance."""
    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            return {"status": "failed", "message": "Timeout waiting for completion"}

        try:
            history_url = f"{comfyui_url}/history/{prompt_id}"
            async with session.get(history_url) as response:
                if response.status == 200:
                    history_data = await response.json()

                    if prompt_id in history_data:
                        prompt_info = history_data[prompt_id]
                        status_info = prompt_info.get("status", {})

                        if status_info.get("status_str") == "success" and status_info.get("completed", False):
                            outputs = prompt_info.get("outputs", {})

                            if save_node_id and save_node_id in outputs:
                                images = outputs[save_node_id].get("images", [])
                                if images:
                                    return {
                                        "status": "complete",
                                        "image_info": images[0],
                                    }

                            for node_id, node_output in outputs.items():
                                if "images" in node_output and node_output["images"]:
                                    return {
                                        "status": "complete",
                                        "image_info": node_output["images"][0],
                                    }

                            return {
                                "status": "failed",
                                "message": "No image output found in completed prompt",
                            }

                        elif status_info.get("status_str") == "error":
                            return {
                                "status": "failed",
                                "message": f"Prompt execution failed: {status_info.get('error', 'Unknown error')}",
                            }
        except Exception:
            pass

        await asyncio.sleep(check_interval)


async def download_image_as_numpy(
    session: aiohttp.ClientSession,
    comfyui_url: str,
    image_info: Dict,
) -> Optional[np.ndarray]:
    """Download image from ComfyUI and return as RGB numpy array."""
    try:
        view_url = f"{comfyui_url}/view"
        params = {"filename": image_info["filename"]}

        async with session.get(view_url, params=params) as response:
            if response.status == 200:
                img_data = await response.read()
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error downloading image from ComfyUI: {e}")
    return None


async def colorize_single_image_comfyui(
    session: aiohttp.ClientSession,
    comfyui_url: str,
    image: np.ndarray,
    image_index: int,
    positive_prompt: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Process a single image through ComfyUI and return colorized numpy array."""
    upload_result = await upload_image_to_comfyui(session, comfyui_url, image, image_index)
    if upload_result["status"] != "success":
        logger.error(f"ComfyUI upload failed for image {image_index}: {upload_result.get('message')}")
        return None

    uploaded_filename = upload_result["filename"]
    client_id = str(uuid.uuid4())
    unique_prefix = f"colorize_{image_index}_{str(uuid.uuid4())[:8]}"

    workflow = create_comfyui_colorize_workflow(
        image_name=uploaded_filename,
        output_prefix=unique_prefix,
        positive_prompt=positive_prompt,
    )

    inference_result = await send_inference_request(session, comfyui_url, workflow, client_id)
    if inference_result["status"] != "success":
        logger.error(f"ComfyUI inference failed for image {image_index}: {inference_result.get('message')}")
        return None

    completion_result = await wait_for_completion(
        session, comfyui_url, inference_result["prompt_id"]
    )
    if completion_result["status"] != "complete":
        logger.error(f"ComfyUI completion failed for image {image_index}: {completion_result.get('message')}")
        return None

    result_image = await download_image_as_numpy(session, comfyui_url, completion_result["image_info"])
    if result_image is None:
        logger.error(f"ComfyUI download failed for image {image_index}")
        return None

    logger.info(f"ComfyUI colorized image {image_index} successfully")
    return result_image


async def colorize_batch_comfyui(
    images: List[np.ndarray],
    comfyui_url: str = "http://localhost:9010",
    positive_prompt: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Colorize a batch of RGB numpy images using ComfyUI.
    Images are sent sequentially to a single ComfyUI instance.

    Returns list of colorized RGB numpy arrays (original returned on failure).
    """
    if not images:
        return []

    if positive_prompt is None:
        positive_prompt = "colorize this manga image"

    results = [None] * len(images)

    async with aiohttp.ClientSession() as session:
        for idx, img in enumerate(images):
            colorized = await colorize_single_image_comfyui(
                session, comfyui_url, img, idx, positive_prompt
            )
            results[idx] = colorized if colorized is not None else img

    return results
