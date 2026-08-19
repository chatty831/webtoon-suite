import json
import requests 
import streamlit as st
from typing import Generator, List, Tuple, Union, Optional, Dict, Any
import re
import os
import sys

from src.constants import FASTAPI_URL

# =======================================================================
# API Communication Functions
# ============================================================================


def process_images_api(
    images: List[str],
    translate: bool,
    colorize: bool,
    upscale: bool,
    colorize_method: str = "gan",
    comfyui_url: str = "http://localhost:9010",
) -> Optional[List[str]]:
    """Send images to API for processing."""
    try:
        response = requests.post(
            url=f"{FASTAPI_URL}/process-images",
            json={"images": images},
            params={
                "translate": translate,
                "colorize": colorize,
                "upscale": upscale,
                "colorize_method": colorize_method,
                "comfyui_url": comfyui_url,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json().get("translated_images", [])
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def process_images_stream_api(
    images: List[str],
    translate: bool,
    colorize: bool,
    upscale: bool,
    colorize_method: str = "gan",
    comfyui_url: str = "http://localhost:9010",
) -> Generator[Tuple[int, Optional[str]], None, None]:
    """
    Stream-process images via API. Yields (index, base64_image) tuples
    as each image completes processing on the backend.
    """
    try:
        response = requests.post(
            url=f"{FASTAPI_URL}/process-images-stream",
            json={"images": images},
            params={
                "translate": translate,
                "colorize": colorize,
                "upscale": upscale,
                "colorize_method": colorize_method,
                "comfyui_url": comfyui_url,
            },
            headers={"Content-Type": "application/json"},
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                yield data.get("index"), data.get("image")

    except requests.exceptions.RequestException as e:
        st.error(f"API Streaming Error: {str(e)}")


def delete_cache_api(images: List[str]) -> bool:
    """Delete cache via API."""
    try:
        response = requests.post(
            url=f"{FASTAPI_URL}/delete-cache",
            json={"images": images},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Cache deletion error: {str(e)}")
        return False


def scrape_images_api(url: str) -> Optional[List[str]]:
    """Scrape images from URL via API."""
    try:
        response = requests.get(
            url=f"{FASTAPI_URL}/scrape-images",
            params={"url": url},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["scraped_images"]
    except requests.exceptions.RequestException as e:
        st.error(f"Scraping Error: {str(e)}")
        return None

def colorizer_model_change_api(model_config_tab):
    """Handle the colorization model change."""
    option = st.session_state.get("colorization_model")
    if option:
        full_path = f"models/colorize_models/{option}"
        response = requests.get(url=f"{FASTAPI_URL}/colorize-model-change", params={"model_path": full_path})
        
        with model_config_tab:
            if response.ok:
                st.success(f"Colorization model changed to {option}")
            else:
                st.error(f"Failed to change colorization model: {response.text}")

def colorize_param_change_api(model_config_tab):
    """Handle the colorization parameter change."""
    denoise_sigma = st.session_state.get("denoise_sigma")
    image_tile_size = st.session_state.get("image_tile_size")

    response = requests.get(
        url=f"{FASTAPI_URL}/colorize-param-change",
        params={"denoise_sigma": denoise_sigma, "image_tile_size": image_tile_size},
    )

    with model_config_tab:
        if response.ok:
            st.success("Colorization parameters updated successfully")
        else:
            st.error(f"Failed to update colorization parameters: {response.text}")
