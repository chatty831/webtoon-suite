import base64
import json
import os
import re
import sys
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import requests
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_shortcuts import add_keyboard_shortcuts, button

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.config import DEFAULT_CONFIG
from src.frontend.image_utils import display_current_images
from src.frontend.file_process_utils import process_uploaded_images
from src.frontend.api_calls import (
    process_images_api,
    process_images_stream_api,
    delete_cache_api,
    scrape_images_api,
    colorizer_model_change_api,
    colorize_param_change_api,
)
from src.frontend.ui_blocks import (
    render_action_buttons,
    render_control_panel,
    render_file_uploader,
    render_url_input_section,
)
from src.frontend.image_utils import render_single_image
from src.frontend.model_config import (
    colorize_param_change,
    colorizer_model_change
)

# Page Configuration
st.set_page_config(layout="wide")
st.title("Manga Translate App")


# ============================================================================
# Session State Management
# ============================================================================


def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        "display_mode": "List",
        "prev_images": [],
        "translate": True,
        "colorize": True,
        "upscale": True,
        "colorize_method": "GAN",
        "comfyui_url": "http://localhost:9010",
        "images": None,
        "translated_images": None,
        "page": 0,
        "url_input": "",  # Added for URL input
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ============================================================================
# Main Application Logic
# ============================================================================


def handle_image_upload(uploaded_files):
    """Handle the processing of uploaded images."""
    st.session_state["translated_images"] = None
    st.session_state["images"] = process_uploaded_images(uploaded_files)
    st.success(f"Successfully processed {len(uploaded_files)} images")


def handle_url_scraping(url: str):
    """Handle the scraping of images from URL."""
    if not url:
        st.error("Please enter a valid URL")
        return

    with st.spinner("Scraping images from URL..."):
        scraped_images = scrape_images_api(url)

    if scraped_images:
        st.session_state["translated_images"] = None
        st.session_state["images"] = scraped_images
        st.success(f"Successfully scraped {len(scraped_images)} images from URL")
    else:
        st.error("Failed to scrape images from the provided URL")


def handle_image_processing():
    """Handle the API processing of images with progressive streaming display."""
    images = st.session_state["images"]
    colorize_method = st.session_state.get("colorize_method", "GAN").lower()
    comfyui_url = st.session_state.get("comfyui_url", "http://localhost:9010")

    n = len(images)
    results = [None] * n

    progress_bar = st.progress(0, text="Processing images...")
    image_placeholder = st.empty()
    completed = 0

    for index, b64_img in process_images_stream_api(
        images,
        st.session_state["translate"],
        st.session_state["colorize"],
        st.session_state["upscale"],
        colorize_method=colorize_method,
        comfyui_url=comfyui_url,
    ):
        if index is not None and b64_img is not None:
            results[index] = b64_img
        completed += 1
        progress_bar.progress(completed / n, text=f"Processed {completed}/{n} images")

        with image_placeholder.container():
            for img in results:
                if img:
                    render_single_image(img)

    progress_bar.empty()
    image_placeholder.empty()
    st.session_state["images"] = None
    st.session_state["translated_images"] = results


def tab1():

    # File upload section
    uploaded_files = render_file_uploader()

    # Control panel
    process_uploaded = render_control_panel()

    # Handle uploaded files
    if process_uploaded and uploaded_files:
        handle_image_upload(uploaded_files)

    # URL input section
    url_input, scrape_button = render_url_input_section()

    # Handle URL scraping
    if scrape_button:
        handle_url_scraping(url_input)

    # Action buttons
    actions = render_action_buttons()

    # Handle processing
    if actions.get("process") and st.session_state.get("images"):
        handle_image_processing()

    # Display images
    display_current_images()


def tab2(model_config_tab):
    # Model configuration section
    st.subheader("Model Configuration")
    colorizer_model_change(model_config_tab)
    colorize_param_change(model_config_tab)

    st.markdown("---")


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    image_proc_tab, model_config_tab = st.tabs(["Image Processing", "Model Configuration"])

    with image_proc_tab:
        tab1()
    with model_config_tab:
        tab2(model_config_tab)


# Run the application
if __name__ == "__main__":
    main()
