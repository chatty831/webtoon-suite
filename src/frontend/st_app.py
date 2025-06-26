import base64
import json
import os
import re
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import requests
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from streamlit_shortcuts import add_keyboard_shortcuts, button

from ..constants import FASTAPI_URL

WIDTH = 750

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
        "images": None,
        "translated_images": None,
        "page": 0,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def update_display_state(new_images: List[str]):
    """Update session state when new images are loaded."""
    if st.session_state["prev_images"] != new_images:
        st.session_state["prev_images"] = new_images
        if "page" in st.session_state:
            del st.session_state["page"]  # Reset pagination


def navigate_page(direction: int):
    """Navigate between pages in paginated mode."""
    st.session_state["page"] += direction
    st.rerun()


# ============================================================================
# Image Display Functions
# ============================================================================


def render_single_image(image_base64: Optional[str], width: int = WIDTH):
    """Render a single image or placeholder."""
    if image_base64:
        st.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="data:image/jpeg;base64,{image_base64}" width="{width}">'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        render_image_placeholder(width)


def render_image_placeholder(width: int):
    """Render a placeholder for missing images."""
    st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<div style="width: {width}px; height: {width}px; border: 2px dashed #ccc; '
        f'display: flex; justify-content: center; align-items: center;">'
        f'<span style="font-size: 24px; color: #999;">🖼️❌</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def display_list_mode(images: List[str], width: int = WIDTH):
    """Display all images in a list format."""
    for img in images:
        render_single_image(img, width)


def display_paginated_mode(images: List[str], width: int = WIDTH):
    """Display images in paginated format with navigation."""
    if not images:
        return

    # Initialize page state if needed
    if "page" not in st.session_state:
        st.session_state["page"] = 0

    total_pages = len(images)
    current_page = st.session_state["page"]

    # Ensure current page is within bounds
    current_page = max(0, min(current_page, total_pages - 1))
    st.session_state["page"] = current_page

    # Page indicator
    st.markdown(
        f'<div style="text-align: center; margin-top: 10px;">' f"Page {current_page + 1} of {total_pages}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Display current image
    render_single_image(images[current_page], width)

    # Navigation controls
    if total_pages > 1:
        render_navigation_controls(current_page, total_pages)


def render_navigation_controls(current_page: int, total_pages: int):
    """Render previous/next navigation buttons."""
    col1, col2 = st.columns([1, 1])

    with col1:
        if current_page > 0:
            if button("⬅️ Previous", "ArrowLeft", on_click=lambda: True):
                navigate_page(-1)

    with col2:
        with stylable_container(key="next_button", css_styles="button { float: right; }"):
            if current_page < total_pages - 1:
                if button("Next ➡️", "ArrowRight", on_click=lambda: True):
                    navigate_page(1)


def display_images_center(base64_images: List[str], width: int = WIDTH):
    """Main function to display images with mode selection."""
    update_display_state(base64_images)

    # Display mode selection
    display_mode = st.radio("Select Display Mode", ["Paginated", "List"], horizontal=True)

    if display_mode == "List":
        display_list_mode(base64_images, width)
    else:
        display_paginated_mode(base64_images, width)


# ============================================================================
# File Processing Functions
# ============================================================================


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """Helper function for natural sorting of filenames."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def encode_uploaded_image_to_base64(uploaded_file) -> str:
    """Convert an uploaded image file to base64 encoded string."""
    try:
        file_bytes = uploaded_file.getvalue()
        return base64.b64encode(file_bytes).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image {uploaded_file.name}: {str(e)}")
        return ""


def process_uploaded_images(uploaded_files) -> List[str]:
    """Process multiple uploaded image files with natural sorting."""
    if not uploaded_files:
        return []

    # Create list of image info with metadata
    image_infos = [{"name": file.name, "file": file} for file in uploaded_files]

    # Sort images naturally
    image_infos.sort(key=lambda x: natural_sort_key(x["name"]))

    # Encode to base64
    return [encode_uploaded_image_to_base64(info["file"]) for info in image_infos]


# ============================================================================
# API Communication Functions
# ============================================================================


def process_images_api(images: List[str], translate: bool, colorize: bool, upscale: bool) -> Optional[List[str]]:
    """Send images to API for processing."""
    try:
        response = requests.post(
            url=f"{FASTAPI_URL}/process-images",
            json={"images": images},
            params={"translate": translate, "colorize": colorize, "upscale": upscale},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json().get("translated_images", [])
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


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


# ============================================================================
# File Download Functions
# ============================================================================


def create_images_zip(images: List[str]) -> BytesIO:
    """Create a zip file containing all images."""
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img_data in enumerate(images):
            img_bytes = base64.b64decode(img_data)
            zip_file.writestr(f"translated_image_{idx + 1}.jpg", img_bytes)

    zip_buffer.seek(0)
    return zip_buffer


# ============================================================================
# UI Component Functions
# ============================================================================


def render_file_uploader():
    """Render the file uploader component."""
    return st.file_uploader(
        "Select multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "webp", "bmp"]
    )


def render_control_panel():
    """Render the control panel with checkboxes and process button."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        process_uploaded = st.button("Process Images")

    with col2:
        st.session_state["translate"] = st.checkbox("Translate", value=st.session_state["translate"])

    with col3:
        st.session_state["colorize"] = st.checkbox("Colorize", value=st.session_state["colorize"])

    with col4:
        st.session_state["upscale"] = st.checkbox("Upscale", value=st.session_state["upscale"])

    return process_uploaded


def render_action_buttons():
    """Render download and process buttons."""
    actions = {}

    # Download button (if translated images exist)
    if st.session_state.get("translated_images"):
        if st.button("Download Images"):
            zip_buffer = create_images_zip(st.session_state["translated_images"])
            st.download_button(
                label="Download All Images as ZIP",
                data=zip_buffer,
                file_name="translated_images.zip",
                mime="application/zip",
            )

    # Process button
    actions["process"] = st.button("Process")

    return actions


def render_cache_delete_button():
    """Render the cache delete button."""
    if st.button("Delete Cache"):
        if delete_cache_api(st.session_state["images"]):
            st.success("Cache deleted successfully.")
        else:
            st.error("Failed to delete cache.")


# ============================================================================
# Main Application Logic
# ============================================================================


def handle_image_upload(uploaded_files):
    """Handle the processing of uploaded images."""
    st.session_state["translated_images"] = None
    st.session_state["images"] = process_uploaded_images(uploaded_files)
    st.success(f"Successfully processed {len(uploaded_files)} images")


def handle_image_processing():
    """Handle the API processing of images."""
    images = st.session_state["images"]
    translated = process_images_api(
        images, st.session_state["translate"], st.session_state["colorize"], st.session_state["upscale"]
    )

    if translated is not None:
        st.session_state["images"] = None
        st.session_state["translated_images"] = translated


def display_current_images():
    """Display either original or translated images based on state."""
    if st.session_state.get("images"):
        display_images_center(st.session_state["images"], width=WIDTH)
        render_cache_delete_button()
    elif st.session_state.get("translated_images"):
        display_images_center(st.session_state["translated_images"], width=WIDTH)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # File upload section
    uploaded_files = render_file_uploader()

    # Control panel
    process_uploaded = render_control_panel()

    # Handle uploaded files
    if process_uploaded and uploaded_files:
        handle_image_upload(uploaded_files)

    # Action buttons
    actions = render_action_buttons()

    # Handle processing
    if actions.get("process") and st.session_state.get("images"):
        handle_image_processing()

    # Display images
    display_current_images()


# Run the application
if __name__ == "__main__":
    main()
