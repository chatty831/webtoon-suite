import streamlit as st 
import os
import requests
import sys
from streamlit_extras.stylable_container import stylable_container
from streamlit_shortcuts import add_keyboard_shortcuts, button
from typing import List, Optional

WIDTH = 750

# ============================================================================
# Image Display Functions
# ============================================================================

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


def display_current_images():
    """Display either original or translated images based on state."""
    if st.session_state.get("images"):
        # with action_button_cols[2]:
        #     render_cache_delete_button()
        display_images_center(st.session_state["images"], width=WIDTH)
    elif st.session_state.get("translated_images"):
        display_images_center(st.session_state["translated_images"], width=WIDTH)


