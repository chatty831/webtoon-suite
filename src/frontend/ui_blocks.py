import streamlit as st 
import requests
from typing import List, Union, Optional, Dict, Any
import re
import os
import sys
from src.frontend.api_calls import delete_cache_api
from src.frontend.file_process_utils import create_images_zip

# ============================================================================
# UI Component Functions
# ============================================================================


def render_file_uploader():
    """Render the file uploader component."""
    return st.file_uploader(
        "Select multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "webp", "bmp"]
    )


def render_url_input_section():
    """Render URL input section with textbox and button."""
    st.markdown("---")
    st.subheader("Or scrape images from URL")

    col1, col2 = st.columns([3, 1])

    with col1:
        url_input = st.text_input(
            "Enter URL to scrape images from:",
            value=st.session_state.get("url_input", ""),
            placeholder="https://example.com/manga-chapter",
            key="url_input",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align button
        scrape_button = st.button("Scrape Images", type="secondary")

    return url_input, scrape_button


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

    if st.session_state["colorize"]:
        col_method, col_url = st.columns(2)

        with col_method:
            st.selectbox(
                "Colorize Method",
                options=["GAN", "ComfyUI"],
                index=0,
                key="colorize_method",
            )

        with col_url:
            if st.session_state.get("colorize_method") == "ComfyUI":
                st.text_input(
                    "ComfyUI URL",
                    value=st.session_state.get("comfyui_url", "http://localhost:9010"),
                    key="comfyui_url",
                )

    return process_uploaded


def render_cache_delete_button():
    """Render the cache delete button."""
    if st.button("Delete Cache"):
        if delete_cache_api(st.session_state["images"]):
            st.success("Cache deleted successfully.")
        else:
            st.error("Failed to delete cache.")


def render_action_buttons():
    """Render download and process buttons."""
    actions = {}

    action_button_cols = st.columns(3)

    with action_button_cols[0]:
        # Download button (if translated images exist)
        if st.button("Download Images"):
            if st.session_state.get("translated_images"):
                zip_buffer = create_images_zip(st.session_state["translated_images"])
                st.download_button(
                    label="Download All Images as ZIP",
                    data=zip_buffer,
                    file_name="translated_images.zip",
                    mime="application/zip",
                )

    with action_button_cols[1]:
        # Process button
        actions["process"] = st.button("Process")

    with action_button_cols[2]:
        if st.session_state.get("images"):
            render_cache_delete_button()

    return actions
