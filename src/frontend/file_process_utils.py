from typing import List, Union
import base64
import re
import streamlit as st
from io import BytesIO
import zipfile


# ============================================================================
# File Processing Functions
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