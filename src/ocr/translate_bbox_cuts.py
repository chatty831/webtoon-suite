import asyncio
import base64
import os
import traceback
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.logger import logger
from src.ml_models import FONT_MODEL_PATH
from src.ocr.gpt_ocr import batch_raw_ocr, contextual_translate

OCR_BATCH_SIZE = 5  # Number of bbox cutouts sent per LLM OCR call


def extract_cutout_to_base64(image: np.ndarray, bbox: List[float]) -> str:
    """Extract a region from the image based on bbox and convert to base64."""
    x, y, w, h = map(int, bbox)  # Convert to integers for slicing

    # Ensure coordinates are within image boundaries
    height, width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(width - x, w)
    h = min(height - y, h)

    # Extract the region
    cutout = image[y : y + h, x : x + w]

    # Convert to base64
    _, buffer = cv2.imencode(".jpg", cutout)
    base64_cutout = base64.b64encode(buffer).decode("utf-8")

    return base64_cutout


def get_font(font_path, font_size):
    """Retrieve the appropriate font based on the provided font path."""
    if font_path and os.path.exists(font_path):
        # Start with a reasonable default size
        pil_font_size = 20 if font_size is None else int(font_size * 20)
        try:
            return ImageFont.truetype(font_path, size=pil_font_size)
        except Exception as e:
            logger.error(f"Error loading custom font: {e}. Falling back to default.")
            return ImageFont.load_default()
    else:
        # Fall back to default font if no custom font is provided
        logger.warning("No valid font path provided. Using default font.")
        return ImageFont.load_default()


# Map common Unicode characters to ASCII-safe equivalents for font rendering
_UNICODE_REPLACEMENTS = {
    "\u2018": "'",   # '
    "\u2019": "'",   # '
    "\u201C": '"',   # "
    "\u201D": '"',   # "
    "\u2014": "--",  # —  em dash
    "\u2013": "-",   # –  en dash
    "\u2026": "...", # …  ellipsis
    "\u00A0": " ",   # non-breaking space
    "\u200B": "",    # zero-width space
    "\u00AB": '"',   # «
    "\u00BB": '"',   # »
    "\u2032": "'",   # ′  prime
    "\u2033": '"',   # ″  double prime
    "\u2010": "-",   # ‐  hyphen
    "\u2011": "-",   # ‑  non-breaking hyphen
    "\u2012": "-",   # ‒  figure dash
    "\u00B7": ".",   # ·  middle dot
    "\u2022": "*",   # •  bullet
    "\u25CF": "*",   # ●  black circle
    "\u2661": "<3",  # ♡
    "\u2665": "<3",  # ♥
    "\u266A": "~",   # ♪
    "\u266B": "~",   # ♫
    "\uFF01": "!",   # ！ fullwidth exclamation
    "\uFF1F": "?",   # ？ fullwidth question mark
    "\uFF0C": ",",   # ， fullwidth comma
    "\uFF0E": ".",   # ． fullwidth period
}


def sanitize_text_for_font(text: str) -> str:
    """Replace Unicode characters that may be missing from the manga font
    with ASCII-safe equivalents, then strip any remaining non-printable chars."""
    # Apply explicit replacements
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Normalize remaining Unicode (e.g. accented chars -> base + combining)
    # then drop anything that isn't printable ASCII or basic Latin-1
    cleaned = []
    for ch in unicodedata.normalize("NFKD", text):
        if ord(ch) < 128 and ch.isprintable():
            cleaned.append(ch)
        elif ord(ch) == 10:  # keep newlines
            cleaned.append(ch)
    return "".join(cleaned)


def draw_text_on_image(
    image: np.ndarray,
    text: str,
    bbox: List[float],
    allow_expansion: bool = True,
    expansion_percent: float = 0.0,
    font_size: float = None,
    font_path: str = FONT_MODEL_PATH,
) -> None:
    """
    Draw text on the image at the bbox location with white background.
    Advanced version with adaptive font sizing and optional bbox expansion.

    Args:
        image: The image to draw on (modified in-place)
        text: The text to draw
        bbox: The bounding box [x, y, width, height]
        allow_expansion: Whether to allow the bbox to expand slightly if needed
        expansion_percent: How much to allow expansion (0.1 = 10%)
        font_size: Optional specific font size to use (overrides automatic sizing)
        font_path: Path to a .ttf font file for rendering text with special characters
    """
    # Skip empty text
    if not text or text.strip() == "":
        logger.info(f"Skipping empty text for bbox {bbox}")
        return

    # Sanitize text to avoid missing-glyph boxes in the font
    text = sanitize_text_for_font(text)
    if not text or text.strip() == "":
        logger.info(f"Text empty after sanitization for bbox {bbox}")
        return

    # Convert bbox coordinates to integers for drawing
    x, y, w, h = map(int, map(float, bbox))

    # Store original bbox dimensions for potential expansion
    orig_x, orig_y, orig_w, orig_h = x, y, w, h

    # Make sure the bbox is within image boundaries
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        logger.warning(f"Invalid bbox dimensions after boundary checking: {[x, y, w, h]}")
        return

    # Calculate max possible expansion if allowed
    if allow_expansion:
        # Maximum expansion based on percentage
        max_expansion_w = int(w * expansion_percent)
        max_expansion_h = int(h * expansion_percent)

        # Ensure expansion doesn't go outside image bounds
        expand_left = min(max_expansion_w // 2, x)
        expand_right = min(max_expansion_w // 2, img_w - (x + w))
        expand_top = min(max_expansion_h // 2, y)
        expand_bottom = min(max_expansion_h // 2, img_h - (y + h))

        # Apply expansion
        x -= expand_left
        y -= expand_top
        w += expand_left + expand_right
        h += expand_top + expand_bottom

        logger.debug(f"Expanded bbox from {[orig_x, orig_y, orig_w, orig_h]} to {[x, y, w, h]}")

    # Create a white background patch as a PIL Image
    patch = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(patch)

    # Set text parameters
    padding = 5
    text_color = (0, 0, 0)  # Black text

    font = get_font(font_path, font_size)

    # Improved test function to check if text fits with current settings
    def test_text_fit(text, font_size, max_width, max_height):
        padding_space = 2 * padding
        available_width = max_width - padding_space
        available_height = max_height - padding_space

        # Create temporary font with test size
        try:
            test_font = (
                ImageFont.truetype(font_path, size=font_size)
                if font_path and os.path.exists(font_path)
                else ImageFont.load_default()
            )
        except Exception:
            test_font = ImageFont.load_default()

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            # Check if the word itself is too long for the available width
            try:
                # For newer PIL versions
                word_bbox = draw.textbbox((0, 0), word, font=test_font)
                word_width = word_bbox[2] - word_bbox[0]
            except AttributeError:
                # For older PIL versions
                word_width, _ = draw.textsize(word, font=test_font)

            # If a single word is too wide for the available width, the font size is too large
            if word_width > available_width:
                return [], False, 0

            test_line = current_line + " " + word if current_line else word

            # Get text size
            try:
                # For newer PIL versions
                line_bbox = draw.textbbox((0, 0), test_line, font=test_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_height = line_bbox[3] - line_bbox[1]
            except AttributeError:
                # For older PIL versions
                line_width, line_height = draw.textsize(test_line, font=test_font)

            if line_width <= available_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # This should not happen now that we check individual words
                    lines.append(word)
                    current_line = ""

        if current_line:
            lines.append(current_line)

        # Calculate total height needed
        line_height_with_spacing = int(line_height * 1.2)  # Slightly reduced spacing factor
        total_height = len(lines) * line_height_with_spacing

        # Check if the text fits vertically
        fits = total_height <= available_height

        return lines, fits, line_height_with_spacing

    # Initial font size estimation
    if font_size is not None:
        # Use provided font size
        actual_font_size = int(font_size * 20)  # Scale to reasonable PIL font size
    else:
        # Start with a reasonable size based on both width and height
        avg_char_pixels = 10  # Approximate pixels per character at font size 20
        max_chars_per_line = max(1, (w - 2 * padding) // avg_char_pixels)

        text_length = len(text)
        num_words = len(text.split())

        # Estimate lines based on characters and words
        estimated_lines = max(1, min(20, text_length // max(1, max_chars_per_line) + 1))

        # Get a rough estimate of the longest word length
        longest_word_length = max(len(word) for word in text.split()) if text.split() else 1

        # Calculate font size based on available height, width, and estimated lines
        height_based_size = int((h * 0.8) / estimated_lines)
        width_based_size = int((w * 0.8) / longest_word_length)

        # Take the minimum to ensure text fits in both dimensions
        actual_font_size = min(width_based_size, height_based_size)

        # Constrain to reasonable range (8 to 50 pixels for PIL)
        actual_font_size = min(50, max(8, actual_font_size))

    # Find the optimal font size
    max_iterations = 12  # Increased max iterations for better optimization
    iteration = 0
    optimal_lines = []
    optimal_line_height = 0

    # Don't optimize if font size is provided
    if font_size is not None:
        optimal_lines, fits, optimal_line_height = test_text_fit(text, actual_font_size, w, h)
        if not fits:
            logger.warning(f"Provided font size is too large for text to fit in bbox {[x, y, w, h]}")
    else:
        # Binary search for optimal font size - considering both width and height
        min_size = 6  # Reduced minimum size to accommodate very long words
        max_size = 60
        current_size = actual_font_size

        while iteration < max_iterations and max_size - min_size > 1:
            current_size = (min_size + max_size) // 2
            lines, fits, line_height = test_text_fit(text, current_size, w, h)

            if fits:
                # Save these results and try larger
                optimal_lines = lines
                optimal_line_height = line_height
                min_size = current_size
            else:
                # Try smaller
                max_size = current_size

            iteration += 1

        # If we couldn't find a good fit, use the smallest size
        if not optimal_lines:
            optimal_lines, _, optimal_line_height = test_text_fit(text, min_size, w, h)

        # Update the font with final size
        try:
            font = (
                ImageFont.truetype(font_path, size=min_size)
                if font_path and os.path.exists(font_path)
                else ImageFont.load_default()
            )
        except Exception:
            font = ImageFont.load_default()

    # Calculate vertical starting position to center the text block
    total_height = len(optimal_lines) * optimal_line_height
    start_y = (h - total_height) // 2

    # Ensure start_y is not too small
    start_y = max(padding, start_y)

    # Draw each line of text on the patch
    for i, line in enumerate(optimal_lines):
        text_y = start_y + i * optimal_line_height

        # Make sure we don't exceed the patch
        if text_y > h - padding:
            logger.warning(f"Text exceeds vertical space: line {i+1}/{len(optimal_lines)} cut off")
            break

        # Calculate horizontal position to center this line
        try:
            # For newer PIL versions
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            # For older PIL versions
            text_width, _ = draw.textsize(line, font=font)

        text_x = (w - text_width) // 2

        # Ensure text_x is not negative
        text_x = max(padding, text_x)

        # Draw text on the patch
        draw.text((text_x, text_y), line, fill=text_color, font=font)

    # Convert PIL image back to OpenCV format
    patch_cv = np.array(patch)
    # Convert RGB to BGR (PIL uses RGB, OpenCV uses BGR)
    patch_cv = cv2.cvtColor(patch_cv, cv2.COLOR_RGB2BGR)

    # Apply the patch to the original image
    try:
        image[y : y + h, x : x + w] = patch_cv
    except ValueError as e:
        logger.error(
            f"Error applying patch: {e}, bbox:{[x,y,w,h]}, patch shape:{patch_cv.shape}, image shape:{image.shape}"
        )


def process_images_with_translations(
    images: List[np.ndarray],
    translations: List[str],
    cutout_info: List[Tuple[int, List[float]]],
    allow_expansion: bool = True,
    font_size: float = None,
) -> List[str]:
    """
    Process images by adding translations as text on white patches.

    Args:
        images: List of original images as numpy arrays
        translations: List of translated text strings
        cutout_info: List of tuples containing (image_index, bbox)
        allow_expansion: Whether to allow bboxes to expand slightly for better text fitting
        font_size: Optional specific font size to use (overrides automatic sizing)

    Returns:
        List of base64 encoded processed images
    """
    logger.info("Creating copies of original images for modification.")
    modified_images = [img.copy() for img in images]

    # Process the results and modify images
    for (img_idx, bbox), text in zip(cutout_info, translations):
        # Skip empty translations
        if not text or text.strip() == "":
            logger.info(f"Skipping empty translation for image {img_idx}, bbox {bbox}")
            continue

        # Skip invalid image indices
        if img_idx < 0 or img_idx >= len(modified_images):
            logger.warning(f"Invalid image index {img_idx}. Skipping.")
            continue

        logger.info(
            f"Drawing text on image {img_idx} at bbox {bbox[:2]} + {bbox[2:]}: '{text[:30]}{'...' if len(text) > 30 else ''}'"
        )
        try:
            draw_text_on_image(
                modified_images[img_idx], text, bbox, allow_expansion=allow_expansion, font_size=font_size
            )
        except Exception as e:
            logger.error(f"Error drawing text on image {img_idx}: {e}")

    # Convert modified images to base64
    logger.info("Converting modified images to base64.")
    processed_base64_images = []

    for img in modified_images:
        try:
            # Ensure the image is in the correct format
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Append the numpy array directly
            processed_base64_images.append(img)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Add a placeholder array to maintain the correct number of items
            processed_base64_images.append(np.zeros_like(img, dtype=np.uint8))

    return processed_base64_images


def _image_to_base64(image: np.ndarray) -> str:
    """Convert a full numpy image to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


async def ocr_page(image: np.ndarray, bboxes: List[List[float]]) -> List[str]:
    """
    Phase 1: Extract raw OCR from all bboxes on a page using batched LLM calls.

    Cutouts are grouped into batches of OCR_BATCH_SIZE and sent concurrently.
    No rate-limiting — OCR is unbounded for maximum throughput.

    Returns:
        List of raw OCR strings, same length as bboxes.
    """
    if not bboxes:
        return []

    cutout_b64s = [extract_cutout_to_base64(image, bbox) for bbox in bboxes]

    # Fire all batches concurrently
    batch_tasks = []
    for start in range(0, len(cutout_b64s), OCR_BATCH_SIZE):
        batch = cutout_b64s[start : start + OCR_BATCH_SIZE]
        batch_tasks.append(asyncio.create_task(batch_raw_ocr(batch)))

    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    # Flatten results, handling errors per-batch
    raw_texts: List[str] = []
    for batch_idx, result in enumerate(batch_results):
        batch_start = batch_idx * OCR_BATCH_SIZE
        batch_size = min(OCR_BATCH_SIZE, len(cutout_b64s) - batch_start)
        if isinstance(result, Exception):
            logger.error(f"Batch OCR failed for batch {batch_idx}: {result}")
            raw_texts.extend([""] * batch_size)
        else:
            # Pad/truncate to expected batch size
            texts = list(result)
            if len(texts) < batch_size:
                texts.extend([""] * (batch_size - len(texts)))
            elif len(texts) > batch_size:
                texts = texts[:batch_size]
            raw_texts.extend(texts)

    logger.info(
        f"[OCR] Batch OCR complete: {sum(1 for t in raw_texts if t)}/{len(raw_texts)} "
        f"bboxes yielded text ({len(batch_tasks)} batches)"
    )
    return raw_texts


async def translate_and_render_page(
    image: np.ndarray,
    bboxes: List[List[float]],
    raw_texts: List[str],
    prev_page_raw_ocr: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Phase 2: Contextual translation + rendering for a single page.

    Sends the full page image + all raw OCR texts to the LLM for coherent
    translation, then renders translated text onto the image.

    Args:
        image: Page image as numpy array.
        bboxes: Detected text bounding boxes.
        raw_texts: Raw OCR results from ocr_page().
        prev_page_raw_ocr: Raw OCR from the previous page for context.

    Returns:
        Modified image with translated text rendered.
    """
    if not bboxes or not raw_texts:
        return image

    full_page_b64 = _image_to_base64(image)

    try:
        translations = await contextual_translate(
            full_page_base64=full_page_b64,
            raw_texts=raw_texts,
            prev_page_raw_ocr=prev_page_raw_ocr,
        )
    except Exception as e:
        logger.error(f"Contextual translation failed: {e}")
        traceback.print_exc()
        translations = [""] * len(raw_texts)

    logger.info(
        f"[Translate] {sum(1 for t in translations if t)}/{len(translations)} snippets translated"
    )

    cutout_info = [(0, bbox) for bbox in bboxes]
    [result_image] = process_images_with_translations(
        [image], translations, cutout_info, font_size=None
    )
    return result_image


async def translate_text_regions(
    images: List[np.ndarray],
    bbox_results: List[List],
    prev_page_raw_ocr: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], Optional[List[str]]]:
    """
    Convenience wrapper: batch OCR + contextual translation + rendering.

    Used by the batch endpoint.  The streaming endpoint calls ocr_page() and
    translate_and_render_page() directly for finer-grained concurrency control.

    Args:
        images: List of page images as numpy arrays.
        bbox_results: Per-image list of bboxes from YOLO detection.
        prev_page_raw_ocr: Raw OCR texts from the previous page for context.

    Returns:
        (translated_images, last_page_raw_ocr)
    """
    last_raw_ocr: Optional[List[str]] = None

    for local_idx, bboxes in enumerate(bbox_results):
        if not bboxes:
            continue

        raw_texts = await ocr_page(images[local_idx], bboxes)
        images[local_idx] = await translate_and_render_page(
            images[local_idx], bboxes, raw_texts, prev_page_raw_ocr
        )

        # Chain context: this page's OCR feeds the next page
        prev_page_raw_ocr = raw_texts
        last_raw_ocr = raw_texts

    return images, last_raw_ocr
