import asyncio
import base64
import re
from typing import Dict, List, Optional

from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm_models import DEEPSEEK_V4_1_FLASH_MODEL, MIMO_V2_6_PRO_MODEL
from src.logger import logger

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAW_OCR_SYS_PROMPT = """You are an expert OCR system for manga and comics. Your task is to accurately transcribe the text visible in the provided image cutout.

Rules:
1. Transcribe the text EXACTLY as it appears, preserving the original language.
2. If there are multiple lines, join them with a space.
3. If the image contains no readable text (only art, watermarks, or sound effects), return an empty \\bbox{}.
4. Do NOT translate - only transcribe the raw text.

Output format:
\\bbox{<transcribed text in original language>}"""

CONTEXTUAL_TRANSLATE_SYS_PROMPT = """You are a multilingual manga localization expert with deep knowledge of various languages and cultures. You will receive:
1. The full manga page image for visual context
2. A list of raw OCR text extracted from speech bubbles on this page

Your task is to:
1. Analyze the full page image to understand the scene, characters, and narrative flow
2. Determine the correct reading order of the provided text snippets (manga reads right-to-left, top-to-bottom)
3. Translate ALL snippets into natural, fluid English in one coherent pass

Translation guidelines:
- Natural and fluid, as if originally written in English
- Appropriate to the manga context and character speaking
- Similar in length to the original to fit in the same speech bubble
- True to the original intent and emotional tone
- For humor: Preserve the joke's impact, even if the specific wordplay must change
- For emotional dialogue: Maintain the emotional weight and intensity
- For vulgar/erotic content: Translate with appropriate English vulgarity/sensuality that matches the original tone
- For cultural references: Find suitable English equivalents when direct translation would lose meaning

CRITICAL: You MUST return EXACTLY the same number of \\bbox{} elements as provided in the input list. Each output \\bbox{} corresponds to the input \\bbox{} at the same position. If a text snippet is a sound effect or untranslatable, return an empty \\bbox{} for that position.

Output format (one per line, same count as input):
\\bbox{<translated text for snippet 1>}
\\bbox{<translated text for snippet 2>}
...
"""


rate_limiter = AsyncLimiter(max_rate=300, time_period=60)


class LLMInferenceError(Exception):
    """Base exception for LLM inference errors"""

    pass


class ImageProcessingError(LLMInferenceError):
    """Exception for image processing errors"""

    pass


class ModelInferenceError(LLMInferenceError):
    """Exception for model inference errors"""

    pass


class ResponseParsingError(LLMInferenceError):
    """Exception for response parsing errors"""

    pass


def validate_base64_image(base64_string: str) -> bool:
    """
    Validate if the string is a proper base64 encoded image
    """
    try:
        base64.b64decode(base64_string)
        return True
    except Exception as e:
        logger.error(f"Invalid base64 string: {e}")
        return False


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from thinking model responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_bbox_content(text: str) -> Optional[str]:
    """Extract content from a single \\bbox{...} block."""
    text = rf"{text}"
    match = re.search(r"\\bbox\{(.*?)\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_all_bbox_contents(text: str) -> List[str]:
    """Extract all \\bbox{...} contents from text, preserving order."""
    text = rf"{text}"
    matches = re.findall(r"\\bbox\{(.*?)\}", text, re.DOTALL)
    return [m.strip() for m in matches]


# ---------------------------------------------------------------------------
# Phase 1: Raw OCR (per-cutout, no translation)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ModelInferenceError),
)
async def raw_ocr(base64_image: str, model=None) -> str:
    """
    Extract raw text from a bbox cutout image using LLM vision.
    Returns the transcribed text in the original language (no translation).
    Defaults to MIMO v2.6 Pro for OCR tasks.
    """
    async with rate_limiter:
        if not base64_image:
            raise ImageProcessingError("Empty base64 image string provided")

        if not validate_base64_image(base64_image):
            raise ImageProcessingError("Invalid base64 image format")

        llm_model = model if model is not None else (MIMO_V2_6_PRO_MODEL or DEEPSEEK_V4_1_FLASH_MODEL)

        if llm_model is None:
            raise ModelInferenceError("No LLM model provided or available")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": RAW_OCR_SYS_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe the text in this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        try:
            output = await llm_model.ainvoke(messages)
            output_text = output.content if hasattr(output, "content") else str(output)
            output_text = _strip_thinking(output_text)

            extracted = _extract_bbox_content(output_text)
            if extracted is not None:
                return extracted
            # Fallback: return stripped text if no \bbox{} found
            logger.warning("No \\bbox{} in raw OCR output, returning full response")
            return output_text.strip()

        except asyncio.TimeoutError:
            raise ModelInferenceError("Raw OCR inference timed out")
        except ModelInferenceError:
            raise
        except Exception as e:
            logger.error(f"Error during raw OCR inference: {e}")
            raise ModelInferenceError(f"Raw OCR inference failed: {e}")


# ---------------------------------------------------------------------------
# Phase 2: Contextual translation (full page + all OCR texts)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ModelInferenceError),
)
async def contextual_translate(
    full_page_base64: str,
    raw_texts: List[str],
    prev_page_context: Optional[Dict] = None,
    model=None,
) -> List[str]:
    """
    Translate all raw OCR texts for a page in one coherent LLM call.

    Args:
        full_page_base64: Base64 encoded full manga page image
        raw_texts: List of raw OCR strings extracted from bboxes
        prev_page_context: Optional dict with previous page's texts:
            {"original": ["text1", ...], "translated": ["trans1", ...]}
        model: Optional LLM model override

    Returns:
        List of translated strings, same length as raw_texts
    """
    async with rate_limiter:
        if not raw_texts:
            return []

        llm_model = model if model is not None else DEEPSEEK_V4_1_FLASH_MODEL

        if llm_model is None:
            raise ModelInferenceError("No LLM model provided or available")

        # Build the input text block
        bbox_list = "\n".join(f"\\bbox{{{text}}}" for text in raw_texts)
        input_text = f"Here are the OCR-extracted text snippets from the speech bubbles on this page:\n\n{bbox_list}\n\nAnalyze the full page image for context, then translate each snippet into English. Return exactly {len(raw_texts)} \\bbox{{}} elements."

        # Add previous page context if available
        if prev_page_context:
            orig = prev_page_context.get("original", [])
            trans = prev_page_context.get("translated", [])
            if orig and trans:
                history_lines = []
                for o, t in zip(orig, trans):
                    history_lines.append(f"  Original: {o}")
                    history_lines.append(f"  Translated: {t}")
                history_block = "\n".join(history_lines)
                input_text = f"For narrative continuity, here is the dialogue from the previous page:\n{history_block}\n\n---\n\n{input_text}"

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": CONTEXTUAL_TRANSLATE_SYS_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{full_page_base64}"},
                    },
                ],
            },
        ]

        try:
            output = await llm_model.ainvoke(messages)
            output_text = output.content if hasattr(output, "content") else str(output)
            output_text = _strip_thinking(output_text)

            translations = _extract_all_bbox_contents(output_text)

            # Validate count matches
            if len(translations) != len(raw_texts):
                logger.warning(
                    f"Translation count mismatch: expected {len(raw_texts)}, got {len(translations)}. "
                    f"Padding/truncating to match."
                )
                # Pad with empty strings if too few, truncate if too many
                if len(translations) < len(raw_texts):
                    translations.extend([""] * (len(raw_texts) - len(translations)))
                else:
                    translations = translations[: len(raw_texts)]

            logger.info(f"Contextual translation completed: {len(translations)} snippets")
            return translations

        except asyncio.TimeoutError:
            raise ModelInferenceError("Contextual translation timed out")
        except ModelInferenceError:
            raise
        except Exception as e:
            logger.error(f"Error during contextual translation: {e}")
            raise ModelInferenceError(f"Contextual translation failed: {e}")
