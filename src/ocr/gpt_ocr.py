import asyncio
import base64
import os
import re
from typing import Optional
from aiolimiter import AsyncLimiter

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.logger import logger
from llm_models import AZURE_OPENAI_GPT_MODEL

SYS_PROMPT = """You are a multilingual manga localization expert with deep knowledge of various languages and cultures. Your task is to analyze dialogue images from manga, identify the source language, transcribe the original text, and create a high-quality English translation.

First, identify and transcribe the original text in whatever language it appears. Then, translate it into English, considering not just the literal meaning but the cultural context, character voice, and intended impact.

Your English translation must be:
1. Natural and fluid, as if originally written in English
2. Appropriate to the manga context and character speaking
3. Similar in length to the original to fit in the same speech bubble
4. True to the original intent and emotional tone

Remember that direct word-for-word translations often miss cultural nuances and idioms. You must understand what the dialogue is truly conveying in its original language and find the most authentic English equivalent. This may require significant adaptation rather than literal translation.

For specialized content:
- For humor: Preserve the joke's impact, even if the specific wordplay must change
- For emotional dialogue: Maintain the emotional weight and intensity
- For vulgar/erotic content: Translate with appropriate English vulgarity/sensuality that matches the original tone
- For cultural references: Find suitable English equivalents when direct translation would lose meaning

If the image contains no dialogue text (only contains art), or only shows trademarks/watermarks/sound effects, return an empty \\boxed{} translation.

Your output format must be:
Original Dialogue: <transcribed text in original language>
Language Identified: <language of original text>
Intent Analysis: <brief analysis of the dialogue's purpose and tone>
English Translation: <initial translation>

Rephrased Final answer
\\boxed{<the translated dialogue>}"""


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
        # Check if it can be decoded as base64
        base64.b64decode(base64_string)
        return True
    except Exception as e:
        logger.error(f"Invalid base64 string: {e}")
        return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ModelInferenceError),
)
async def gpt_ocr(
    base64_image: str, model=None, timeout: int = 30, extract_dialogue: bool = True
) -> Optional[str]:
    """
    Process an image with a LLM model and extract dialogue translation

    Args:
        base64_image: str,  # Base64 encoded image string
        model: Optional[Any] = None,  # The LLM model to use, defaults to the global GPT_MODEL
        timeout: int = 30,  # Timeout in seconds for the API call
        extract_dialogue: bool = True  # Whether to extract dialogue from the image

    Returns:
        Translated text or None if processing failed

    Raises:
        ImageProcessingError: If the image is invalid
        ModelInferenceError: If the model fails to process the request
        ResponseParsingError: If the response can't be parsed
    """
    async with rate_limiter:
        # Validate inputs
        if not base64_image:
            raise ImageProcessingError("Empty base64 image string provided")

        if not validate_base64_image(base64_image):
            raise ImageProcessingError("Invalid base64 image format")

        # Use provided model or fall back to global
        llm_model = model if model is not None else AZURE_OPENAI_GPT_MODEL

        if llm_model is None:
            raise ModelInferenceError("No LLM model provided or available")

        # Construct messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYS_PROMPT,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Translate the dialogue in this image into English.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        try:
            # Run inference with timeout
            output = await llm_model.ainvoke(messages)

            # Extract content from BaseMessage
            if hasattr(output, "content"):
                output_text = output.content

            # logger.info(f"Raw model output: {output_text[:100]}...")

            # Extract the translation between backticks if needed
            if extract_dialogue:
                try:
                    # Look for text between \boxed{}
                    output_text = rf"{output_text}"
                    matches = re.search(r"ed\{(.*?)\}", output_text, re.DOTALL)

                    if matches:
                        extracted_text = matches.group(1).strip()
                        logger.info(f"Extracted text between \\boxed{{}}: {extracted_text[:50]}...")
                        return extracted_text
                    else:
                        # If no \boxed{} found, log warning and return the whole text
                        logger.warning("No content found between \\boxed{}, returning full response")
                        return output_text.strip()
                except Exception as e:
                    raise ResponseParsingError(f"Failed to parse response: {e}")
            else:
                # Return the full response if not extracting backticks
                return output_text.strip()

        except asyncio.TimeoutError:
            logger.error(f"Model inference timed out after {timeout} seconds")
            raise ModelInferenceError(f"Inference timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise ModelInferenceError(f"Model inference failed: {e}")
