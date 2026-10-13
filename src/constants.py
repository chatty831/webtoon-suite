import os
import re

import torch
from dotenv import load_dotenv

from src.scraper.imhen import imhen_scraper
from src.scraper.one_punch import one_punch_scraper
from src.scraper.wfwf import wfwf_scraper

load_dotenv()

DEVICE = torch.device("cuda")
DTYPE = torch.float32

# Exact domain -> scraper
DOMAIN_MAPS = {
    "imhentai.xxx": imhen_scraper,
    "onepunchmanmangaa.com": one_punch_scraper,
}

# Regex pattern -> scraper (checked when exact match fails)
DOMAIN_PATTERN_MAPS = [
    (re.compile(r"^wfwf\d+\.com$"), wfwf_scraper),
]


def resolve_scraper(domain: str):
    """Resolve a domain to its scraper. Exact match first, then patterns."""
    if domain in DOMAIN_MAPS:
        return DOMAIN_MAPS[domain]
    for pattern, scraper in DOMAIN_PATTERN_MAPS:
        if pattern.match(domain):
            return scraper
    return None

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")
AZURE_SPEECH_API_KEY = os.environ.get("AZURE_SPEECH_API_KEY")

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_STT_URL = os.environ.get("SARVAM_STT_URL")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SKYLARK_API_KEY = os.environ.get("SKYLARK_API_KEY")
SKYLARK_BASE_URL = os.environ.get("SKYLARK_BASE_URL")

NANO_API_KEY = os.environ.get("NANO_API_KEY")

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8001")
