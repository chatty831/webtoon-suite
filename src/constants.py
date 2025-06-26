import os

import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda")
DTYPE = torch.float32

DOMAIN_MAPS = {
    "imhentai.xxx": "",
}

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

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8001")
