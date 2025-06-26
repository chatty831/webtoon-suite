import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

from src.constants import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT

# Initialize model
AZURE_OPENAI_GPT_MODEL = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.0,
    model_kwargs={"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0},
)

GPT_MODEL_JSON = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.0,
    model_kwargs={
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": {"type": "json_object"},
    },
)

GPT_4o_MODEL = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1-mini",
    temperature=0.0,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1,
)

GPT_4v_MODEL = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4v",
    temperature=0.0,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1,
)

# GPT_MODEL = ChatVertexAI(model="gemini-2.0-flash-lite", temperature=0, max_retries=1, stop=None)

AZURE_OPENAI_GPT_MODEL = AzureChatOpenAI(
    openai_api_type="azure",
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment="gpt-4o-mini",
    temperature=0,
)

GPT_MODEL_JSON = AzureChatOpenAI(
    openai_api_type="azure",
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment="gpt-4o-mini",
    temperature=0.0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={"type": "json_object"},
)

# embedding model
embedding_model = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    api_key=AZURE_OPENAI_API_KEY,  # os.getenv("AZURE_OPENAI_KEY_US"),
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    # os.getenv("AZURE_OPENAI_ENDPOINT_US"),
    azure_deployment="ada-text-embedding",
)
