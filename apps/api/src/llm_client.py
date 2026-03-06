"""OpenAI-compatible client factory bound to the configured gateway."""

import os
from openai import OpenAI
from .settings import settings

def get_client() -> OpenAI:
    # The Gateway is OpenAI-compatible; we just point base_url at it.
    return OpenAI(
        api_key=settings.gateway_api_key,
        base_url=settings.llm_gateway_base_url,
    )
