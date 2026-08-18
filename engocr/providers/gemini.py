"""Google Gemini vision provider (google-genai SDK)."""

from __future__ import annotations

import os
import threading

from PIL import Image

from engocr.config import (
    GEMINI_VISION_MAX_OUTPUT_TOKENS,
    GEMINI_VISION_MODEL,
    GEMINI_VISION_TEMPERATURE,
)
from engocr.providers.base import _require


class GeminiVisionProvider:
    """Sends a page image to Gemini via google-genai; returns JSON text."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = GEMINI_VISION_TEMPERATURE,
        max_output_tokens: int = GEMINI_VISION_MAX_OUTPUT_TOKENS,
    ):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        _require("gemini", self.api_key, "api.gemini_key / GEMINI_API_KEY")
        self.model = model or os.environ.get("VISION_MODEL", "") \
            or GEMINI_VISION_MODEL
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._clients = threading.local()

    def _new_client(self):
        from google import genai
        return genai.Client(api_key=self.api_key)

    def _get_client(self):
        client = getattr(self._clients, "client", None)
        if client is None:
            client = self._new_client()
            self._clients.client = client
        return client

    def analyze(self, prompt: str, image: Image.Image) -> str:
        from google.genai import types

        generated = self._get_client().models.generate_content(
            model=self.model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                # No tools are registered — disable AFC explicitly (the SDK
                # warns on direct generate_content use with AFC enabled).
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True),
            ),
        )
        return generated.text or ""


def build_provider() -> GeminiVisionProvider:
    return GeminiVisionProvider()
