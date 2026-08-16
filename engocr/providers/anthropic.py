"""Anthropic Claude vision provider (anthropic SDK, messages API)."""

from __future__ import annotations

import base64
import io
import os

from PIL import Image

from engocr.providers.base import _require


def _b64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class AnthropicVisionProvider:
    """Sends a page image to Claude; returns JSON text.

    No native JSON mode — the prompt enforces JSON and the extractor strips
    ``` fences from the response.
    """

    default_model = "claude-sonnet-4-5"
    max_tokens = 4096

    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _require("anthropic", self.api_key,
                 "api.anthropic_key / ANTHROPIC_API_KEY")
        self.model = os.environ.get("VISION_MODEL", "") or self.default_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def analyze(self, prompt: str, image: Image.Image) -> str:
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _b64_png(image),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        return "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        ) or ""


def build_provider() -> AnthropicVisionProvider:
    return AnthropicVisionProvider()
