"""Provider-agnostic image captioning.

Goes through the same provider abstraction as page extraction, so it
honors VISION_PROVIDER / VISION_MODEL like everything else. Providers
that force JSON output (gemini, openai-compatible) are asked for a small
{"caption": "..."} object; parsing is lenient (fences stripped, raw text
as fallback).
"""

from __future__ import annotations

import json

from PIL import Image

from engocr.config import VISION_RETRY_BASE_SECONDS, VISION_RETRY_MAX
from engocr.extractor import _call_with_retry
from engocr.logging import get_logger
from engocr.providers.base import VisionProvider, make_provider

_logger = get_logger(__name__)

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in rich detail. Include:\n"
    "- What is depicted (objects, people, scenes, text, diagrams)\n"
    "- Any visible text or labels (transcribe them)\n"
    "- The likely purpose or context (photograph, diagram, screenshot, "
    "chart, whiteboard, sketch, etc.)\n\n"
    "Return ONLY a JSON object: {\"caption\": \"<one dense paragraph>\"}"
)

_FALLBACK = "image"


def _parse_caption(raw: str) -> str:
    """Extract the caption text from a (possibly JSON) provider response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text or _FALLBACK
    if isinstance(data, dict):
        caption = data.get("caption", "")
        return str(caption).strip() or _FALLBACK
    return _FALLBACK


def caption_image(
    image: Image.Image,
    provider: str | VisionProvider | None = None,
    prompt: str | None = None,
) -> str:
    """Caption an image with the configured vision provider.

    provider: a provider name, an instance (DI seam for tests), or None
    for the configured default. prompt: override the caption prompt.
    Fail-soft: any provider/parse error returns "image".
    """
    try:
        p = make_provider(provider) if isinstance(provider, str) \
            else (provider or make_provider())
        raw = _call_with_retry(
            lambda: p.analyze(prompt or DEFAULT_CAPTION_PROMPT, image),
            max_attempts=VISION_RETRY_MAX,
            base_seconds=VISION_RETRY_BASE_SECONDS,
        )
        return _parse_caption(raw)
    except Exception as e:
        _logger.warning("image_captioning_failed: %s", e)
        return _FALLBACK
