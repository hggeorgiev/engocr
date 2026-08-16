"""Vision providers: send (prompt, page image) to a model API and return raw
JSON text. The VisionExtractor (engocr.extractor) owns prompt building,
response parsing, and rate-limit retry; a provider is the only per-API part.
"""

from __future__ import annotations

import base64
import importlib
import io
import os
from typing import Protocol

from PIL import Image


class VisionProvider(Protocol):
    """Send a page image to a vision model; return the raw JSON text."""

    def analyze(self, prompt: str, image: Image.Image) -> str: ...


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URL (OpenAI-compatible APIs)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _require(name: str, value: str, hint: str) -> None:
    if not value:
        raise RuntimeError(
            f"Vision provider '{name}' needs {hint}. Set it in "
            f"~/.config/engocr/config.json or the environment.")


# name -> (module, build function)
_MODULES = {
    "gemini": ("gemini", "build_provider"),
    "anthropic": ("anthropic", "build_provider"),
    # the rest share the OpenAI-compatible / Azure implementation
    "openai": ("openai_compat", "build_openai"),
    "qwen": ("openai_compat", "build_qwen"),
    "mistral": ("openai_compat", "build_mistral"),
    "openrouter": ("openai_compat", "build_openrouter"),
    "ollama": ("openai_compat", "build_ollama"),
    "xai": ("openai_compat", "build_xai"),
    "azure": ("openai_compat", "build_azure"),
}


def make_provider(name: str | None = None) -> VisionProvider:
    """Build the configured vision provider (default: VISION_PROVIDER)."""
    if name is None:
        name = os.environ.get("VISION_PROVIDER", "gemini")
    name = (name or "").strip().lower()
    if name not in _MODULES:
        raise RuntimeError(
            f"Unknown vision provider: {name!r}. Valid: "
            + ", ".join(sorted(_MODULES)))
    module_name, builder = _MODULES[name]
    module = importlib.import_module(f"engocr.providers.{module_name}")
    return getattr(module, builder)()


__all__ = ["VisionProvider", "image_to_data_url", "make_provider", "_require"]
