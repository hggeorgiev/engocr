"""Image generation for sketches that resist code reconstruction.

When a diagram/sketch can't be expressed as Mermaid/TikZ (venn, freehand,
unclear structure), an image-generation model redraws it as a clean
digital diagram. The model sees the *cropped sketch region* plus an
instruction — much higher fidelity than a text-only prompt.

Provider = gemini (default, image-input capable) | openai (gpt-image-1;
edit endpoint when a source image is given). Configured via
IMAGE_GEN_PROVIDER / IMAGE_GEN_MODEL.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Protocol

from PIL import Image

from engocr.logging import get_logger
from engocr.providers.base import _require

_logger = get_logger(__name__)

GEMINI_IMAGE_GEN_MODEL = "gemini-2.5-flash-image"   # override: IMAGE_GEN_MODEL
OPENAI_IMAGE_GEN_MODEL = "gpt-image-1"

_INSTRUCTION = (
    "Recreate this hand-drawn sketch as a clean, precise digital diagram. "
    "It depicts: {description}\n\n"
    "Requirements: white background, crisp lines, print-ready, preserve all "
    "labels and annotations as legible text, keep the structure and spatial "
    "relationships faithful to the sketch. No decorative elements."
)


def build_gen_prompt(description: str) -> str:
    return _INSTRUCTION.format(description=description.strip())


def crop_from_bbox(page_image: Image.Image, bbox: list[float]) -> Image.Image:
    """Crop a region from a page image using normalized coordinates.

    Returns the full page image if the bbox covers the entire page or
    would result in a zero-area crop.
    """
    if len(bbox) != 4 or all(abs(v - d) < 0.01 for v, d in
                             zip(bbox, [0.0, 0.0, 1.0, 1.0])):
        return page_image
    w, h = page_image.size
    x0 = max(0, int(bbox[0] * w))
    y0 = max(0, int(bbox[1] * h))
    x1 = min(w, int(bbox[2] * w))
    y1 = min(h, int(bbox[3] * h))
    if x1 <= x0 or y1 <= y0:
        return page_image
    return page_image.crop((x0, y0, x1, y1))


class ImageGenProvider(Protocol):
    """Generate a clean diagram image from a sketch + instruction."""

    def generate(self, prompt: str,
                 source_image: Image.Image | None = None) -> Image.Image: ...


class GeminiImageGen:
    """Image generation via google-genai (accepts a source sketch image)."""

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        _require("gemini-image", self.api_key, "api.gemini_key / GEMINI_API_KEY")
        self.model = model or os.environ.get("IMAGE_GEN_MODEL", "") \
            or GEMINI_IMAGE_GEN_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str,
                 source_image: Image.Image | None = None) -> Image.Image:
        from google.genai import types

        contents: list = [prompt]
        if source_image is not None:
            contents.append(source_image.convert("RGB"))
        response = self._get_client().models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                # No tools are registered — disable AFC explicitly.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True),
            ),
        )
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                inline = getattr(part, "inline_data", None)
                if inline is not None and inline.data:
                    return Image.open(io.BytesIO(inline.data)).convert("RGB")
        raise RuntimeError("image generation returned no image data")


class OpenAIImageGen:
    """Image generation via OpenAI (gpt-image-1; edit with a source image)."""

    def __init__(self, model: str | None = None):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        _require("openai-image", self.api_key, "api.openai_key / OPENAI_API_KEY")
        self.model = model or os.environ.get("IMAGE_GEN_MODEL", "") \
            or OPENAI_IMAGE_GEN_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str,
                 source_image: Image.Image | None = None) -> Image.Image:
        images = self._get_client().images
        if source_image is not None:
            buf = io.BytesIO()
            source_image.convert("RGB").save(buf, format="PNG")
            buf.seek(0)
            buf.name = "sketch.png"
            response = images.edit(model=self.model, image=buf, prompt=prompt)
        else:
            response = images.generate(model=self.model, prompt=prompt)
        b64 = response.data[0].b64_json
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# name -> builder
_MODULES = {
    "gemini": GeminiImageGen,
    "openai": OpenAIImageGen,
}


def make_image_gen_provider(name: str | None = None) -> ImageGenProvider:
    """Build the configured image-generation provider (IMAGE_GEN_PROVIDER)."""
    if name is None:
        name = os.environ.get("IMAGE_GEN_PROVIDER", "gemini")
    name = (name or "").strip().lower()
    if name not in _MODULES:
        raise RuntimeError(
            f"Unknown image generation provider: {name!r}. Valid: "
            + ", ".join(sorted(_MODULES)))
    return _MODULES[name]()


__all__ = [
    "GeminiImageGen",
    "ImageGenProvider",
    "OpenAIImageGen",
    "build_gen_prompt",
    "crop_from_bbox",
    "make_image_gen_provider",
]
