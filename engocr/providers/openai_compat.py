"""OpenAI-compatible vision providers (chat.completions + base64 image).

One parameterized implementation serves OpenAI, Qwen (DashScope), Mistral,
OpenRouter, xAI, and local Ollama; Azure OpenAI uses the same request shape
through its own client constructor.
"""

from __future__ import annotations

import os

from PIL import Image

from engocr.providers.base import (
    _require,
    image_to_data_url,
)


class _OpenAICompatProvider:
    """Vision via any OpenAI-compatible chat.completions endpoint."""

    name = "openai"
    base_url: str | None = None
    default_model = ""
    key_env = ""
    config_key = ""

    def __init__(self):
        if self.key_env:
            key = os.environ.get(self.key_env, "")
            _require(self.name, key,
                     f"api.{self.config_key} / {self.key_env}")
            self.api_key = key
        else:
            self.api_key = ""
        self.model = os.environ.get("VISION_MODEL", "") or self.default_model
        self._client = None

    def _new_client(self):
        import openai
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return openai.OpenAI(**kwargs)

    def _get_client(self):
        if self._client is None:
            self._client = self._new_client()
        return self._client

    def analyze(self, prompt: str, image: Image.Image) -> str:
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": image_to_data_url(image), "detail": "high"}},
                ]},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class OpenAIVisionProvider(_OpenAICompatProvider):
    name = "openai"
    default_model = "gpt-4o"
    key_env = "OPENAI_API_KEY"
    config_key = "openai_key"


class QwenVisionProvider(_OpenAICompatProvider):
    name = "qwen"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen-vl-max"
    key_env = "DASHSCOPE_API_KEY"
    config_key = "qwen_key"


class MistralVisionProvider(_OpenAICompatProvider):
    name = "mistral"
    base_url = "https://api.mistral.ai/v1"
    default_model = "pixtral-large-latest"
    key_env = "MISTRAL_API_KEY"
    config_key = "mistral_key"


class OpenRouterVisionProvider(_OpenAICompatProvider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    default_model = "qwen/qwen2.5-vl-72b-instruct"
    key_env = "OPENROUTER_API_KEY"
    config_key = "openrouter_key"


class OllamaVisionProvider(_OpenAICompatProvider):
    name = "ollama"
    default_model = "qwen2.5vl:7b"
    key_env = ""  # local — no key

    def __init__(self):
        super().__init__()
        self.base_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1")


class XAIVisionProvider(_OpenAICompatProvider):
    name = "xai"
    base_url = "https://api.x.ai/v1"
    default_model = "grok-2-vision-1212"
    key_env = "XAI_API_KEY"
    config_key = "xai_key"


class AzureVisionProvider(_OpenAICompatProvider):
    """Azure OpenAI: same chat.completions shape, AzureOpenAI client."""

    name = "azure"
    default_model = ""  # deployment name comes from api.vision_model

    def __init__(self):
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_version = os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-10-21")
        _require("azure", self.api_key, "api.azure_key / AZURE_OPENAI_API_KEY")
        _require("azure", self.endpoint,
                 "api.azure_endpoint / AZURE_OPENAI_ENDPOINT")
        _require("azure", os.environ.get("VISION_MODEL", ""),
                 "api.vision_model (your deployment name)")
        self.model = os.environ.get("VISION_MODEL", "")
        self._client = None

    def _new_client(self):
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )


def build_openai() -> OpenAIVisionProvider:
    return OpenAIVisionProvider()


def build_qwen() -> QwenVisionProvider:
    return QwenVisionProvider()


def build_mistral() -> MistralVisionProvider:
    return MistralVisionProvider()


def build_openrouter() -> OpenRouterVisionProvider:
    return OpenRouterVisionProvider()


def build_ollama() -> OllamaVisionProvider:
    return OllamaVisionProvider()


def build_xai() -> XAIVisionProvider:
    return XAIVisionProvider()


def build_azure() -> AzureVisionProvider:
    return AzureVisionProvider()
