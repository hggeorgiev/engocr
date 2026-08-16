"""Vision provider tests: factory resolution, request shapes (mocked SDKs),
and key/endpoint validation. No real API calls."""

from types import SimpleNamespace

import pytest
from PIL import Image

from engocr.extractor import VisionExtractor
from engocr.providers.anthropic import (
    AnthropicVisionProvider,
)
from engocr.providers.base import make_provider
from engocr.providers.gemini import (
    GeminiVisionProvider,
)
from engocr.providers.openai_compat import (
    AzureVisionProvider,
    MistralVisionProvider,
    OllamaVisionProvider,
    OpenAIVisionProvider,
    OpenRouterVisionProvider,
    QwenVisionProvider,
    XAIVisionProvider,
)

IMG = Image.new("RGB", (4, 4))


def _img(img=IMG):
    return img


# ── factory / config validation ───────────────────────

def test_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="Unknown vision provider"):
        make_provider("does-not-exist")


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="gemini_key"):
        make_provider("gemini")


def test_gemini_builds_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(make_provider("gemini"), GeminiVisionProvider)


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="openai_key"):
        make_provider("openai")


def test_ollama_needs_no_key(monkeypatch):
    for var in ("DASHSCOPE_API_KEY", "MISTRAL_API_KEY",
                "OPENROUTER_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(make_provider("ollama"), OllamaVisionProvider)


def test_azure_requires_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="azure_endpoint"):
        make_provider("azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    with pytest.raises(RuntimeError, match="vision_model"):
        make_provider("azure")


def test_default_provider_from_env(monkeypatch):
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(make_provider(), OpenAIVisionProvider)


# ── OpenAI-compatible request shape ───────────────────

class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def _mock_openai_client(provider):
    fake = FakeOpenAIClient()
    provider._client = fake
    return fake.chat.completions


def test_openai_compat_request_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    provider = OpenAIVisionProvider()
    completions = _mock_openai_client(provider)

    provider.analyze("the prompt", _img())

    call = completions.calls[0]
    assert call["model"] == "gpt-4o"
    assert call["response_format"] == {"type": "json_object"}
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "the prompt"
    img = content[1]
    assert img["type"] == "image_url"
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    assert img["image_url"]["detail"] == "high"


@pytest.mark.parametrize("cls,env,default,base_url", [
    (QwenVisionProvider, "DASHSCOPE_API_KEY", "qwen-vl-max",
     "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    (MistralVisionProvider, "MISTRAL_API_KEY", "pixtral-large-latest",
     "https://api.mistral.ai/v1"),
    (OpenRouterVisionProvider, "OPENROUTER_API_KEY",
     "qwen/qwen2.5-vl-72b-instruct", "https://openrouter.ai/api/v1"),
    (XAIVisionProvider, "XAI_API_KEY", "grok-2-vision-1212",
     "https://api.x.ai/v1"),
])
def test_compat_provider_configs(monkeypatch, cls, env, default, base_url):
    monkeypatch.setenv(env, "k")
    provider = cls()
    assert provider.model == default
    assert provider.base_url == base_url
    completions = _mock_openai_client(provider)
    provider.analyze("p", _img())
    assert completions.calls[0]["model"] == default


def test_ollama_base_url_default_and_override(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert OllamaVisionProvider().base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu:11434/v1")
    assert OllamaVisionProvider().base_url == "http://gpu:11434/v1"


def test_azure_uses_azure_client(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("VISION_MODEL", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")

    captured = {}

    class FakeAzureOpenAI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import openai
    monkeypatch.setattr(openai, "AzureOpenAI", FakeAzureOpenAI)

    provider = AzureVisionProvider()
    assert provider.model == "my-deployment"
    provider.analyze("p", _img())

    assert captured["kwargs"] == {
        "api_key": "k",
        "azure_endpoint": "https://x.openai.azure.com",
        "api_version": "2025-01-01",
    }


# ── Anthropic request shape ───────────────────────────

def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    provider = AnthropicVisionProvider()

    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[
                SimpleNamespace(type="text", text='{"page_summary": "ok"}')])

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    result = provider.analyze("the prompt", _img())

    assert calls[0]["model"] == "claude-sonnet-4-5"
    content = calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "the prompt"}
    assert result == '{"page_summary": "ok"}'


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="anthropic_key"):
        make_provider("anthropic")


# ── Extractor uses the provider ───────────────────────

def test_extractor_routes_through_provider_and_parses():
    class FakeProvider:
        def analyze(self, prompt, image):
            return '{"page_summary": "a page", "text_elements": []}'

    extractor = VisionExtractor(provider=FakeProvider())
    result = extractor.extract_page(_img())
    assert result.page_summary == "a page"
