"""Image generation for non-reconstructable sketches + progress bar.
All with fakes — no API calls."""

import json

import pytest
from PIL import Image

import engocr.cli as cli
from engocr.convert import (
    _visual_block,
    convert_file,
    enhance_with_generated_images,
)
from engocr.extractor import VisionExtractor
from engocr.imagegen import (
    GeminiImageGen,
    OpenAIImageGen,
    build_gen_prompt,
    crop_from_bbox,
    make_image_gen_provider,
)
from engocr.progress import page_progress
from engocr.schema import DiagramElement, PageVisionResult

# ── crop / prompt helpers ─────────────────────────────


def test_crop_from_bbox_crops_normalized_region():
    img = Image.new("RGB", (100, 200), "white")
    crop = crop_from_bbox(img, [0.0, 0.0, 0.5, 0.5])
    assert crop.size == (50, 100)


def test_crop_from_bbox_full_or_degenerate_returns_page():
    img = Image.new("RGB", (100, 100))
    assert crop_from_bbox(img, [0.0, 0.0, 1.0, 1.0]) is img
    assert crop_from_bbox(img, [0.5, 0.5, 0.4, 0.4]) is img
    assert crop_from_bbox(img, []) is img


def test_build_gen_prompt_includes_description():
    prompt = build_gen_prompt("a venn diagram of sets A and B")
    assert "a venn diagram of sets A and B" in prompt
    assert "digital diagram" in prompt


# ── providers (mocked SDKs) ──────────────────────────


def test_make_image_gen_provider_unknown():
    with pytest.raises(RuntimeError, match="Unknown image generation"):
        make_image_gen_provider("nope")


def test_gemini_image_gen_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="gemini_key"):
        GeminiImageGen()


def test_gemini_image_gen_sends_sketch_and_prompt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    captured = {}

    png = Image.new("RGB", (4, 4), "red")
    buf = __import__("io").BytesIO()
    png.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["modalities"] = config.response_modalities
            part = type("P", (), {"inline_data": type("D", (), {
                "data": png_bytes})()})()
            content = type("C", (), {"parts": [part]})()
            return type("R", (), {
                "candidates": [type("C2", (), {"content": content})()]})()

    provider = GeminiImageGen()
    provider._client = type("C", (), {"models": FakeModels()})()

    sketch = Image.new("RGB", (8, 8))
    out = provider.generate(build_gen_prompt("a venn"), sketch)

    assert captured["model"] == "gemini-2.5-flash-image"
    assert captured["modalities"] == ["TEXT", "IMAGE"]
    sent = captured["contents"][1]
    assert isinstance(sent, Image.Image) and sent.size == sketch.size
    assert out.size == (4, 4)


def test_openai_image_gen_uses_edit_with_source(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    captured = {}

    png = Image.new("RGB", (4, 4), "blue")
    buf = __import__("io").BytesIO()
    png.save(buf, format="PNG")
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode()

    class FakeImages:
        def edit(self, model, image, prompt):
            captured["endpoint"] = "edit"
            captured["prompt"] = prompt
            return type("R", (), {"data": [type("D", (), {
                "b64_json": b64})()]})()

        def generate(self, model, prompt):
            captured["endpoint"] = "generate"
            return type("R", (), {"data": [type("D", (), {
                "b64_json": b64})()]})()

    provider = OpenAIImageGen()
    provider._client = type("C", (), {"images": FakeImages()})()

    provider.generate("p", Image.new("RGB", (8, 8)))
    assert captured["endpoint"] == "edit"
    provider.generate("p", None)
    assert captured["endpoint"] == "generate"


# ── enhancement pass ─────────────────────────────────


class FakeGenProvider:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, source_image=None):
        self.calls.append({"prompt": prompt, "source": source_image})
        return Image.new("RGB", (6, 6), "green")


def _page_with_elements():
    return PageVisionResult(diagram_elements=[
        DiagramElement(type="venn", description="two overlapping sets",
                       bbox_approx=[0.1, 0.1, 0.6, 0.6]),          # → generated
        DiagramElement(type="graph", description="has mermaid",
                       source="flowchart TD\n A --> B",
                       source_lang="mermaid"),                     # → skipped
        DiagramElement(type="other", description=""),              # → skipped
    ])


def test_enhance_generates_only_for_source_less_described(tmp_path):
    page = _page_with_elements()
    provider = FakeGenProvider()
    written = enhance_with_generated_images(
        [page], [Image.new("RGB", (20, 20))], provider, tmp_path, "scan")

    assert written == 1
    assert len(provider.calls) == 1
    assert "two overlapping sets" in provider.calls[0]["prompt"]
    assert provider.calls[0]["source"] is not None   # the crop was sent
    venn = page.diagram_elements[0]
    assert venn.generated_image == "scan_p1_diagram_1.png"
    assert (tmp_path / venn.generated_image).exists()
    assert page.diagram_elements[1].generated_image == ""


def test_enhance_failure_is_fail_soft(tmp_path):
    class Down:
        def generate(self, prompt, source_image=None):
            raise RuntimeError("quota")

    page = _page_with_elements()
    written = enhance_with_generated_images(
        [page], [Image.new("RGB", (20, 20))], Down(), tmp_path, "scan")
    assert written == 0
    assert page.diagram_elements[0].generated_image == ""


def test_visual_block_embeds_generated_image():
    block = _visual_block("venn", "two overlapping sets", "", "",
                          "scan_p1_diagram_1.png")
    assert "- **venn**: two overlapping sets" in block
    assert "![two overlapping sets](scan_p1_diagram_1.png)" in block


def test_visual_block_source_and_image_coexist():
    block = _visual_block("graph", "desc", "flowchart TD\n A --> B",
                          "mermaid", "img.png")
    assert "```mermaid" in block
    assert "![desc](img.png)" in block


# ── convert_file integration ─────────────────────────


def test_convert_file_gen_diagrams_end_to_end(tmp_path):
    payload = {
        "page_summary": "s",
        "diagram_elements": [{
            "type": "venn", "description": "two overlapping sets",
            "bbox_approx": [0.0, 0.0, 1.0, 1.0],
        }],
    }

    class FakeVision:
        def analyze(self, prompt, image):
            return json.dumps(payload)

    img_path = tmp_path / "notes.png"
    Image.new("RGB", (10, 10)).save(img_path)

    result = convert_file(
        VisionExtractor(provider=FakeVision()), img_path,
        gen_diagrams=True, image_gen=FakeGenProvider())

    assert result.error == ""
    venn = result.pages[0].diagram_elements[0]
    assert venn.generated_image == "notes_p1_diagram_1.png"
    assert (tmp_path / venn.generated_image).exists()
    assert "![two overlapping sets](notes_p1_diagram_1.png)" in result.markdown


def test_convert_file_gen_diagrams_off_by_default(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10)).save(img_path)

    class FakeVision:
        def analyze(self, prompt, image):
            return json.dumps({"diagram_elements": [
                {"type": "venn", "description": "sets"}]})

    result = convert_file(VisionExtractor(provider=FakeVision()), img_path)
    assert result.pages[0].diagram_elements[0].generated_image == ""
    assert not list(tmp_path.glob("*_diagram_*.png"))


# ── progress bar ─────────────────────────────────────


def test_progress_disabled_is_noop():
    with page_progress(total=3, desc="t", enabled=False) as bar:
        bar.update(2)   # must not raise


def test_convert_file_progress_param_accepted(tmp_path):
    img_path = tmp_path / "p.png"
    Image.new("RGB", (10, 10)).save(img_path)

    class FakeVision:
        def analyze(self, prompt, image):
            return '{"page_summary": "ok"}'

    result = convert_file(VisionExtractor(provider=FakeVision()), img_path,
                          progress=True)   # non-TTY in tests → silent
    assert result.error == ""


# ── CLI plumbing ─────────────────────────────────────


def test_cli_gen_diagrams_requires_configured_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("IMAGE_GEN_PROVIDER", raising=False)
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)

    with pytest.raises(SystemExit) as e:
        cli.main(["convert", str(img), "--gen-diagrams"])
    assert e.value.code == 1


def test_cli_gen_diagrams_passes_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGOCR_CONFIG_DIR", str(tmp_path / "none"))
    img = tmp_path / "b.png"
    Image.new("RGB", (8, 8)).save(img)

    seen = {}

    class FakeVision:
        def analyze(self, prompt, image):
            return json.dumps({"diagram_elements": [
                {"type": "venn", "description": "sets"}]})

    class FakeGen:
        def generate(self, prompt, source_image=None):
            seen["called"] = True
            return Image.new("RGB", (4, 4))

    from engocr.extractor import VisionExtractor as RealExtractor
    monkeypatch.setattr(cli, "VisionExtractor",
                        lambda: RealExtractor(provider=FakeVision()))
    monkeypatch.setattr("engocr.imagegen.make_image_gen_provider",
                        lambda: FakeGen())

    cli.main(["convert", str(img), "--gen-diagrams"])

    assert seen.get("called")
    assert (tmp_path / "b_p1_diagram_1.png").exists()
    md = (tmp_path / "b.md").read_text()
    assert "![sets](b_p1_diagram_1.png)" in md
