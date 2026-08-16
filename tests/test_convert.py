"""Conversion pipeline tests: markdown rendering, PDF conversion, and the
provider-agnostic caption helper — all with fakes (no API calls)."""

import json

from PIL import Image

from engocr.captions import caption_image
from engocr.convert import (
    _math_block,
    convert_file,
    pages_to_markdown,
    to_markdown,
)
from engocr.extractor import VisionExtractor
from engocr.schema import (
    PageVisionResult,
)


class FakeProvider:
    """Returns a canned structured page, echoes prompts for inspection."""

    def __init__(self, payload=None):
        self.payload = payload or {
            "page_summary": "A page about integrals.",
            "text_elements": [
                {"text": "The Gaussian integral is classic.", "type": "printed"},
            ],
            "equations": [
                {"latex": "\\[\\int e^{-x^2} dx = \\sqrt{\\pi}\\]",
                 "eq_type": "display", "label": "Eq. (1)",
                 "description": "Gaussian integral"},
            ],
            "code_elements": [
                {"code": "def f(x):\n    return x * 2", "language": "python"},
            ],
            "diagram_elements": [
                {"type": "graph", "description": "Two disjoint vertex sets"},
            ],
            "sketch_elements": [
                {"type": "freehand", "description": "A quick arrow"},
            ],
            "captions_and_annotations": ["Margin note: important"],
        }
        self.calls = 0

    def analyze(self, prompt, image):
        self.calls += 1
        return json.dumps(self.payload)


def _extractor(payload=None):
    return VisionExtractor(provider=FakeProvider(payload))


def _image(tmp_path, name="notes.jpg"):
    path = tmp_path / name
    Image.new("RGB", (8, 8), "white").save(path)
    return path


# ── to_markdown ───────────────────────────────────────


def test_to_markdown_renders_all_element_kinds():
    md = to_markdown(_extractor().extract_page(Image.new("RGB", (4, 4))))

    assert md.startswith("> A page about integrals.")
    assert "The Gaussian integral is classic." in md
    assert "$$\n\\int e^{-x^2} dx = \\sqrt{\\pi}\n$$" in md   # \[..\] → $$
    assert "**Eq. (1)**" in md
    assert "*Gaussian integral*" in md
    assert "```python\ndef f(x):\n    return x * 2\n```" in md
    assert "- **graph**: Two disjoint vertex sets" in md
    assert "- **sketch (freehand)**: A quick arrow" in md
    assert "- Margin note: important" in md


def test_to_markdown_empty_result():
    assert to_markdown(PageVisionResult()) == ""


def test_math_block_delimiter_normalization():
    assert _math_block("\\[x^2\\]", "display") == "$$\nx^2\n$$"
    assert _math_block("$$x^2$$", "display") == "$$\nx^2\n$$"
    assert _math_block("x^2", "display") == "$$\nx^2\n$$"
    assert _math_block("\\(x^2\\)", "inline") == "$x^2$"
    assert _math_block("x^2", "inline") == "$x^2$"


def test_pages_to_markdown_separates_pages():
    pages = [PageVisionResult(page_summary="one"),
             PageVisionResult(page_summary="two")]
    md = pages_to_markdown(pages, title="scan")
    assert md.startswith("# scan")
    assert "<!-- page 2 -->" in md
    assert "---" in md
    assert md.index("one") < md.index("two")


# ── convert_file ─────────────────────────────────────


def test_convert_image_file(tmp_path):
    result = convert_file(_extractor(), _image(tmp_path))
    assert result.error == ""
    assert len(result.pages) == 1
    assert result.markdown.startswith("# notes")
    assert "Gaussian integral" in result.markdown


def test_convert_missing_or_bad_file(tmp_path):
    bad = tmp_path / "not-an-image.jpg"
    bad.write_text("garbage")
    result = convert_file(_extractor(), bad)
    assert result.error != ""


def test_convert_pdf_renders_every_page(tmp_path):
    import fitz
    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=200, height=200)
    doc.save(pdf)
    doc.close()

    extractor = _extractor()
    result = convert_file(extractor, pdf)

    assert result.error == ""
    assert len(result.pages) == 3
    assert extractor.provider.calls == 3
    # every page is marked; separators only between pages
    assert result.markdown.count("<!-- page") == 3
    assert result.markdown.count("\n---\n") == 2


def test_convert_pdf_page_failure_isolated(tmp_path):
    import fitz
    pdf = tmp_path / "scan.pdf"
    doc = fitz.open()
    for _ in range(2):
        doc.new_page(width=200, height=200)
    doc.save(pdf)
    doc.close()

    class FlakyProvider:
        def __init__(self):
            self.n = 0

        def analyze(self, prompt, image):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("API_KEY_INVALID")
            return '{"page_summary": "ok page"}'

    result = convert_file(VisionExtractor(provider=FlakyProvider()), pdf,
                          workers=1)

    assert result.error == ""   # file-level conversion succeeded
    assert len(result.pages) == 2
    summaries = [p.page_summary for p in result.pages]
    assert "ok page" in summaries
    assert any("vision extraction failed" in s for s in summaries)


# ── caption_image ─────────────────────────────────────


def test_caption_parses_json_response():
    class P:
        def analyze(self, prompt, image):
            assert '{"caption"' in prompt   # JSON-mode providers get a JSON ask
            return '{"caption": "a whiteboard with a graph"}'

    assert caption_image(Image.new("RGB", (4, 4)), provider=P()) == \
        "a whiteboard with a graph"


def test_caption_lenient_fence_and_raw_text():
    class Fenced:
        def analyze(self, prompt, image):
            return '```json\n{"caption": "boxed answer"}\n```'

    class Raw:
        def analyze(self, prompt, image):
            return "plain caption text"

    img = Image.new("RGB", (4, 4))
    assert caption_image(img, provider=Fenced()) == "boxed answer"
    assert caption_image(img, provider=Raw()) == "plain caption text"


def test_caption_failure_falls_back():
    class Down:
        def analyze(self, prompt, image):
            raise RuntimeError("api down")

    assert caption_image(Image.new("RGB", (4, 4)), provider=Down()) == "image"


def test_caption_custom_prompt_passed_through():
    seen = {}

    class P:
        def analyze(self, prompt, image):
            seen["prompt"] = prompt
            return '{"caption": "x"}'

    caption_image(Image.new("RGB", (4, 4)), provider=P(), prompt="CUSTOM")
    assert seen["prompt"] == "CUSTOM"
