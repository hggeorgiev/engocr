"""Sketch/diagram → diagram-source reconstruction (mermaid/tikz/markdown).
All with fakes — no API calls."""

import json

from PIL import Image

from engocr.convert import _visual_block, to_markdown
from engocr.extractor import VisionExtractor, _route_lang
from engocr.schema import DiagramElement, PageVisionResult, SketchElement


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload

    def analyze(self, prompt, image):
        return json.dumps(self.payload)


def _extract(payload):
    return VisionExtractor(provider=FakeProvider(payload)) \
        .extract_page(Image.new("RGB", (4, 4)))


# ── parsing ──────────────────────────────────────────


def test_source_fields_parsed():
    result = _extract({
        "diagram_elements": [{
            "type": "flowchart",
            "description": "Start to End",
            "source": "flowchart TD\n    A[Start] --> B[End]",
            "source_lang": "mermaid",
        }],
        "sketch_elements": [{
            "type": "coordinate_system",
            "description": "parabola",
            "source": "\\begin{tikzpicture}...\\end{tikzpicture}",
            "source_lang": "tikz",
        }],
    })
    d = result.diagram_elements[0]
    assert d.source.startswith("flowchart TD")
    assert d.source_lang == "mermaid"
    s = result.sketch_elements[0]
    assert s.source_lang == "tikz"


def test_source_fields_default_empty():
    result = _extract({
        "diagram_elements": [{"type": "graph", "description": "d"}],
        "sketch_elements": [{"type": "freehand", "description": "s"}],
    })
    assert result.diagram_elements[0].source == ""
    assert result.diagram_elements[0].source_lang == "mermaid"  # routed
    assert result.sketch_elements[0].source == ""
    assert result.sketch_elements[0].source_lang == ""   # freehand: no route


def test_route_lang_normalization():
    assert _route_lang("graph", "") == "mermaid"
    assert _route_lang("flowchart", "") == "mermaid"
    assert _route_lang("tree", "") == "mermaid"
    assert _route_lang("table", "") == "markdown"
    assert _route_lang("coordinate_system", "") == "tikz"
    assert _route_lang("geometry", "") == "tikz"
    assert _route_lang("freehand", "") == ""
    assert _route_lang("venn", "") == ""
    assert _route_lang("graph", "tikz") == "tikz"      # declared wins
    assert _route_lang("", "Mermaid") == "mermaid"     # normalized


# ── markdown rendering ───────────────────────────────


def test_mermaid_source_rendered_as_fence():
    md = to_markdown(PageVisionResult(diagram_elements=[DiagramElement(
        type="graph",
        description="Two disjoint vertex sets",
        source="flowchart TD\n    U1 --> V1",
        source_lang="mermaid",
    )]))
    assert "- **graph**: Two disjoint vertex sets" in md
    assert "```mermaid\nflowchart TD\n    U1 --> V1\n```" in md


def test_tikz_source_rendered_as_fence():
    md = to_markdown(PageVisionResult(sketch_elements=[SketchElement(
        type="coordinate_system",
        description="A parabola",
        source="\\begin{tikzpicture}\n\\draw ...\n\\end{tikzpicture}",
        source_lang="tikz",
    )]))
    assert "```tikz\n\\begin{tikzpicture}" in md


def test_markdown_table_source_rendered_inline():
    md = to_markdown(PageVisionResult(diagram_elements=[DiagramElement(
        type="table",
        description="truth table",
        source="| A | B |\n|---|---|",
        source_lang="markdown",
    )]))
    assert "| A | B |" in md
    assert "```" not in md


def test_no_source_keeps_description_only():
    md = to_markdown(PageVisionResult(diagram_elements=[DiagramElement(
        type="venn", description="overlapping sets",
    )]))
    assert "- **venn**: overlapping sets" in md
    assert "```" not in md


def test_source_without_description_still_rendered():
    md = to_markdown(PageVisionResult(diagram_elements=[DiagramElement(
        type="tree", source="flowchart TD\n    R --> L\n    R --> R2",
        source_lang="mermaid",
    )]))
    assert "```mermaid" in md
    assert "## Diagrams & sketches" in md


def test_visual_block_empty_when_nothing():
    assert _visual_block("graph", "", "", "") == ""
