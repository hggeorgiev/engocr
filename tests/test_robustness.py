"""Truncated-JSON salvage, malformed-output handling, bbox hardening, and
the compilable TikZ document wrapper — no API calls."""

import json

from PIL import Image

from engocr.convert import _visual_block, _wrap_tikz_document
from engocr.extractor import (
    _MALFORMED_MARKER,
    VisionExtractor,
    _clean_bbox,
    _repair_truncated_json,
)

FULL_PAYLOAD = {
    "page_summary": "Handwritten math notes on the Orthogonal "
                    "Decomposition Theorem.",
    "text_elements": [
        {"text": "Orthogonal Decomposition Theorem", "type": "handwriting",
         "bbox_approx": [0.03, 0.06, 0.65, 0.12]},
        {"text": "Let \\(W\\) be a subspace in \\(\\mathbb{R}^n\\).",
         "type": "handwriting", "bbox_approx": [0.01, 0.13, 0.91, 0.2]},
    ],
    "equations": [{"latex": "\\[x = \\hat{x} + z\\]", "eq_type": "display"}],
}
FULL_JSON = json.dumps(FULL_PAYLOAD)


def _parse(raw):
    class FakeProvider:
        def analyze(self, prompt, image):
            return raw
    return VisionExtractor(provider=FakeProvider()) \
        .extract_page(Image.new("RGB", (4, 4)))


# ── truncation salvage (the "cut off page" bug) ──────


def test_truncation_inside_bbox_salvages_complete_elements():
    # cut mid-way through the second element's bbox array (real report)
    truncated = FULL_JSON[:FULL_JSON.index("0.91")]
    result = _parse(truncated)

    assert result.page_summary.startswith("Handwritten math notes")
    assert len(result.text_elements) >= 1
    assert result.text_elements[0].text == "Orthogonal Decomposition Theorem"


def test_truncation_mid_string_salvages_summary():
    # cut inside the second element's text string (mid-string truncation)
    cut = FULL_JSON.index("subspace")
    result = _parse(FULL_JSON[:cut])
    assert result.page_summary.startswith("Handwritten math notes")
    assert result.text_elements[0].text == "Orthogonal Decomposition Theorem"


def test_truncation_after_complete_element_keeps_it():
    boundary = FULL_JSON.index('"equations"')
    result = _parse(FULL_JSON[:boundary - 2])   # cut after text_elements ]
    assert len(result.text_elements) == 2
    assert result.equations == []               # tail cleanly absent


def test_repair_returns_none_for_non_json():
    assert _repair_truncated_json("plain prose answer") is None
    assert _repair_truncated_json("") is None


def test_repair_returns_none_when_balanced():
    assert _repair_truncated_json('{"a": 1}') is None


def test_repair_drops_dangling_comma():
    repaired = _repair_truncated_json('{"a": 1, "b": [2,')
    assert json.loads(repaired) == {"a": 1, "b": [2]}


def test_unrecoverable_json_becomes_marker_not_dump():
    result = _parse('{"page_summary": "x", } trailing garbage }}}')
    assert result.page_summary == _MALFORMED_MARKER
    assert "page_summary" not in result.page_summary  # no raw JSON leaked


def test_prose_response_still_salvaged_as_summary():
    result = _parse("This page contains notes on eigenvalues.")
    assert result.page_summary == "This page contains notes on eigenvalues."


# ── tolerant parse chain (the "unrecoverable json" bug) ──


def test_literal_newlines_inside_strings_parse():
    """Models emit literal line breaks inside multi-line source strings;
    strict json.loads rejects them ("Invalid control character")."""
    raw = ('{"page_summary": "Gram-Schmidt notes.", "diagram_elements": ['
           '{"type": "flowchart", "description": "d", "source": '
           '"flowchart TD\n    A --> B\n    B --> C", '
           '"source_lang": "mermaid"}]}')
    result = _parse(raw)

    assert result.page_summary == "Gram-Schmidt notes."
    source = result.diagram_elements[0].source
    assert source.startswith("flowchart TD")
    assert "A --> B" in source and "B --> C" in source


def test_trailing_commentary_salvaged():
    raw = '{"page_summary": "ok", "text_elements": []}\nHere is the JSON!'
    result = _parse(raw)
    assert result.page_summary == "ok"


def test_truncation_plus_literal_newline_salvaged():
    raw = ('{"page_summary": "notes", "diagram_elements": ['
           '{"type": "flowchart", "description": "d", "source": '
           '"flowchart TD\n    A --> B", "source_lang": "mermaid"}, '
           '{"type": "tree", "description": "trunc')
    result = _parse(raw)
    assert result.page_summary == "notes"
    # first element fully intact; the truncated second element is kept as
    # its complete prefix (type only — empty description renders nothing)
    assert "A --> B" in result.diagram_elements[0].source
    assert result.diagram_elements[0].source_lang == "mermaid"
    assert result.diagram_elements[1].type == "tree"
    assert result.diagram_elements[1].description == ""


def test_marker_still_used_for_truly_unrecoverable():
    result = _parse('{"page_summary": "x", } garbage }}}')
    assert result.page_summary == _MALFORMED_MARKER


# ── lone-backslash (LaTeX) escape sanitizing ─────────


def test_lone_backslash_latex_parses():
    """Single-backslash LaTeX ("\\int") is an invalid JSON escape and used
    to kill every parse layer — the reported 'unrecoverable' pages."""
    raw = ('{"page_summary": "Orthogonal projection notes.", '
           '"equations": [{"latex": "\\[\\int_0^\\infty e^{-x^2}dx\\]", '
           '"eq_type": "display"}]}'.replace("\\\\", "\\"))
    result = _parse(raw)

    assert result.page_summary == "Orthogonal projection notes."
    assert result.equations[0].latex == "\\[\\int_0^\\infty e^{-x^2}dx\\]"


def test_escape_sanitizer_idempotent_for_valid_json():
    from engocr.extractor import _escape_lone_backslashes as san
    good = '{"a": "line1\\nline2 \\\\ \\"q\\" \\u0041"}'
    assert san(good) == good   # valid escapes untouched
    assert json.loads(san(good))["a"] == 'line1\nline2 \\ "q" A'


def test_escape_sanitizer_defangs_incomplete_unicode_escape():
    from engocr.extractor import _escape_lone_backslashes as san
    assert san('{"a": "abc \\u12') == '{"a": "abc \\\\u12'


def test_combined_newline_backslash_truncation_salvaged():
    raw = ('{"page_summary": "Gram-Schmidt", "text_elements": ['
           '{"text": "line one\nand \\alpha here", "type": "handw')
    result = _parse(raw)
    assert result.page_summary == "Gram-Schmidt"
    assert "line one" in result.text_elements[0].text
    assert "\\alpha" in result.text_elements[0].text


def test_unrecoverable_warning_includes_error_reason(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="engocr.extractor"):
        _parse('{"page_summary": "x", } garbage }}}')
    assert "json_unrecoverable" in caplog.text
    assert "error=" in caplog.text   # the JSONDecodeError reason is logged


def test_prompt_requires_escaped_backslashes():
    class FakeProvider:
        def analyze(self, prompt, image):
            return "{}"
    prompt = VisionExtractor(provider=FakeProvider())._build_prompt()
    assert "escape every backslash" in prompt


# ── bbox hardening ───────────────────────────────────


def test_clean_bbox_accepts_valid():
    assert _clean_bbox([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]


def test_clean_bbox_rejects_malformed():
    full = [0.0, 0.0, 1.0, 1.0]
    assert _clean_bbox([0.1, 0.2]) == full          # partial (salvage)
    assert _clean_bbox("nope") == full
    assert _clean_bbox(None) == full
    assert _clean_bbox([0.1, 0.2, "x", 0.4]) == full


def test_partial_bbox_in_payload_defaults_to_full_page():
    payload = dict(FULL_PAYLOAD)
    payload["text_elements"] = [{"text": "t", "bbox_approx": [0.1, 0.2]}]
    result = _parse(json.dumps(payload))
    assert result.text_elements[0].bbox_approx == [0.0, 0.0, 1.0, 1.0]


# ── TikZ document wrapper ────────────────────────────


def test_tikz_wrapped_in_document_preamble():
    wrapped = _wrap_tikz_document("\\begin{tikzpicture}\n\\draw (0,0);\n"
                                  "\\end{tikzpicture}")
    assert "\\documentclass" not in wrapped   # toolchain supplies the class
    assert wrapped.startswith("\\usepackage{tikz}")
    assert "\\usepackage{amsmath}" in wrapped
    assert "\\usepackage{pgfplots}" in wrapped
    assert "\\begin{document}" in wrapped
    assert "\\end{document}" in wrapped
    assert "\\begin{tikzpicture}" in wrapped


def test_tikz_wrap_is_idempotent():
    doc = ("\\documentclass{article}\n\\usepackage{tikz}\n"
           "\\begin{document}\nx\n\\end{document}")
    assert _wrap_tikz_document(doc) == doc


def test_visual_block_tikz_contains_full_document():
    block = _visual_block("sketch (coordinate_system)", "a parabola",
                          "\\begin{tikzpicture}\nx\n\\end{tikzpicture}",
                          "tikz")
    assert block.startswith("- **sketch (coordinate_system)**: a parabola")
    assert "```tikz\n\\usepackage{tikz}" in block
    assert "\\end{document}\n```" in block


def test_visual_block_mermaid_not_wrapped():
    block = _visual_block("graph", "d", "flowchart TD\n A --> B", "mermaid")
    assert "\\documentclass" not in block
    assert "```mermaid\nflowchart TD" in block


def test_prompt_instructs_tikz_body_only():
    class FakeProvider:
        def analyze(self, prompt, image):
            return "{}"
    extractor = VisionExtractor(provider=FakeProvider())
    prompt = extractor._build_prompt()
    assert "emit ONLY the tikzpicture" in prompt
    assert "document wrapper" in prompt
