from __future__ import annotations

import json
import random
import re
import time

from PIL import Image

from engocr.config import (
    VISION_RETRY_BASE_SECONDS,
    VISION_RETRY_MAX,
)
from engocr.logging import get_logger
from engocr.providers.base import (
    VisionProvider,
    make_provider,
)
from engocr.schema import (
    CodeElement,
    DiagramElement,
    EquationElement,
    PageVisionResult,
    SketchElement,
    TextElement,
)

_logger = get_logger(__name__)

# ── Rate-limit retry ─────────────────────────────────

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


def _call_with_retry(fn, max_attempts: int, base_seconds: float):
    """Retry fn() on rate-limit errors with exponential backoff + jitter."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts or not _is_rate_limit_error(e):
                raise
            delay = base_seconds * (3 ** (attempt - 1)) + random.uniform(0, base_seconds)
            _logger.warning("vision_rate_limited attempt=%s delay=%.1f",
                            attempt, delay)
            time.sleep(delay)


# ── The extractor ────────────────────────────────────

class VisionExtractor:
    """Sends a page image to a configurable vision provider and returns a
    structured page analysis. Provider = gemini | openai | qwen | anthropic
    | mistral | openrouter | ollama | xai | azure (env: VISION_PROVIDER,
    model override: VISION_MODEL).
    """

    def __init__(self, provider: str | VisionProvider | None = None):
        self.provider = make_provider(provider) if isinstance(provider, str) \
            else (provider or make_provider())
        self._taxonomy_tags: list[str] = []

    def set_taxonomy_tags(self, tags: list[str]) -> None:
        """Inject a constrained tag vocabulary into the prompt (optional;
        used by EngMem's taxonomy — standalone engocr usage can ignore it)."""
        self._taxonomy_tags = tags

    def _build_prompt(self) -> str:
        tag_list = ""
        if self._taxonomy_tags:
            tag_list = ("\nAvailable taxonomy tags (choose ONLY from this "
                        "list for tags_suggestion):\n")
            tag_list += ", ".join(sorted(self._taxonomy_tags)) + "\n"

        return f"""Analyze this page image. It may contain a mix of: printed text,
handwritten notes, mathematical diagrams, flowcharts, mind maps, sketches,
margin notes, and annotations.

{tag_list}
Return ONLY a JSON object with this exact structure:
{{
  "page_summary": "One sentence describing the page holistically.",
  "text_elements": [
    {{"text": "...", "type": "handwriting|printed|mixed", "bbox_approx": [x0,y0,x1,y1]}}
  ],
  "diagram_elements": [
    {{"type": "graph|flowchart|tree|table|venn|other",
     "description": "Detailed description including labels, node names, relationships",
     "structured_data": {{}},
     "source": "flowchart TD\\n    A[Start] --> B[End]",
     "source_lang": "mermaid",
     "bbox_approx": [x0,y0,x1,y1]}}
  ],
  "sketch_elements": [
    {{"type": "freehand|coordinate_system|geometry|other",
     "description": "What the sketch depicts",
     "source": "\\\\begin{{tikzpicture}}...\\\\end{{tikzpicture}}",
     "source_lang": "tikz",
     "bbox_approx": [x0,y0,x1,y1]}}
  ],
  "equations": [
    {{
      "latex": "\\\\int_0^\\\\infty e^{{-x^2}}dx = \\\\frac{{\\\\sqrt{{\\pi}}}}{{2}}",
      "eq_type": "display",
      "label": "Eq. (3.14)",
      "description": "Gaussian integral over the positive real line",
      "bbox_approx": [0.1, 0.3, 0.9, 0.45]
    }}
  ],
  "code_elements": [
    {{"code": "def fib(n):\\n    if n < 2:\\n        return n\\n    return fib(n-1) + fib(n-2)",
      "language": "python",
      "bbox_approx": [0.1, 0.5, 0.9, 0.7]}}
  ],
  "captions_and_annotations": ["Arrow label: ...", "Margin note: ..."],
  "tags_suggestion": ["Tag1", "Tag2"],
  "half_life_suggestion": "foundational|paradigms|tools|ephemeral"
}}

Rules:
- bbox_approx: [x0, y0, x1, y1] as fractions of page width/height (0.0 to 1.0).
- For diagram_elements, include structured_data with
  {{"nodes": [...], "edges": [...]}} if applicable.
- tags_suggestion: 1-5 taxonomy tags the page seems about. Use ONLY exact tag names.
- half_life_suggestion: one of the four layers.
- Be precise and do not hallucinate content not visible.

Rules for source reconstruction (diagram_elements and sketch_elements):
- Reconstruct each diagram/sketch as editable diagram source code when the
  structure is clearly visible. Language by element type:
  * graph | flowchart | tree | mindmap → Mermaid (source_lang "mermaid").
    Use a conservative subset: "flowchart TD", simple A --> B edges,
    quoted node labels. No subgraphs, no styling, no exotic syntax.
  * table → a markdown table (source_lang "markdown").
  * coordinate_system | geometry sketches → TikZ/pgfplots
    (source_lang "tikz"): emit ONLY the tikzpicture (use an axis
    environment for plots) — NO \\documentclass, \\usepackage, or
    document environment; a compilable document wrapper is added
    automatically.
- Add a line for including the amsmath in every TikZ/pgfplots drawing
  * venn | other | freehand → source "" (description only).
- Only emit source you can ground in the image: real node labels, real
  edge directions, real curve shapes. If any part is unclear, omit that
  part; if the whole structure is unclear, use source "".
- The description stays mandatory even when source is present.
- Inside JSON string values, escape newlines as \\n — never put a literal
  line break inside a string (breaks JSON parsing).
- In JSON strings, escape every backslash as \\\\ — write LaTeX commands as
  \\\\int, \\\\frac, \\\\mathbb (a single backslash breaks JSON parsing).

Rules for equations:
- Transcribe ALL visible mathematical expressions into proper LaTeX format.
- Use $...$ for inline math and \\[...\\] for display/block math (do NOT use $$).
- Inside text_elements and descriptions, wrap EVERY mathematical expression,
  symbol, or variable in $...$ — never leave bare LaTeX commands
  (\\int, \\frac, \\sum, ^, _, \\alpha, ...) unwrapped in prose.
- For matrices, use \\begin{{pmatrix}} or \\begin{{bmatrix}} environments.
- For aligned equations, use \\begin{{aligned}} within \\[...\\].
- For chemical formulas, use \\ce{{...}} notation where applicable.
- Preserve equation numbers and theorem labels when visible (e.g. "(3.14)", "Theorem 2.1").
- description: one short natural-language sentence stating what the formula
  expresses or computes (e.g. "Gaussian integral over the real line",
  "Definition of the Laplace transform"). Never repeat the LaTeX.
- Never use raw Unicode math symbols: use \\int \\sum \\prod \\infty \\alpha
  \\beta \\gamma \\delta \\epsilon \\lambda \\mu \\sigma \\omega \\to
  \\Rightarrow \\forall \\exists \\in \\subset \\oplus \\otimes \\partial
  \\nabla \\approx \\equiv \\leq \\geq \\neq \\pm \\cdot \\times \\cup \\cap
  \\setminus \\subseteq \\supseteq \\langle \\rangle \\mathbb{{R}}
  \\mathbb{{N}} \\mathbb{{Z}} \\mathbb{{C}}.
- For commutative diagrams, describe the structure then provide a minimal tikz-cd representation.
- eq_type MUST be exactly "inline" or "display".
- If any symbol is truly ambiguous, flag it with a LaTeX comment: % uncertain symbol.

Rules for code_elements:
- Transcribe each programming code listing VERBATIM, preserving indentation and line breaks.
- One element per listing; do not merge separate listings.
- Do NOT repeat code listing text in text_elements — code belongs only in code_elements.
- language: your best guess (e.g. "python", "rust", "c"); use "" if unclear.
- Pseudocode and algorithm blocks count as code."""

    def _parse_response(self, raw_text: str) -> PageVisionResult:
        """Parse the provider's JSON response into a PageVisionResult.

        Tolerant cascade (each layer salvages a common LLM failure):
        strict=False loads (literal newlines/tabs inside strings — the
        multi-line mermaid/tikz sources invite these) → raw_decode (valid
        JSON followed by trailing commentary) → truncation repair (output
        cap hit on dense pages). Unrecoverable JSON-looking output becomes
        a clean marker; prose-looking output (no-JSON-mode providers) is
        kept as the page summary.
        """
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        # LaTeX-heavy content invites single-backslash sequences ("\int",
        # "\frac") that are invalid JSON escapes; idempotent for valid JSON.
        text = _escape_lone_backslashes(text)

        data = None
        error: Exception | None = None
        try:
            data = json.loads(text, strict=False)
        except json.JSONDecodeError as e:
            error = e
        if data is None and text.startswith("{"):
            try:
                data, _ = json.JSONDecoder(strict=False).raw_decode(text)
                _logger.info("json_salvaged trailing_commentary")
            except json.JSONDecodeError as e:
                error = e
                data = None
        if data is None:
            repaired = _repair_truncated_json(text)
            if repaired is not None:
                try:
                    data = json.loads(repaired, strict=False)
                    _logger.info("json_salvaged dropped_tail_chars=%d",
                                 len(text) - len(repaired))
                except json.JSONDecodeError as e:
                    error = e
                    data = None
        if data is None:
            if text.startswith("{"):
                _logger.warning("json_unrecoverable error=%s head=%.80s",
                                error, text)
                _logger.debug("json_unrecoverable_raw\n%s", raw_text)
                return PageVisionResult(page_summary=_MALFORMED_MARKER)
            return PageVisionResult(page_summary=raw_text[:500])

        result = PageVisionResult(
            page_summary=data.get("page_summary", ""),
            captions_and_annotations=data.get("captions_and_annotations", []),
            tags_suggestion=data.get("tags_suggestion", []),
            half_life_suggestion=data.get(
                "half_life_layer", data.get("half_life_suggestion", "paradigms")),
        )

        for t in data.get("text_elements", []):
            result.text_elements.append(TextElement(
                text=t.get("text", ""),
                type=t.get("type", "handwriting"),
                bbox_approx=_clean_bbox(t.get("bbox_approx")),
            ))

        for d in data.get("diagram_elements", []):
            result.diagram_elements.append(DiagramElement(
                type=d.get("type", ""),
                description=d.get("description", ""),
                structured_data=d.get("structured_data", {}),
                source=d.get("source", ""),
                source_lang=_route_lang(d.get("type", ""), d.get("source_lang", "")),
                bbox_approx=_clean_bbox(d.get("bbox_approx")),
            ))

        for s in data.get("sketch_elements", []):
            result.sketch_elements.append(SketchElement(
                type=s.get("type", "freehand"),
                description=s.get("description", ""),
                source=s.get("source", ""),
                source_lang=_route_lang(s.get("type", ""), s.get("source_lang", "")),
                bbox_approx=_clean_bbox(s.get("bbox_approx")),
            ))

        for e in data.get("equations", []):
            result.equations.append(EquationElement(
                latex=e.get("latex", ""),
                eq_type=e.get("eq_type", "display"),
                label=e.get("label", ""),
                description=e.get("description", ""),
                bbox_approx=_clean_bbox(e.get("bbox_approx")),
            ))

        for c in data.get("code_elements", []):
            result.code_elements.append(CodeElement(
                code=c.get("code", ""),
                language=c.get("language", ""),
                bbox_approx=_clean_bbox(c.get("bbox_approx")),
            ))

        return result

    def extract_page(self, page_image: Image.Image) -> PageVisionResult:
        """Send a page image to the configured provider; parse the result."""
        generated = _call_with_retry(
            lambda: self.provider.analyze(self._build_prompt(), page_image),
            max_attempts=VISION_RETRY_MAX,
            base_seconds=VISION_RETRY_BASE_SECONDS,
        )

        return self._parse_response(generated)


# ── Diagram-source language routing ──────────────────

_MERMAID_TYPES = {"graph", "flowchart", "tree", "mindmap"}
_TIKZ_TYPES = {"coordinate_system", "geometry"}


def _route_lang(element_type: str, declared: str) -> str:
    """Normalize source_lang: trust a declared value, else route by type."""
    declared = (declared or "").strip().lower()
    if declared:
        return declared
    t = (element_type or "").strip().lower()
    if t in _MERMAID_TYPES:
        return "mermaid"
    if t == "table":
        return "markdown"
    if t in _TIKZ_TYPES:
        return "tikz"
    return ""


# ── Truncated-JSON salvage ───────────────────────────

_MALFORMED_MARKER = ("[malformed model output — re-run `engocr convert` "
                     "to retry this page]")

_DEFAULT_BBOX = [0.0, 0.0, 1.0, 1.0]

# Valid JSON escapes are consumed whole (first alternative); a lone
# backslash (second alternative, captured) is doubled. LaTeX content like
# "\int" or "\frac" is an invalid JSON escape unless the model doubles
# the backslash — this makes those responses parseable. Idempotent for
# valid JSON.
_LONE_BACKSLASH_RE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|[\\"/bfnrt])|(\\)')


def _escape_lone_backslashes(text: str) -> str:
    """Double backslashes that are not part of a valid JSON escape."""
    return _LONE_BACKSLASH_RE.sub(
        lambda m: "\\\\" if m.group(1) else m.group(0), text)


def _clean_bbox(value) -> list[float]:
    """bbox_approx must be 4 normalized numbers; anything else → full page.

    Models sometimes emit 2-point boxes, and the truncation salvage can
    leave a partial array — a malformed bbox would crash cropping later.
    """
    if (isinstance(value, list) and len(value) == 4
            and all(isinstance(v, (int, float)) for v in value)):
        return value
    return list(_DEFAULT_BBOX)


def _repair_truncated_json(raw: str) -> str | None:
    """Salvage a truncated JSON object from a model response.

    Walks the text tracking string state and the open-bracket stack,
    cuts back to the last clean element boundary (dropping the
    incomplete tail), and closes every open bracket. Returns the
    repaired string, or None when nothing salvageable remains (or the
    text isn't JSON-shaped at all).
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None

    stack: list[str] = []      # expected closers, innermost last
    in_string = False
    escaped = False
    cut = 0                    # clean cut candidate (element boundary)

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
            cut = i + 1
        elif ch in "}]":
            if stack:
                stack.pop()
                cut = i + 1
        elif ch == ",":
            cut = i            # cut BEFORE the comma: no dangling comma

    if not stack or cut == 0:
        return None

    prefix = text[:cut].rstrip()
    if not prefix or prefix[-1] in ":,":
        return None
    return prefix + "".join(reversed(stack))
