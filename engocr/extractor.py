from __future__ import annotations

import json
import random
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
     "bbox_approx": [x0,y0,x1,y1]}}
  ],
  "sketch_elements": [
    {{"type": "freehand|coordinate_system|geometry|other",
     "description": "What the sketch depicts",
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

Rules for equations:
- Transcribe ALL visible mathematical expressions into proper LaTeX format.
- Use \\(...\\) for inline math and \\[...\\] for display/block math (do NOT use $$).
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
        """Parse the provider's JSON response into a PageVisionResult."""
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
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
                bbox_approx=t.get("bbox_approx", [0.0, 0.0, 1.0, 1.0]),
            ))

        for d in data.get("diagram_elements", []):
            result.diagram_elements.append(DiagramElement(
                type=d.get("type", ""),
                description=d.get("description", ""),
                structured_data=d.get("structured_data", {}),
                bbox_approx=d.get("bbox_approx", [0.0, 0.0, 1.0, 1.0]),
            ))

        for s in data.get("sketch_elements", []):
            result.sketch_elements.append(SketchElement(
                type=s.get("type", "freehand"),
                description=s.get("description", ""),
                bbox_approx=s.get("bbox_approx", [0.0, 0.0, 1.0, 1.0]),
            ))

        for e in data.get("equations", []):
            result.equations.append(EquationElement(
                latex=e.get("latex", ""),
                eq_type=e.get("eq_type", "display"),
                label=e.get("label", ""),
                description=e.get("description", ""),
                bbox_approx=e.get("bbox_approx", [0.0, 0.0, 1.0, 1.0]),
            ))

        for c in data.get("code_elements", []):
            result.code_elements.append(CodeElement(
                code=c.get("code", ""),
                language=c.get("language", ""),
                bbox_approx=c.get("bbox_approx", [0.0, 0.0, 1.0, 1.0]),
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
