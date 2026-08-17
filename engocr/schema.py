"""Structured output schema for page vision extraction.

This is engocr's public contract: one PageVisionResult per page image.
Consumers (e.g. EngMem's ingestion pipeline) map it into their own data
model; field names are stable.

`tags_suggestion` / `half_life_suggestion` are optional consumer-facing
extension fields (EngMem constrains tags to its taxonomy and uses
half-life layers for retention); standalone engocr usage can ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextElement:
    text: str
    type: str = "handwriting"                # "handwriting" | "printed" | "mixed"
    bbox_approx: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class DiagramElement:
    type: str = ""                            # "graph" | "flowchart" | "tree" | "venn" | ...
    description: str = ""
    structured_data: dict = field(default_factory=dict)
    source: str = ""                          # reconstructed diagram code ("" = not grounded)
    source_lang: str = ""                     # "mermaid" | "markdown" | "tikz"
    bbox_approx: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class SketchElement:
    type: str = "freehand"
    description: str = ""
    source: str = ""                          # reconstructed diagram code ("" = not grounded)
    source_lang: str = ""                     # "tikz" for coordinate/geometry sketches
    bbox_approx: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class EquationElement:
    """A LaTeX equation found on the page."""
    latex: str = ""
    eq_type: str = "display"                  # "inline" or "display"
    label: str = ""                           # e.g. "Eq. (3.14)", "Theorem 2.1", or ""
    description: str = ""                     # one-line NL gloss of what the formula expresses
    bbox_approx: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class CodeElement:
    """A programming code listing found on the page."""
    code: str = ""
    language: str = ""                        # best guess: "python", "rust", ...
    bbox_approx: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


@dataclass
class PageVisionResult:
    """Complete structured output for one page image."""
    page_summary: str = ""
    text_elements: list[TextElement] = field(default_factory=list)
    diagram_elements: list[DiagramElement] = field(default_factory=list)
    sketch_elements: list[SketchElement] = field(default_factory=list)
    equations: list[EquationElement] = field(default_factory=list)
    code_elements: list[CodeElement] = field(default_factory=list)
    captions_and_annotations: list[str] = field(default_factory=list)
    tags_suggestion: list[str] = field(default_factory=list)
    half_life_suggestion: str = "paradigms"
