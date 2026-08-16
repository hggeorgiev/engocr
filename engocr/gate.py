"""Quality-first vision gate: decide when a digital page ALSO goes to the
vision extractor (pure functions, no I/O)."""

from __future__ import annotations

from engocr.config import (
    VISION_TRIGGER_DRAWING_COUNT,
    VISION_TRIGGER_IMAGE_COUNT,
    VISION_TRIGGER_MATH_DENSITY,
    VISION_TRIGGER_MIN_CHARS,
    VISION_TRIGGER_MIN_MATH_SYMBOLS,
)

_MATH_SYMBOLS = frozenset(
    "∫∑∏√∂∇∈∀∃≠≤≥±×÷⊗⊕∪∩⊂⊃∧∨¬⇒⇔→←↔∞≈∝∅"
    "αβγδεζηθικλμνξπρστυφχψω"
    "ΓΔΘΛΞΠΣΦΨΩ"
)


def math_density(text: str) -> tuple[float, int]:
    """Return (math-symbol share of text, absolute math-symbol count)."""
    if not text:
        return 0.0, 0
    count = sum(1 for ch in text if ch in _MATH_SYMBOLS)
    return count / len(text), count


def page_needs_vision(
    text_chars: int,
    image_count: int,
    drawing_count: int,
    math_sym_density: float,
    math_sym_count: int,
    force_all: bool = False,
) -> bool:
    """Decide whether a page goes to the vision extractor.

    Deliberately trigger-happy (quality-first): any sign of scanning,
    figures, vector graphics, or non-trivial math routes to vision. Only
    confident pure-prose pages stay on the cheap text path.
    """
    if force_all:
        return True
    if text_chars < VISION_TRIGGER_MIN_CHARS:
        return True   # likely scanned or handwritten
    if image_count >= VISION_TRIGGER_IMAGE_COUNT:
        return True   # embedded figures/diagrams
    if drawing_count >= VISION_TRIGGER_DRAWING_COUNT:
        return True   # vector graphics (tikz, charts, tables)
    if math_sym_density >= VISION_TRIGGER_MATH_DENSITY:
        return True   # equation-heavy page
    if math_sym_count >= VISION_TRIGGER_MIN_MATH_SYMBOLS:
        return True   # sparse but present math
    return False
