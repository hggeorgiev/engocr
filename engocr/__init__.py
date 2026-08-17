"""EngOCR — local-first OCR for technical handwritten notes.

Images and PDFs → structured text, LaTeX equations, code listings, and
markdown, via a configurable vision provider (gemini | openai | qwen |
anthropic | mistral | openrouter | ollama | xai | azure).

    from engocr import VisionExtractor, convert_file, to_markdown

    extractor = VisionExtractor()            # env-configured provider
    result = extractor.extract_page(image)   # → PageVisionResult
    conversion = convert_file(extractor, "notes.jpg")
    print(conversion.markdown)
"""

from engocr.captions import caption_image
from engocr.convert import (
    ConversionResult,
    convert_file,
    load_image,
    pages_to_markdown,
    render_pdf_pages,
    to_markdown,
)
from engocr.extractor import VisionExtractor
from engocr.providers.base import VisionProvider, make_provider
from engocr.schema import (
    CodeElement,
    DiagramElement,
    EquationElement,
    PageVisionResult,
    SketchElement,
    TextElement,
)

__version__ = "0.1.0"

__all__ = [
    "CodeElement",
    "ConversionResult",
    "DiagramElement",
    "EquationElement",
    "PageVisionResult",
    "SketchElement",
    "TextElement",
    "VisionExtractor",
    "VisionProvider",
    "caption_image",
    "convert_file",
    "load_image",
    "make_provider",
    "pages_to_markdown",
    "render_pdf_pages",
    "to_markdown",
]
