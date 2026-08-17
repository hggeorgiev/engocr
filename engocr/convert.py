"""High-level conversion: images and PDFs → PageVisionResult → markdown."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from engocr.config import LLM_MAX_CONCURRENT, PDF_RENDER_DPI
from engocr.extractor import VisionExtractor
from engocr.logging import get_logger
from engocr.schema import PageVisionResult

_logger = get_logger(__name__)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif",
}


@dataclass
class ConversionResult:
    """The conversion of one input file (possibly multi-page)."""
    source: str
    pages: list[PageVisionResult] = field(default_factory=list)
    markdown: str = ""
    error: str = ""


# ── Input rendering ──────────────────────────────────

def load_image(path: Path | str) -> Image.Image:
    """Open an image file as RGB."""
    return Image.open(path).convert("RGB")


def render_pdf_pages(
    path: Path | str,
    dpi: int = PDF_RENDER_DPI,
) -> Iterator[tuple[int, Image.Image]]:
    """Render each PDF page to a PIL image at `dpi` (for vision)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    try:
        for page_index in range(len(doc)):
            pix = doc[page_index].get_pixmap(dpi=dpi)
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            yield page_index, img.convert("RGB")
    finally:
        doc.close()


# ── Conversion ───────────────────────────────────────

def convert_file(
    extractor: VisionExtractor,
    path: Path | str,
    dpi: int = PDF_RENDER_DPI,
    workers: int = LLM_MAX_CONCURRENT,
) -> ConversionResult:
    """Convert one image or PDF file to markdown (per-page vision calls)."""
    path = Path(path)
    result = ConversionResult(source=str(path))
    try:
        if path.suffix.lower() == ".pdf":
            pages = _convert_pdf(extractor, path, dpi, workers)
        else:
            pages = [extractor.extract_page(load_image(path))]
        result.pages = pages
        result.markdown = pages_to_markdown(pages, title=path.stem)
    except Exception as e:
        result.error = str(e)
        _logger.warning("conversion_failed file=%s error=%s", path, e)
    return result


def _convert_pdf(
    extractor: VisionExtractor,
    path: Path,
    dpi: int,
    workers: int,
) -> list[PageVisionResult]:
    from concurrent.futures import ThreadPoolExecutor

    rendered = list(render_pdf_pages(path, dpi))
    results: dict[int, PageVisionResult] = {}
    errors: dict[int, str] = {}

    def _call(pn: int, img: Image.Image):
        try:
            return pn, extractor.extract_page(img), None
        except Exception as e:
            return pn, None, str(e)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(rendered)))) as ex:
        for pn, vr, error in ex.map(lambda a: _call(*a), rendered):
            if vr is not None:
                results[pn] = vr
            else:
                errors[pn] = str(error)
                _logger.warning("vision_failed file=%s page=%s error=%s",
                                path, pn, error)

    pages: list[PageVisionResult] = []
    for pn, _ in rendered:
        if pn in results:
            pages.append(results[pn])
        else:
            pages.append(PageVisionResult(
                page_summary=f"[page {pn + 1}: vision extraction failed — "
                             f"{errors.get(pn, 'unknown error')}]"))
    return pages


# ── Markdown rendering ───────────────────────────────

def pages_to_markdown(pages: list[PageVisionResult], title: str = "") -> str:
    """Render a multi-page conversion as one markdown document."""
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
    for i, page in enumerate(pages):
        if len(pages) > 1 and i > 0:
            parts.append("\n---\n")
        if len(pages) > 1:
            parts.append(f"<!-- page {i + 1} -->\n")
        parts.append(to_markdown(page))
    return "\n".join(p for p in parts if p is not None).strip() + "\n"


def to_markdown(result: PageVisionResult) -> str:
    """Render one page as markdown: transcription, LaTeX, code, diagrams."""
    sections: list[str] = []

    if result.page_summary:
        sections.append(f"> {result.page_summary}")

    texts = [t.text.strip() for t in result.text_elements if t.text.strip()]
    if texts:
        sections.append("\n\n".join(texts))

    for eq in result.equations:
        if not eq.latex.strip():
            continue
        block = _math_block(eq.latex, eq.eq_type)
        if eq.label:
            block = f"**{eq.label}**\n\n{block}"
        if eq.description:
            block = f"{block}\n\n*{eq.description}*"
        sections.append(block)

    for ce in result.code_elements:
        if ce.code.strip():
            sections.append(f"```{ce.language}\n{ce.code.rstrip()}\n```")

    visual_blocks = []
    for d in result.diagram_elements:
        block = _visual_block(d.type, d.description, d.source, d.source_lang)
        if block:
            visual_blocks.append(block)
    for s in result.sketch_elements:
        block = _visual_block(f"sketch ({s.type})", s.description,
                              s.source, s.source_lang)
        if block:
            visual_blocks.append(block)
    if visual_blocks:
        sections.append("## Diagrams & sketches\n\n" + "\n\n".join(visual_blocks))

    notes = [n.strip() for n in result.captions_and_annotations if n.strip()]
    if notes:
        sections.append("## Notes\n\n" + "\n".join(f"- {n}" for n in notes))

    return "\n\n".join(sections)


def _math_block(latex: str, eq_type: str) -> str:
    """Normalize delimiters: display → $$…$$, inline → $…$."""
    body = latex.strip()
    for opener, closer in ((r"\[", r"\]"), (r"\(", r"\)")):
        if body.startswith(opener) and body.endswith(closer):
            body = body[len(opener):-len(closer)].strip()
            break
    else:
        if body.startswith("$$") and body.endswith("$$") and len(body) > 4:
            body = body[2:-2].strip()
    if eq_type == "inline":
        return f"${body}$"
    return f"$$\n{body}\n$$"


def _visual_block(kind: str, description: str, source: str,
                  source_lang: str) -> str:
    """One diagram/sketch entry: description bullet + optional source code.

    markdown-table sources render inline (no fence); everything else gets
    a language-tagged fence (mermaid, tikz).
    """
    parts: list[str] = []
    if description.strip():
        parts.append(f"- **{kind}**: {description}")
    if source.strip():
        if source_lang == "markdown":
            parts.append(source.strip())
        else:
            parts.append(f"```{source_lang}\n{source.strip()}\n```")
    return "\n\n".join(parts)
