"""High-level conversion: images and PDFs → PageVisionResult → markdown."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from engocr.config import LLM_MAX_CONCURRENT, PDF_RENDER_DPI
from engocr.extractor import VisionExtractor
from engocr.imagegen import build_gen_prompt, crop_from_bbox, make_image_gen_provider
from engocr.logging import get_logger
from engocr.progress import page_progress
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
    import pymupdf

    doc = pymupdf.open(str(path))
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
    progress: bool = False,
    gen_diagrams: bool = False,
    image_gen=None,
    out_dir: Path | str | None = None,
) -> ConversionResult:
    """Convert one image or PDF file to markdown (per-page vision calls).

    progress: show a tqdm page bar on stderr (CLI default; off for
    library use).
    gen_diagrams: for diagrams/sketches without reconstructed source,
    redraw them with an image-generation provider (cropped sketch +
    instruction) and embed the PNG in the markdown. image_gen: provider
    instance (default: IMAGE_GEN_PROVIDER). out_dir: where generated
    PNGs go (default: next to the input).
    """
    path = Path(path)
    result = ConversionResult(source=str(path))
    try:
        if path.suffix.lower() == ".pdf":
            pages, page_images = _convert_pdf(extractor, path, dpi, workers,
                                              progress)
        else:
            img = load_image(path)
            pages, page_images = [extractor.extract_page(img)], [img]
            _logger.info("generated content for %s", path.name)
        if gen_diagrams:
            provider = image_gen or make_image_gen_provider()
            enhance_with_generated_images(
                pages, page_images, provider,
                Path(out_dir) if out_dir else path.parent, path.stem)
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
    progress: bool = False,
) -> tuple[list[PageVisionResult], list[Image.Image]]:
    from concurrent.futures import ThreadPoolExecutor

    rendered = list(render_pdf_pages(path, dpi))
    results: dict[int, PageVisionResult] = {}
    errors: dict[int, str] = {}

    def _call(pn: int, img: Image.Image):
        try:
            return pn, extractor.extract_page(img), None
        except Exception as e:
            return pn, None, str(e)

    with (ThreadPoolExecutor(max_workers=max(1, min(workers, len(rendered)))) as ex,
          page_progress(total=len(rendered), desc=path.name,
                        enabled=progress) as bar):
        for pn, vr, error in ex.map(lambda a: _call(*a), rendered):
            bar.update(1)
            if vr is not None:
                results[pn] = vr
                _logger.info("generated content for %s (page %d/%d)",
                             path.name, pn + 1, len(rendered))
            else:
                errors[pn] = str(error)
                _logger.warning("vision_failed file=%s page=%s error=%s",
                                path, pn, error)

    pages: list[PageVisionResult] = []
    page_images: list[Image.Image] = []
    for pn, img in rendered:
        page_images.append(img)
        if pn in results:
            pages.append(results[pn])
        else:
            pages.append(PageVisionResult(
                page_summary=f"[page {pn + 1}: vision extraction failed — "
                             f"{errors.get(pn, 'unknown error')}]"))
    return pages, page_images


# ── Image generation for non-reconstructable sketches ──

def enhance_with_generated_images(
    pages: list[PageVisionResult],
    page_images: list[Image.Image],
    provider,
    out_dir: Path,
    stem: str,
) -> int:
    """Redraw source-less diagrams/sketches with an image-gen provider.

    Only elements that resisted code reconstruction (empty source) and
    have a description are eligible. Fail-soft per element: a failed
    generation leaves the description as the only representation.
    Returns the number of images written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for page_idx, (page, img) in enumerate(zip(pages, page_images)):
        n = 0
        for elem in (*page.diagram_elements, *page.sketch_elements):
            if elem.source.strip() or not elem.description.strip():
                continue
            n += 1
            filename = f"{stem}_p{page_idx + 1}_diagram_{n}.png"
            try:
                crop = crop_from_bbox(img, elem.bbox_approx)
                generated = provider.generate(
                    build_gen_prompt(elem.description), crop)
                generated.save(out_dir / filename, format="PNG")
                elem.generated_image = filename
                written += 1
            except Exception as e:
                _logger.warning("image_gen_failed file=%s error=%s",
                                filename, e)
    return written


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
        block = _visual_block(d.type, d.description, d.source, d.source_lang,
                              d.generated_image)
        if block:
            visual_blocks.append(block)
    for s in result.sketch_elements:
        block = _visual_block(f"sketch ({s.type})", s.description,
                              s.source, s.source_lang, s.generated_image)
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


_TIKZ_DOC_TEMPLATE = """\\usepackage{tikz}
\\usepackage{amsmath}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}

\\begin{document}
%s
\\end{document}"""


def _wrap_tikz_document(source: str) -> str:
    """Wrap tikz content in the document preamble + body.

    (No \\documentclass — the consuming toolchain supplies the class.)

    Idempotent: a source that is already a complete document (contains
    \\begin{document}) passes through unchanged.
    """
    body = source.strip()
    if "\\begin{document}" in body:
        return body
    return _TIKZ_DOC_TEMPLATE % body


def _visual_block(kind: str, description: str, source: str,
                  source_lang: str, generated_image: str = "") -> str:
    """One diagram/sketch entry: description bullet + optional source code
    or generated image.

    markdown-table sources render inline (no fence); tikz sources are
    wrapped in a compilable standalone document; everything else gets a
    language-tagged fence (mermaid).
    """
    parts: list[str] = []
    if description.strip():
        parts.append(f"- **{kind}**: {description}")
    if source.strip():
        if source_lang == "markdown":
            parts.append(source.strip())
        else:
            fenced = (_wrap_tikz_document(source) if source_lang == "tikz"
                      else source.strip())
            parts.append(f"```{source_lang}\n{fenced}\n```")
    if generated_image.strip():
        parts.append(f"![{description.strip()}]({generated_image.strip()})")
    return "\n\n".join(parts)
