# EngOCR
Local-first OCR for technical handwritten notes: images and PDFs → structured
text, LaTeX equations, code listings, and markdown.

EngOCR drives a configurable vision model and produces clean markdown: transcribed handwriting, `$$…$$` display
math, fenced code blocks, and descriptions of diagrams and sketches.


**Example 1**

| Before | After |
|---|---|
| ![Before](/examples/engocr-example-1-before.png) | ![After](/examples/engocr-example-1-after.png) |

**Example 2**

| Before | After |
|---|---|
| ![Before](/examples/engocr-2-before-example.png) | ![After](/examples/engocr-example-2-after.png) |


## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install engocr

engocr config            # writes ~/.config/engocr/config.json — edit it
```

Requires Python ≥3.12 (uv fetches one if needed).

## Usage

```bash
engocr convert notes.jpg              # → notes.md (next to the input)
engocr convert *.jpg -o converted/    # one .md per image, into a directory
engocr convert scan.pdf               # every page rendered + transcribed
                                      #   (tqdm progress bar on stderr)
engocr convert notes.jpg --stdout     # print markdown instead of writing
engocr convert notes.jpg --json       # raw structured result (JSON)
engocr convert notes.jpg --gen-diagrams  # also redraw freehand sketches as PNGs
engocr convert notes.jpg --provider anthropic --model claude-sonnet-4-5
```


## Providers

Set `vision_provider` in `~/.config/engocr/config.json` (or the
`VISION_PROVIDER` env var) and the matching API key:


## Python API

```python
from engocr import VisionExtractor, convert_file, caption_image, to_markdown

extractor = VisionExtractor()                 # env/config-driven provider
result = extractor.extract_page(pil_image)    # → PageVisionResult (structured)

conversion = convert_file(extractor, "notes.jpg")
print(conversion.markdown)                    # rendered markdown
print(conversion.pages[0].equations[0].latex) # structured access

caption = caption_image(pil_image)            # one-line description, any provider
```

`PageVisionResult` is the stable public schema: `page_summary`,
`text_elements`, `equations` (LaTeX + label + natural-language gloss),
`code_elements`, `diagram_elements` / `sketch_elements` (description +
reconstructed `source` / `source_lang`: mermaid, tikz, or markdown),
`captions_and_annotations`. (`tags_suggestion` / `half_life_suggestion` are
optional consumer extensions used by EngMem; ignore them standalone.)

