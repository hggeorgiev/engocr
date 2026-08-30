# EngOCR

Local-first OCR for technical handwritten notes: images and PDFs → structured
text, LaTeX equations, code listings, and markdown. Built for engineers,
researchers, and students who study from handwritten material and want it in
plain text files.

EngOCR drives a configurable vision model (no local model weights — API or
Ollama) and produces clean markdown: transcribed handwriting, `$$…$$` display
math, fenced code blocks, and descriptions of diagrams and sketches.

**Sketch → diagram source.** Beyond describing diagrams, engocr reconstructs
them as *editable diagram code* — graphs/flowcharts/trees become
[Mermaid](https://mermaid.js.org), coordinate-system/geometry sketches become
TikZ/pgfplots, tables become markdown tables. The result is versionable,
diffable text that renders natively in GitHub/Obsidian/VS Code (Mermaid) or
compiles with LaTeX (TikZ). Only structure the model can ground in the image
is emitted; anything unclear stays a plain description.

**Sketch → generated image** (`--gen-diagrams`, opt-in). Sketches that resist
code reconstruction (venn diagrams, freehand drawings, unclear structure) are
*redrawn as clean digital diagrams* by an image-generation model (Gemini or
OpenAI `gpt-image-1` — `image_gen_provider` / `image_gen_model` in config).
The model sees the cropped sketch region plus an instruction, so the redraw
stays faithful to your original. PNGs are written next to the markdown and
embedded as `![description](file.png)`. One image call per such sketch.

> **Status: 0.1.0 (pre-release)** — expect breaking changes until 1.0.

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

## Examples

Handwritten page → EngOCR markdown output.

**Example 1**

| Before | After |
|---|---|
| ![Before](https://raw.githubusercontent.com/hggeorgiev/engocr/main/examples/engocr-example-1-before.png) | ![After](https://raw.githubusercontent.com/hggeorgiev/engocr/main/examples/engocr-example-1-after.png) |

**Example 2**

| Before | After |
|---|---|
| ![Before](https://raw.githubusercontent.com/hggeorgiev/engocr/main/examples/engocr-2-before-example.png) | ![After](https://raw.githubusercontent.com/hggeorgiev/engocr/main/examples/engocr-example-2-after.png) |

## Providers

Set `vision_provider` in `~/.config/engocr/config.json` (or the
`VISION_PROVIDER` env var) and the matching API key:

| provider | default model | config key | env var |
|---|---|---|---|
| `gemini` (default) | gemini-3.6-flash | `api.gemini_key` | `GEMINI_API_KEY` |
| `openai` | gpt-4o | `api.openai_key` | `OPENAI_API_KEY` |
| `anthropic` | claude-sonnet-4-5 | `api.anthropic_key` | `ANTHROPIC_API_KEY` |
| `qwen` | qwen-vl-max | `api.qwen_key` | `DASHSCOPE_API_KEY` |
| `mistral` | pixtral-large-latest | `api.mistral_key` | `MISTRAL_API_KEY` |
| `openrouter` | qwen2.5-vl-72b-instruct | `api.openrouter_key` | `OPENROUTER_API_KEY` |
| `ollama` (local) | qwen2.5vl:7b | — (no key) | `OLLAMA_BASE_URL` |
| `xai` | grok-2-vision-1212 | `api.xai_key` | `XAI_API_KEY` |
| `azure` | your deployment | `api.azure_key` + `azure_endpoint` | `AZURE_OPENAI_*` |

`vision_model` / `VISION_MODEL` overrides the default model per provider
(required for `azure` = deployment name).

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

## Development

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests -v
.venv/bin/ruff check engocr/ tests/
```
