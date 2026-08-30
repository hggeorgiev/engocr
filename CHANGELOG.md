# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-30

### Added

- First public release.
- Local-first OCR for handwritten technical notes: images and PDFs → structured
  markdown with LaTeX equations, code listings, and diagram descriptions.
- Vision providers: `gemini`, `openai`, `anthropic`, `qwen`, `mistral`,
  `openrouter`, `ollama`, `xai`, `azure`.
- Sketch → diagram reconstruction (Mermaid / TikZ / markdown tables).
- Optional `--gen-diagrams` redraw of freehand sketches via image generation.
- CLI (`engocr convert ...`) with JSON/`--stdout` output modes.
- Python API: `VisionExtractor`, `convert_file`, `caption_image`, `to_markdown`.
