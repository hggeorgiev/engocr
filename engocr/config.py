"""EngOCR configuration constants.

Providers read their API keys and model overrides directly from the
environment (GEMINI_API_KEY, VISION_MODEL, ...). The engocr CLI loads
~/.config/engocr/config.json into the environment at startup (env wins);
embedding applications (e.g. EngMem) do the same through their own
bootstrap, so one config style drives both.
"""

import os

# ── PDF rendering (CLI) ───────────────────────────────
PDF_RENDER_DPI = 200                     # for page image rendering (vision)

# ── Concurrency ───────────────────────────────────────
LLM_MAX_CONCURRENT = 2                   # parallel API calls (raise on paid tier)

# ── Vision gating (quality-first) ─────────────────────
# These thresholds decide when a digital PDF page ALSO goes to the vision
# extractor. Deliberately trigger-happy: a redundant vision call is
# acceptable, a missed equation or diagram is not.
VISION_TRIGGER_MIN_CHARS = 100           # below this the page is likely scanned
VISION_TRIGGER_IMAGE_COUNT = 1           # embedded raster images → vision
VISION_TRIGGER_DRAWING_COUNT = 5         # vector drawings (tikz/charts/tables) → vision
VISION_TRIGGER_MATH_DENSITY = 0.02       # math-symbol share of page text → vision
VISION_TRIGGER_MIN_MATH_SYMBOLS = 8      # absolute math-symbol count → vision

# ── Vision retry (rate-limit backoff) ─────────────────
VISION_RETRY_MAX = 4                     # attempts per page (initial + 3 backoff retries)
VISION_RETRY_BASE_SECONDS = 10           # backoff base (10s → 30s → 90s + jitter)

# ── Gemini vision ─────────────────────────────────────
GEMINI_VISION_MODEL = "gemini-3.6-flash"  # default when VISION_MODEL is unset
GEMINI_VISION_TEMPERATURE = 0.0          # deterministic structured output
GEMINI_VISION_MAX_OUTPUT_TOKENS = 4096

# ── Provider selection (read here for convenience; providers also
#    read the environment directly) ────────────────────
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "gemini")
VISION_MODEL = os.environ.get("VISION_MODEL", "")   # per-provider override
