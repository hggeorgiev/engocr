"""EngOCR configuration constants.

Providers read their API keys and model overrides directly from the
environment (GEMINI_API_KEY, VISION_MODEL, ...). The engocr CLI loads
~/.config/engocr/config.json into the environment at startup (env wins);
embedding applications (e.g. EngMem) do the same through their own
bootstrap, so one config style drives both.
"""

import os

PDF_RENDER_DPI = 200                     # for page image rendering (vision)

# ── Concurrency ───────────────────────────────────────
LLM_MAX_CONCURRENT = 2

# ── Vision retry (rate-limit backoff) ─────────────────
VISION_RETRY_MAX = 4                     # attempts per page (initial + 3 backoff retries)
VISION_RETRY_BASE_SECONDS = 10           # backoff base (10s → 30s → 90s + jitter)

# ── Gemini vision ─────────────────────────────────────
GEMINI_VISION_MODEL = "gemini-3.6-flash"  # default when VISION_MODEL is unset
GEMINI_VISION_TEMPERATURE = 0.0          # deterministic structured output
GEMINI_VISION_MAX_OUTPUT_TOKENS = 16384

# ── Provider selection (read here for convenience; providers also
#    read the environment directly) ────────────────────
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "gemini")
VISION_MODEL = os.environ.get("VISION_MODEL", "")   # per-provider override
