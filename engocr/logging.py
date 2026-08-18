"""Minimal stdlib logging for engocr."""

from __future__ import annotations

import logging
import os

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=os.environ.get("ENGOCR_LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        # HTTP-client request lines are noise; engocr logs its own
        # concise per-page messages instead.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("google_genai").setLevel(logging.WARNING)
        _configured = True
    return logging.getLogger(name)
