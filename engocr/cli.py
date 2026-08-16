"""engocr CLI — convert handwritten notes (images/PDFs) to markdown.

Config: ~/.config/engocr/config.json (same shape as EngMem's —
vision_provider, vision_model, endpoints, api.*_key), loaded into the
environment at startup; environment variables always win.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from engocr.config import LLM_MAX_CONCURRENT, PDF_RENDER_DPI
from engocr.convert import ConversionResult, convert_file
from engocr.extractor import VisionExtractor
from engocr.logging import get_logger

_logger = get_logger(__name__)

# config.json key -> environment variable
_ENV_MAP = {
    "vision_provider": "VISION_PROVIDER",
    "vision_model": "VISION_MODEL",
    "azure_endpoint": "AZURE_OPENAI_ENDPOINT",
    "azure_api_version": "AZURE_OPENAI_API_VERSION",
    "ollama_base_url": "OLLAMA_BASE_URL",
}
_API_MAP = {
    "gemini_key": "GEMINI_API_KEY",
    "openai_key": "OPENAI_API_KEY",
    "anthropic_key": "ANTHROPIC_API_KEY",
    "mistral_key": "MISTRAL_API_KEY",
    "openrouter_key": "OPENROUTER_API_KEY",
    "qwen_key": "DASHSCOPE_API_KEY",
    "xai_key": "XAI_API_KEY",
    "azure_key": "AZURE_OPENAI_API_KEY",
}

_TEMPLATE = {
    "vision_provider": "gemini",
    "vision_model": "",
    "azure_endpoint": "",
    "ollama_base_url": "",
    "api": {
        "gemini_key": "", "openai_key": "", "anthropic_key": "",
        "mistral_key": "", "openrouter_key": "", "qwen_key": "",
        "xai_key": "", "azure_key": "",
    },
}


def config_path() -> Path:
    override = os.environ.get("ENGOCR_CONFIG_DIR")
    if override:
        return Path(os.path.expanduser(override)) / "config.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(os.path.expanduser(xdg)) if xdg else Path.home() / ".config"
    return base / "engocr" / "config.json"


def load_user_config() -> None:
    """Apply config.json values as environment defaults. Never raises."""
    path = config_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning("config_load_failed path=%s error=%s", path, e)
        return
    for key, env_var in _ENV_MAP.items():
        value = data.get(key)
        if value:
            os.environ.setdefault(env_var, os.path.expanduser(str(value)))
    for key, env_var in _API_MAP.items():
        value = (data.get("api") or {}).get(key)
        if value:
            os.environ.setdefault(env_var, str(value))


def init_config() -> Path:
    """Write the template config.json (mode 0600). Refuses to overwrite."""
    path = config_path()
    if path.exists():
        raise FileExistsError(f"Config already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_TEMPLATE, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


# ── convert ──────────────────────────────────────────

def _write_output(result: ConversionResult, args, input_path: Path) -> None:
    as_json = args.json
    if as_json:
        payload = {
            "source": result.source,
            "pages": [asdict(p) for p in result.pages],
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        suffix = ".json"
    else:
        content = result.markdown
        suffix = ".md"

    if args.stdout:
        sys.stdout.write(content)
        return

    out = Path(args.out).expanduser() if args.out else None
    if out is None:
        dest = input_path.with_suffix(suffix)
    elif out.is_dir() or (len(args.files) > 1 and not out.suffix):
        out.mkdir(parents=True, exist_ok=True)
        dest = out / (input_path.stem + suffix)
    else:
        dest = out if out.suffix == suffix else out.with_suffix(suffix)
    dest.write_text(content, encoding="utf-8")
    print(f"{input_path} → {dest}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="engocr",
        description="Convert handwritten notes (images/PDFs) to markdown "
                    "with a vision model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config", help="Write a template config.json")

    convert = subparsers.add_parser(
        "convert", help="Convert images/PDFs to markdown (default)")
    convert.add_argument("files", nargs="+", help="Image or PDF files")
    convert.add_argument("-o", "--out", default=None,
                         help="Output file (single input) or directory")
    convert.add_argument("--stdout", action="store_true",
                         help="Print markdown to stdout")
    convert.add_argument("--json", action="store_true",
                         help="Emit the raw structured result as JSON")
    convert.add_argument("--provider", default=None,
                         help="Vision provider (overrides config)")
    convert.add_argument("--model", default=None,
                         help="Vision model (overrides config)")
    convert.add_argument("--dpi", type=int, default=PDF_RENDER_DPI,
                         help=f"PDF render DPI (default: {PDF_RENDER_DPI})")
    convert.add_argument("-j", "--workers", type=int,
                         default=LLM_MAX_CONCURRENT,
                         help="Parallel page conversions for PDFs "
                              f"(default: {LLM_MAX_CONCURRENT})")

    args = parser.parse_args(argv)

    if args.command == "config":
        try:
            path = init_config()
        except FileExistsError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        print(f"Wrote {path} — fill in your provider and API key.")
        return

    # convert
    load_user_config()
    if args.provider:
        os.environ["VISION_PROVIDER"] = args.provider
    if args.model:
        os.environ["VISION_MODEL"] = args.model

    try:
        extractor = VisionExtractor()
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    failures = 0
    for name in args.files:
        input_path = Path(name).expanduser()
        if not input_path.is_file():
            print(f"Not found: {input_path}", file=sys.stderr)
            failures += 1
            continue
        result = convert_file(extractor, input_path,
                              dpi=args.dpi, workers=args.workers)
        if result.error:
            print(f"Failed: {input_path}: {result.error}", file=sys.stderr)
            failures += 1
            continue
        _write_output(result, args, input_path)

    if failures == len(args.files):
        sys.exit(1)


if __name__ == "__main__":
    main()
