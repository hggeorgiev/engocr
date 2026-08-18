"""Progress display: tqdm on a TTY, silent no-op otherwise.

Bars always go to stderr, so `--stdout` conversions stay pipe-clean.
Library callers pass ``progress=True`` explicitly (CLI does this by
default; ``-q/--quiet`` or a non-TTY stderr disables the bar).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager


class _NullBar:
    def update(self, n: int = 1) -> None:
        pass


@contextmanager
def page_progress(total: int, desc: str, enabled: bool = True) -> Iterator:
    """Yield a bar with .update(n); tqdm on a TTY, no-op stand-in else."""
    if enabled and sys.stderr.isatty():
        from tqdm import tqdm
        with tqdm(total=total, desc=desc, unit="page", file=sys.stderr) as bar:
            yield bar
    else:
        yield _NullBar()
