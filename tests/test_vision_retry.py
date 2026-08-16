"""Tests for vision rate-limit retry and thread-local clients (no API calls)."""

import threading
import time

import pytest

from engocr.extractor import (
    _call_with_retry,
    _is_rate_limit_error,
)
from engocr.providers.gemini import (
    GeminiVisionProvider,
)


def test_is_rate_limit_error():
    assert _is_rate_limit_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert _is_rate_limit_error(Exception("Rate limit reached"))
    assert not _is_rate_limit_error(Exception("500 INTERNAL"))
    assert not _is_rate_limit_error(Exception("404 NOT_FOUND"))


def test_retry_succeeds_after_rate_limits(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert _call_with_retry(fn, max_attempts=4, base_seconds=1) == "ok"
    assert calls["n"] == 3


def test_retry_exhaustion_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise Exception("429 rate limit")

    with pytest.raises(Exception, match="429"):
        _call_with_retry(fn, max_attempts=3, base_seconds=1)
    assert calls["n"] == 3


def test_non_rate_limit_fails_fast(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise Exception("404 NOT_FOUND")

    with pytest.raises(Exception, match="404"):
        _call_with_retry(fn, max_attempts=4, base_seconds=1)
    assert calls["n"] == 1  # no retry on non-rate-limit errors


def test_thread_local_clients_are_distinct_per_thread(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    extractor = GeminiVisionProvider()

    class FakeClient:
        pass

    extractor._new_client = FakeClient
    per_thread = {}
    lock = threading.Lock()

    def worker(name):
        c1 = extractor._get_client()
        c2 = extractor._get_client()
        with lock:
            per_thread[name] = (c1, c2)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    main_client = extractor._get_client()

    assert all(c1 is c2 for c1, c2 in per_thread.values())
    clients = [c for pair in per_thread.values() for c in pair] + [main_client]
    assert len({id(c) for c in clients}) == 4
