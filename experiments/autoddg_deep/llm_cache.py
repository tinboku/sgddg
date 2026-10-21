"""LLM client wrappers for the experiment scripts.

CachingClient sits between AutoDDG and the real client:
  * every request is content-addressed (model, messages, temperature as passed,
    kwargs, run_id) and stored under cache/llm/, so re-runs read from disk and a
    second run_id gives an independent replicate for the noise floor;
  * the temperature each call actually received is recorded, including
    "not passed" (None) for the original SemanticProfiler/SFD calls, whose
    effective value is the client default.

FakeClient returns schema-valid canned answers so the whole pipeline can be
exercised offline (--dry-run). Its outputs are never written next to real runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from autoddg.llm import LLMClient, OpenAICompatibleClient

NOT_PASSED = None  # recorded value when the caller did not pass a temperature
OPENAI_DEFAULT_TEMPERATURE = 0.0  # autoddg.llm.openai_compat default


class FakeClient(LLMClient):
    """Offline stand-in that answers each AutoDDG prompt type with valid output."""

    COLUMN = {
        "Domain-Specific Types": "General",
        "Entity Type": "Entity",
        "Function/Usage Context": "Measurement",
        "Spatial": {"isSpatial": False, "resolution": ""},
        "Temporal": {"isTemporal": False, "resolution": ""},
    }

    def chat_completions_create(self, model, messages, temperature=0.0, **kwargs):
        user = messages[-1]["content"]
        if "Columns to analyze" in user:
            names = re.findall(r"^Column: (.+)$", user, re.M)
            content = json.dumps({n: self.COLUMN for n in names})
        elif "semantic analyzer" in user:
            content = json.dumps(self.COLUMN)
        elif "Topic (2-3 words)" in user:
            content = "Dry Run Topic"
        elif "expand the description" in user:
            content = "Dataset Overview: dry run search description. Keywords: dry, run."
        else:
            digest = hashlib.sha256(user.encode()).hexdigest()[:8]
            content = f"Dry run description {digest}."
        return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": None}


class CachingClient(LLMClient):
    def __init__(
        self,
        inner: LLMClient | None,
        cache_dir: Path,
        run_id: str,
        provider: str,
        offline: bool = False,
    ):
        self.inner = inner
        self.provider = provider
        # one directory per provider so fake (dry-run) answers can never be served to a real run
        self.cache_dir = Path(cache_dir) / provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.offline = offline  # cache-only: raise on miss instead of calling the API
        self.log: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _key(self, model, messages, temperature, kwargs) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "kwargs": kwargs,
            "run_id": self.run_id,
            "provider": self.provider,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat_completions_create(self, model, messages, temperature=NOT_PASSED, **kwargs):
        key = self._key(model, messages, temperature, kwargs)
        path = self.cache_dir / key[:2] / f"{key}.json"
        hit = path.exists()
        if hit:
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            if self.offline or self.inner is None:
                raise RuntimeError(f"LLM cache miss with no live client ({key[:12]})")
            call_kwargs = dict(kwargs)
            if temperature is not NOT_PASSED:
                call_kwargs["temperature"] = temperature
            started = time.time()
            response = self.inner.chat_completions_create(model=model, messages=messages, **call_kwargs)
            record = {
                "key": key,
                "model": model,
                "temperature_passed": temperature,
                "temperature_effective": (
                    OPENAI_DEFAULT_TEMPERATURE if temperature is NOT_PASSED else temperature
                ),
                "kwargs": kwargs,
                "run_id": self.run_id,
                "provider": self.provider,
                "messages": messages,
                "response": response,
                "latency_s": round(time.time() - started, 3),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        with self._lock:
            self.log.append(
                {
                    "key": key,
                    "cache_hit": hit,
                    "model": model,
                    "temperature_passed": record["temperature_passed"],
                    "temperature_effective": record["temperature_effective"],
                    "usage": record["response"].get("usage"),
                }
            )
        return record["response"]

    def drain_log(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self.log = self.log, []
        return out


def make_inner_client(provider: str) -> LLMClient | None:
    if provider == "none":
        return None  # cache-only; any miss raises
    if provider == "fake":
        return FakeClient()
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY is not set. Export it (and OPENAI_BASE_URL for an "
                "OpenAI-compatible endpoint), or use --dry-run / --stages profile."
            )
        import openai

        return OpenAICompatibleClient(openai.OpenAI())
    raise SystemExit(f"unknown provider {provider!r}")
