"""Gemini structured extraction client with deterministic cache keys."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel


class GeminiQuotaError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: str = "data/processed/llm_cache",
        max_retries: int = 2,
        retry_base_seconds: float = 2.0,
    ):
        self.model = model or os.getenv("GEMINI_MODEL")
        if not self.model:
            raise ValueError(
                "GEMINI_MODEL is not set. Record the exact model name used for each run."
            )

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def _cache_path(self, prompt: str, article: dict) -> Path:
        raw = json.dumps(
            {"prompt": prompt, "article": article, "model": self.model},
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def extract(self, prompt: str, article: dict, schema_model: type[BaseModel]):
        path = self._cache_path(prompt, article)
        if path.exists():
            return schema_model.model_validate_json(path.read_text(encoding="utf-8"))

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt + "\n\nARTICLE:\n" + json.dumps(article, ensure_ascii=False),
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=schema_model,
                    ),
                )
                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Gemini returned an empty response.")
                result = schema_model.model_validate_json(text)
                path.write_text(result.model_dump_json(), encoding="utf-8")
                return result
            except Exception as exc:
                last_error = exc
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    raise GeminiQuotaError(msg) from exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_base_seconds * (2**attempt))

        raise RuntimeError(f"Gemini request failed after retries: {last_error}") from last_error
