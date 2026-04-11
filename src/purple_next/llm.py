"""Thin OpenAI-compatible chat client.

- Heartbeat logging while a call is in flight.
- Auto-switches between ``max_tokens`` and ``max_completion_tokens``.
- Auto-omits temperature on reasoning models that reject it.
- Extracts the longest Python code block from a response.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from openai import OpenAI
try:
    from openai import APIError
except Exception:  # pragma: no cover
    APIError = Exception  # type: ignore

from .config import LLMConfig

logger = logging.getLogger("purple_next")


class _Heartbeat:
    def __init__(self, label: str, interval: float = 30.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self._started: float = 0.0

    def __enter__(self):
        self._started = time.time()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            logger.info(f"[llm] {self.label} still waiting {time.time() - self._started:.0f}s...")


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key or "missing-key",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=max(0, cfg.max_retries),
        )
        self._legacy_max_tokens = False
        self._omit_temperature = False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "chat",
    ) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        max_t = self.cfg.max_tokens if max_tokens is None else max_tokens
        started = time.time()
        try:
            content = self._call(messages, temp, max_t, label)
        except APIError as e:
            err = str(e)
            if "max_tokens" in err and "max_completion_tokens" in err:
                self._legacy_max_tokens = not self._legacy_max_tokens
                content = self._call(messages, temp, max_t, label)
            elif self._is_temp_rejection(err):
                self._omit_temperature = True
                content = self._call(messages, temp, max_t, label)
            else:
                raise
        logger.info(f"[llm] <- {label} OK in {time.time() - started:.0f}s chars={len(content)}")
        return content

    def _call(self, messages, temperature, max_tokens, label):
        kwargs = (
            {"max_completion_tokens": max_tokens}
            if not self._legacy_max_tokens
            else {"max_tokens": max_tokens}
        )
        if not self._omit_temperature and temperature is not None:
            kwargs["temperature"] = temperature
        with _Heartbeat(label=label):
            resp = self.client.chat.completions.create(
                model=self.cfg.model, messages=messages, **kwargs  # type: ignore[arg-type]
            )
        if not resp.choices:
            raise RuntimeError("LLM returned no choices")
        content = resp.choices[0].message.content or ""
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content

    @staticmethod
    def _is_temp_rejection(err: str) -> bool:
        e = err.lower()
        return "temperature" in e and ("does not support" in e or "only supports" in e or "unsupported" in e)

    # ── code extraction ───────────────────────────────────────────────────

    _FENCED = re.compile(r"```\s*(?:python|py|python3)?\s*\n(.*?)\n\s*```", re.DOTALL | re.IGNORECASE)
    _TILDE = re.compile(r"~~~(?:python|py)?\s*\n(.*?)\n\s*~~~", re.DOTALL | re.IGNORECASE)
    _PERMISSIVE = re.compile(r"```[^\n`]*?\n?(.*?)```", re.DOTALL)

    def extract_python_code(self, text: str) -> str:
        if not text:
            return ""
        for pattern in (self._FENCED, self._TILDE, self._PERMISSIVE):
            blocks = pattern.findall(text)
            if blocks:
                substantive = [b for b in blocks if len(b.strip()) >= 30]
                pick = max(substantive or blocks, key=len)
                return self._cleanup(pick)
        return ""

    @staticmethod
    def _cleanup(code: str) -> str:
        code = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", code)
        code = re.sub(r"\n\s*```\s*$", "", code)
        return code.strip("\n").strip()
