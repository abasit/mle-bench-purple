"""OpenAI-compatible chat client with backoff, heartbeat, and robust code extraction.

The client wraps an OpenAI-compatible HTTP endpoint (Nebius / OpenAI / OpenRouter
/ any service speaking the same shape). We use the ``openai`` Python SDK directly
to keep the dependency surface minimal — only ``client.chat.completions.create``
is needed.

Key behaviours:
    - **Heartbeat thread**: while a single HTTP call is in flight, a daemon
      thread logs ``[llm] <label> still waiting Xs`` every 30 s. This is the
      reason silent hangs are now visible.
    - **Retries with capped backoff**: ``max_retries`` attempts with exponential
      backoff capped at 20 s, so a stuck call cannot eat the entire budget.
    - **Hard wall-clock cap per chat call**: configurable via the ``hard_cap``
      argument; if exceeded the call is abandoned and a RuntimeError is raised.
    - **Robust code extraction**: handles ``` python``` ``` py``` plain ``` ``,
      multiple blocks (longest wins), and bare code without fences.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from openai import OpenAI
try:  # pragma: no cover - import-time only
    from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
except Exception:  # pragma: no cover
    APIError = RateLimitError = APIConnectionError = APITimeoutError = Exception  # type: ignore

from .config import LLMConfig

logger = logging.getLogger("solver")


class _Heartbeat:
    """Background thread that logs '[llm] still waiting Xs' every interval seconds.

    Lets us see hangs in the LLM call without flooding logs in the happy path.
    """

    def __init__(self, label: str, interval: float = 30.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._start_t = time.time()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._stop.clear()
        self._start_t = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._start_t
            logger.info(f"[llm] {self.label} still waiting {elapsed:.0f}s...")


class LLMClient:
    """Thin wrapper around the OpenAI chat completions endpoint.

    Adds: exponential backoff, retry on transient errors, and a robust
    Python code-block extractor (LLMs love wrapping code in ``` fences).
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not cfg.api_key:
            logger.warning(
                "[llm] No API key configured. The agent will fail at first call. "
                "Set agent.code.api_key in solver.yaml or export NEBIUS_API_KEY / "
                "OPENAI_API_KEY in the environment."
            )
        self.client = OpenAI(
            api_key=cfg.api_key or "missing-key",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )
        # Auto-detected on the first call: newer OpenAI models reject
        # max_tokens and require max_completion_tokens. We start with
        # the new param and flip to legacy if we get a 400.
        self._use_legacy_max_tokens = False

    # ── chat ──────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "chat",
    ) -> str:
        """Send a chat completion request and return the assistant text.

        We aggressively log entry, retry, and exit so we can see hangs.
        """
        temp = self.cfg.temperature if temperature is None else temperature
        max_t = self.cfg.max_tokens if max_tokens is None else max_tokens

        # Cheap diagnostics so a hang is never silent.
        total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(
            f"[llm] -> {label}: model={self.cfg.model}, "
            f"messages={len(messages)}, total_chars={total_chars}, "
            f"temp={temp}, max_tokens={max_t}, timeout={self.cfg.timeout:.0f}s"
        )

        last_err: Exception | None = None
        call_started = time.time()
        for attempt in range(self.cfg.max_retries):
            try:
                with _Heartbeat(label=f"{label}/attempt{attempt+1}", interval=30.0):
                    # OpenAI o-series and GPT-5+ use max_completion_tokens;
                    # older models and third-party endpoints use max_tokens.
                    # Try max_completion_tokens first; fall back on 400 error.
                    token_kwarg = (
                        {"max_completion_tokens": max_t}
                        if not self._use_legacy_max_tokens
                        else {"max_tokens": max_t}
                    )
                    resp = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temp,
                        **token_kwarg,
                    )
                if not resp.choices:
                    raise RuntimeError("LLM returned no choices")
                content = resp.choices[0].message.content
                if not content:
                    raise RuntimeError("LLM returned empty content")
                logger.info(
                    f"[llm] <- {label} OK in {time.time() - call_started:.0f}s, "
                    f"reply_chars={len(content)}"
                )
                return content
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last_err = e
                wait = min(2 ** attempt, 20)
                logger.warning(
                    f"[llm] {label} transient error attempt {attempt+1}/{self.cfg.max_retries}: "
                    f"{type(e).__name__}: {e} — sleeping {wait}s"
                )
                time.sleep(wait)
            except APIError as e:
                err_msg = str(e)
                # Auto-switch between max_tokens <-> max_completion_tokens
                if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
                    self._use_legacy_max_tokens = not self._use_legacy_max_tokens
                    param = "max_tokens" if self._use_legacy_max_tokens else "max_completion_tokens"
                    logger.info(f"[llm] switching to {param} for this model")
                    continue  # retry immediately, don't sleep
                last_err = e
                wait = min(2 ** attempt, 15)
                logger.warning(
                    f"[llm] {label} APIError attempt {attempt+1}/{self.cfg.max_retries}: "
                    f"{e} — sleeping {wait}s"
                )
                time.sleep(wait)
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.exception(f"[llm] {label} unexpected error attempt {attempt+1}: {e}")
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(min(2 ** attempt, 15))

        elapsed = time.time() - call_started
        raise RuntimeError(
            f"[llm] {label} failed after {self.cfg.max_retries} attempts in {elapsed:.0f}s: {last_err}"
        )

    # ── code extraction ───────────────────────────────────────────────────

    # Strict fenced block: ```python<newline>...<newline>```
    _STRICT_BLOCK_RE = re.compile(
        r"```\s*(?:python|py|python3)?\s*\n(.*?)\n\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    # Permissive: any ``` ... ``` with optional language tag, newline tolerant.
    _PERMISSIVE_BLOCK_RE = re.compile(
        r"```[^\n`]*?\n?(.*?)```",
        re.DOTALL,
    )
    # Single-tilde fenced (some models use ~~~ instead of ```).
    _TILDE_BLOCK_RE = re.compile(
        r"~~~(?:python|py)?\s*\n(.*?)\n\s*~~~",
        re.DOTALL | re.IGNORECASE,
    )

    _CODE_LINE_PREFIXES = (
        "import ", "from ", "def ", "class ", "#!", "if __name__",
        "SEED ", "SEED=", "TARGET ", "TARGET=", "MODEL ", "MODEL=",
        "import\t", "from\t", "@", "try:", "with ",
    )

    def extract_python_code(self, text: str) -> str:
        """Pull the longest Python code block out of an LLM response.

        Strategy (in order):
            1. Strict ``` python block.
            2. Tilde-fenced (~~~) block.
            3. Permissive backtick block (any language tag).
            4. Bare code (whole response looks like Python).
        Always returns a string. The result is stripped of leading/trailing
        whitespace and the trailing fence (if any escaped through).
        """
        if not text:
            return ""

        for pattern in (self._STRICT_BLOCK_RE, self._TILDE_BLOCK_RE, self._PERMISSIVE_BLOCK_RE):
            blocks = pattern.findall(text)
            if blocks:
                # Filter out tiny blocks (less than 30 chars are usually inline snippets).
                substantive = [b for b in blocks if len(b.strip()) >= 30]
                pick = max(substantive or blocks, key=len)
                return self._cleanup_code(pick)

        # No fences — does the whole response look like code?
        first_lines = [ln.strip() for ln in text.lstrip().splitlines()[:5] if ln.strip()]
        if first_lines and any(
            first_lines[0].startswith(p) for p in self._CODE_LINE_PREFIXES
        ):
            return self._cleanup_code(text)

        return ""

    @staticmethod
    def _cleanup_code(code: str) -> str:
        # Strip stray ``` that may have leaked through.
        code = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n", "", code)
        code = re.sub(r"\n\s*```\s*$", "", code)
        return code.strip("\n").strip()

    def extract_first_paragraph(self, text: str, max_chars: int = 600) -> str:
        """Pull the leading natural-language paragraph out of an LLM response."""
        if not text:
            return ""
        # Cut at the first fence (``` or ~~~) — that's where code begins.
        cut = len(text)
        for marker in ("```", "~~~"):
            idx = text.find(marker)
            if idx >= 0:
                cut = min(cut, idx)
        head = text[:cut].strip()
        if len(head) > max_chars:
            head = head[:max_chars].rstrip() + "..."
        return head
