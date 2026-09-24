"""Rate-limit-aware Azure OpenAI chat client that returns parsed JSON.

Failure handling:
- 429 / 5xx / timeouts / connection errors: retried with Retry-After or
  exponential backoff with jitter.
- A Retry-After longer than backoff_max_s (e.g. daily quota exhausted) raises
  QuotaExhausted so the run can stop cleanly and be resumed later.
- Content-filter blocks raise ContentFiltered (the case is rejected, not retried).
- Auth / missing deployment raise FatalError.
"""

import json
import random
import re
import threading
import time

import openai
from openai import OpenAI


class LLMError(Exception):
    pass


class FatalError(LLMError):
    """Configuration problem: retrying won't help."""


class QuotaExhausted(LLMError):
    """Rate limit that won't clear soon. Stop and resume later."""


class TransientError(LLMError):
    """Retries exhausted for this request. May succeed on a later run."""


class ContentFiltered(LLMError):
    """Azure content filter blocked the prompt or completion."""


class BadOutput(LLMError):
    """Model returned unparseable or truncated output."""


def _retry_after(exc: openai.APIStatusError) -> float | None:
    headers = exc.response.headers if exc.response is not None else {}
    for key, scale in (("retry-after-ms", 0.001), ("x-ms-retry-after-ms", 0.001), ("retry-after", 1.0)):
        val = headers.get(key)
        if val:
            try:
                return float(val) * scale
            except ValueError:
                pass
    m = re.search(r"retry after (\d+) second", str(exc), re.I)
    return float(m.group(1)) if m else None


def _is_content_filter(exc: openai.APIStatusError) -> bool:
    text = str(exc).lower()
    return "content_filter" in text or "content management policy" in text or "responsibleaipolicyviolation" in text


def parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise BadOutput(f"no JSON object in output: {text[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise BadOutput(f"invalid JSON: {e}") from e


def _logprobs(choice, n_tokens: int = 3) -> list[dict] | None:
    """The first few output tokens with their top alternatives, if logprobs were requested."""
    content = choice.logprobs.content if choice.logprobs and choice.logprobs.content else None
    if not content:
        return None
    return [
        {"token": t.token, "logprob": t.logprob, "top": [[a.token, a.logprob] for a in t.top_logprobs or []]}
        for t in content[:n_tokens]
    ]


class AzureLLM:
    def __init__(self, az: dict, log=print):
        if not az["deployment"]:
            raise FatalError("No deployment set: fill azure.deployment in config.toml or set AZURE_DEPLOYMENT.")
        if not az["endpoint"] or not az["api_key"]:
            raise FatalError("AZURE_OPENAI_ENDPOINT / AZURE_API_KEY missing from .env.")
        self.az = az
        self.log = log
        self.client = OpenAI(
            base_url=az["endpoint"], api_key=az["api_key"], timeout=az["request_timeout_s"], max_retries=0
        )
        self.model = az["deployment"]
        # Dropped on the fly if the deployment rejects them.
        self.temperature = az.get("temperature")
        self.reasoning_effort = az.get("reasoning_effort") or None
        self.seed = az.get("seed")
        self.json_mode = az.get("json_mode", True)
        self._lock = threading.Lock()
        self._last_start = 0.0

    def _pace(self):
        with self._lock:
            wait = self._last_start + self.az["min_interval_s"] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_start = time.monotonic()

    def _backoff(self, attempt: int) -> float:
        base = self.az["backoff_base_s"] * (2**attempt)
        return min(self.az["backoff_max_s"], base) * random.uniform(0.7, 1.3)

    def _kwargs(self, messages, extra):
        kw = {"model": self.model, "messages": messages, "max_completion_tokens": self.az["max_output_tokens"]}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.json_mode:
            kw["response_format"] = {"type": "json_object"}
        kw.update(extra)
        return kw

    def _drop_unsupported(self, msg: str) -> bool:
        """Remove a parameter the deployment rejected. True if something was dropped."""
        msg = msg.lower()
        if "temperature" in msg and self.temperature is not None:
            self.log(f"  deployment rejects temperature={self.temperature}; omitting it")
            self.temperature = None
            return True
        if "seed" in msg and self.seed is not None:
            self.log("  deployment rejects seed; omitting it")
            self.seed = None
            return True
        if "reasoning_effort" in msg and self.reasoning_effort:
            self.log("  deployment rejects reasoning_effort; omitting it")
            self.reasoning_effort = None
            return True
        if "response_format" in msg and self.json_mode:
            self.log("  deployment rejects JSON mode; parsing JSON from plain text")
            self.json_mode = False
            return True
        return False

    def complete(self, messages: list[dict], allow_truncation: bool = False, **extra) -> tuple[str, dict]:
        """Return (content, meta). Raises one of the LLMError subclasses.

        `extra` is passed through to the API (e.g. logprobs, top_logprobs, max_completion_tokens).
        With `allow_truncation`, output cut off at the token limit is returned instead of raising
        (for answers read from the first token's logprobs).
        """
        attempt = 0
        while True:
            self._pace()
            try:
                resp = self.client.chat.completions.create(**self._kwargs(messages, extra))
            except openai.RateLimitError as e:
                wait = _retry_after(e)
                if wait is not None and wait > self.az["backoff_max_s"]:
                    raise QuotaExhausted(f"rate limited for {wait:.0f}s: {e}") from e
                wait = wait if wait is not None else self._backoff(attempt)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                wait = self._backoff(attempt)
            except openai.BadRequestError as e:
                if _is_content_filter(e):
                    raise ContentFiltered(str(e)[:500]) from e
                if self._drop_unsupported(str(e)):
                    continue
                raise FatalError(f"bad request: {e}") from e
            except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
                raise FatalError(str(e)) from e
            except openai.APIStatusError as e:
                if e.status_code and e.status_code >= 500:
                    wait = self._backoff(attempt)
                else:
                    raise FatalError(str(e)) from e
            else:
                choice = resp.choices[0]
                if choice.finish_reason == "content_filter":
                    raise ContentFiltered("completion blocked by content filter")
                content = choice.message.content or ""
                if choice.finish_reason == "length" and not allow_truncation:
                    raise BadOutput("output truncated (raise azure.max_output_tokens)")
                usage = resp.usage.model_dump() if resp.usage else {}
                return content, {
                    "model_version": resp.model,
                    "system_fingerprint": getattr(resp, "system_fingerprint", None),
                    "seed": self.seed,
                    "temperature": self.temperature,
                    "usage": usage,
                    "finish_reason": choice.finish_reason,
                    "logprobs": _logprobs(choice),
                }

            attempt += 1
            if attempt > self.az["max_retries"]:
                raise TransientError(f"gave up after {attempt} attempts")
            self.log(f"  rate limited / transient; waiting {wait:.0f}s (attempt {attempt}/{self.az['max_retries']})")
            time.sleep(wait)

    def chat_json(self, messages: list[dict], validate=None) -> tuple[dict, dict]:
        """Get a JSON object, asking the model to repair output that fails `validate`.

        `validate(obj)` returns a list of problems (empty if valid).
        Returns (obj, meta). Raises BadOutput if still invalid after the repair rounds.
        """
        messages = list(messages)
        meta = {"calls": 0, "usage": {}, "repairs": []}
        for round_ in range(self.az["max_repair_rounds"] + 1):
            content, m = self.complete(messages)
            meta["calls"] += 1
            for k in ("model_version", "system_fingerprint", "seed", "temperature"):
                meta[k] = m.get(k)
            for k, v in m["usage"].items():
                if isinstance(v, int):
                    meta["usage"][k] = meta["usage"].get(k, 0) + v
            try:
                obj = parse_json(content)
                problems = validate(obj) if validate else []
            except BadOutput as e:
                problems = [str(e)]
            if not problems:
                return obj, meta
            meta["repairs"].append(problems)
            if round_ < self.az["max_repair_rounds"]:
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Your output failed validation:\n- "
                        + "\n- ".join(problems)
                        + "\nReturn the corrected JSON object only, following all the original instructions.",
                    },
                ]
        raise BadOutput("; ".join(problems))
