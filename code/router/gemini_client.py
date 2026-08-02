"""Thin wrapper around the Gemini API.

Isolates every SDK-specific detail — client construction, model name, MIME
types, structured-output config, media rate-limit pacing — so callers deal in
plain strings and never touch the `google.genai` types directly.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

# This file lives at <repo_root>/code/router/gemini_client.py.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Bare load_dotenv() searches from the caller's file location, which is
# unreliable once this module is imported from scripts/eval files elsewhere
# in the tree. Point it at the repo root explicitly instead.
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

# gemini-2.5-flash's free-tier daily quota (20 requests/day, measured via a
# live 429 body) is exhausted on this key. gemini-2.5-flash-lite is itself a
# dead model name (404, "no longer available to new users" — another example
# of why model names are verified live rather than trusted from docs).
# gemini-flash-lite-latest is confirmed working on a separate quota bucket,
# with a 6-row stratified diagnostic (2 per action) showing 100% action match
# and reason quality matching the expected template style — good enough to
# commit to, not just flag as a fallback. Still overridable via GEMINI_MODEL
# since the model landscape moves faster than this codebase.
MODEL_NAME: Final[str] = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

# IANA-registered MIME type for MP3, not the commonly-tolerated but
# non-standard "audio/mp3". All voice notes in dataset/voice_notes.csv are
# physically .mp3 regardless of what any extension implies, so this is a
# fixed constant rather than something sniffed from a file path.
AUDIO_MIME_TYPE: Final[str] = "audio/mpeg"

# All 20 images in dataset/images.csv are physically .jpg.
IMAGE_MIME_TYPE: Final[str] = "image/jpeg"

# Free-tier media rate limits are much tighter than text. This fires only
# after a real (non-cached) media call — describe_image/transcribe_voice are
# the only two functions that make one.
MEDIA_DELAY_SECONDS: Final[float] = float(
    os.environ.get("GEMINI_MEDIA_DELAY_SECONDS", "20")
)

# Measured empirically, not assumed: a first real run made 6 unpaced text
# calls in under 30s and hit 429 RESOURCE_EXHAUSTED. The API's own error body
# reported "GenerateRequestsPerMinutePerProjectPerModel-FreeTier" quotaValue=5
# for this model on this key — tighter than the ~10 RPM some docs suggested,
# and it applies to every call (text or media), not just media ones. A second
# run paced at 4/min still hit two separate 429 bursts, so the server-side
# window enforces more strictly than a clean client-side sliding average —
# paced with real headroom below the measured ceiling, not against it.
REQUESTS_PER_MINUTE: Final[int] = int(os.environ.get("GEMINI_RPM_LIMIT", "3"))

# 429 (quota) and 503 (the free tier's shared capacity is genuinely
# overloaded — "high demand", no fault of this client) are both transient and
# worth retrying. Other 4xx (400 bad request, 404 unknown model, 401/403 auth)
# are real errors that retrying cannot fix.
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
_MAX_API_RETRIES: Final[int] = 4
_DEFAULT_BACKOFF_SECONDS: Final[float] = 15.0
_RETRY_DELAY_RE: Final[re.Pattern[str]] = re.compile(r"(\d+(?:\.\d+)?)\s*s")

_call_timestamps: Final[deque[float]] = deque()

_IMAGE_PROMPT: Final[str] = (
    "Describe this image factually in 2-3 sentences for someone deciding "
    "whether a WhatsApp message containing it is safe and relevant. Note any "
    "visible text, offers, urgency language, brand names, or payment/QR "
    "codes. Do not add opinions or recommendations, only observations."
)

_VOICE_PROMPT: Final[str] = (
    "Transcribe this voice note verbatim. If it is not in English, "
    "transcribe it in its original language/script and then give an English "
    "translation on the next line prefixed with 'Translation:'. Output only "
    "the transcription (and translation if applicable), nothing else."
)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Return a process-wide Gemini client, constructed on first use."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _wait_for_rate_limit_slot() -> None:
    """Block until another call would stay under REQUESTS_PER_MINUTE in the
    trailing 60-second window. Shared by every call site — the free-tier
    budget is per-model, not per-function, so text and media calls draw from
    the same pool."""
    now = time.monotonic()
    while _call_timestamps and now - _call_timestamps[0] > 60:
        _call_timestamps.popleft()
    if len(_call_timestamps) >= REQUESTS_PER_MINUTE:
        wait = 60 - (now - _call_timestamps[0]) + 0.5
        if wait > 0:
            print(
                f"  [rate limit] pacing: waiting {wait:.1f}s to stay under "
                f"{REQUESTS_PER_MINUTE} req/min"
            )
            time.sleep(wait)
    _call_timestamps.append(time.monotonic())


def _extract_retry_delay(exc: errors.APIError) -> float | None:
    """Pull the API's own suggested retry delay out of an error body.

    Present on 429 (RetryInfo). Not present on 503 ("high demand") — those
    fall back to the caller's own exponential backoff.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    error_details = details.get("error", {}).get("details", [])
    for entry in error_details:
        if "RetryInfo" not in str(entry.get("@type", "")):
            continue
        delay = entry.get("retryDelay")
        if isinstance(delay, str):
            match = _RETRY_DELAY_RE.search(delay)
            if match:
                return float(match.group(1))
    return None


class DailyQuotaExhausted(RuntimeError):
    """The free tier's per-day request budget for this model is exhausted.

    Retrying cannot fix this within the same day — every retry attempt is
    itself another request against the same exhausted daily counter, so
    retrying makes the problem worse, not better. Raised immediately instead
    of going through the backoff loop.
    """


def _is_daily_quota_error(exc: errors.APIError) -> bool:
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return False
    error_details = details.get("error", {}).get("details", [])
    for entry in error_details:
        for violation in entry.get("violations", []):
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _call(
    contents: object, config: types.GenerateContentConfig | None = None
) -> types.GenerateContentResponse:
    """The one path every real API call goes through.

    Rate-limited proactively via _wait_for_rate_limit_slot, and resilient to
    the free tier's two distinct transient failure modes: 429 (quota, which
    includes its own suggested retry delay) and 503 (shared capacity
    overloaded, which does not — exponential backoff fills the gap there).
    """
    client = get_client()
    last_exc: errors.APIError | None = None
    for attempt in range(_MAX_API_RETRIES + 1):
        _wait_for_rate_limit_slot()
        try:
            return client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 429 and _is_daily_quota_error(exc):
                raise DailyQuotaExhausted(
                    f"Daily free-tier request quota exhausted for model "
                    f"{MODEL_NAME!r}. Retrying will not help until it resets; "
                    f"see: {exc.message}"
                ) from exc
            if code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_API_RETRIES:
                raise
            last_exc = exc
            delay = _extract_retry_delay(exc)
            source = "API-suggested"
            if delay is None:
                delay = _DEFAULT_BACKOFF_SECONDS * (2**attempt)
                source = "exponential backoff"
            print(
                f"  [retry] {code} on attempt {attempt + 1}, backing off "
                f"{delay:.1f}s ({source})"
            )
            time.sleep(delay + 1.0)
    assert last_exc is not None
    raise last_exc


def describe_image(image_bytes: bytes) -> str:
    """One real API call: a factual description of an image message."""
    response = _call(
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=IMAGE_MIME_TYPE),
            _IMAGE_PROMPT,
        ],
    )
    text = (response.text or "").strip()
    time.sleep(MEDIA_DELAY_SECONDS)
    return text


def transcribe_voice(audio_bytes: bytes) -> str:
    """One real API call: a transcription of a voice-note message."""
    response = _call(
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=AUDIO_MIME_TYPE),
            _VOICE_PROMPT,
        ],
    )
    text = (response.text or "").strip()
    time.sleep(MEDIA_DELAY_SECONDS)
    return text


def generate_structured(prompt: str, schema: type[BaseModel]) -> str:
    """One structured call. Returns the raw JSON text; callers validate it.

    Schema enforcement on the SDK side is a hint, not a guarantee of business
    rules like confidence bands or evidence-ID membership — validation is the
    caller's responsibility either way, so this deliberately returns text
    rather than a pre-parsed object.
    """
    response = _call(
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return (response.text or "").strip()
