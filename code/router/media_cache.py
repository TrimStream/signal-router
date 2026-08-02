"""Local cache for media analysis results.

Image descriptions and voice-note transcriptions each cost one Gemini call
against a media rate limit far tighter than the text limit. The free tier
allows retrying and iterating on the text/classification prompt endlessly, but
re-running the pipeline should never re-analyze the same 15 images or 13 voice
notes it already has an answer for.

Cache is keyed by media_id, which is stable across the organizer-provided
dataset, so this cache is shipped with the submission rather than gitignored:
a grader re-running the pipeline benefits from it too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Final, Literal

MediaKind = Literal["image", "voice"]

# <repo_root>/cache/media — this file lives at <repo_root>/code/router/.
CACHE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "cache" / "media"


def _cache_path(media_id: str) -> Path:
    return CACHE_DIR / f"{media_id}.json"


def get_or_create(
    media_id: str, kind: MediaKind, generate: Callable[[], str]
) -> str:
    """Return the cached analysis for ``media_id``, generating it on a miss.

    ``generate`` is only called when no cache entry exists, so it is safe to
    make it an expensive API call.
    """
    path = _cache_path(media_id)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("media_id") == media_id and cached.get("kind") == kind:
            return str(cached["result"])

    result = generate()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"media_id": media_id, "kind": kind, "result": result}, indent=2),
        encoding="utf-8",
    )
    return result


def is_cached(media_id: str) -> bool:
    """Whether a cache entry already exists, without triggering generation."""
    return _cache_path(media_id).is_file()
