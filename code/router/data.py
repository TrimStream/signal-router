"""Typed loaders for the dataset CSVs.

One function per file in ``dataset/``. These are pure I/O plus dtype
correctness: no filtering, no joins, no business logic. Anything that
interprets the data belongs in a downstream module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import pandas as pd

# This file lives at <repo>/code/router/data.py, so the repo root is two
# levels up. Verified rather than assumed: parents[1] is <repo>/code.
DATASET_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "dataset"

Action = Literal["notify", "digest", "mute"]

MessageType = Literal[
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
]

ACTIONS: Final[tuple[Action, ...]] = ("notify", "digest", "mute")

MESSAGE_TYPES: Final[tuple[MessageType, ...]] = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

_TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M"

# Columns shared by messages.csv, sample_messages.csv and message_history.csv.
_MESSAGE_STR_COLUMNS: Final[tuple[str, ...]] = (
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "message_text",
    "media_type",
    "media_id",
)

# Blank here genuinely means "absent", so an empty string is the honest value.
# Deliberately excludes the event flag columns: a missing reaction is unknown,
# not a zero, and collapsing the two would fabricate engagement history.
_MESSAGE_FILL_COLUMNS: Final[tuple[str, ...]] = (
    "group_id",
    "business_id",
    "sender_user_id",
    "message_text",
    "media_type",
    "media_id",
)

_EVENT_FLAG_COLUMNS: Final[tuple[str, ...]] = (
    "message_opened",
    "message_replied",
    "notification_dismissed",
    "muted_after_message",
    "message_reported",
)


def _read_csv(name: str, **kwargs: object) -> pd.DataFrame:
    """Read a CSV from ``dataset/``, failing loudly if it is missing."""
    path = DATASET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Expected dataset file at {path}")
    return pd.read_csv(path, **kwargs)  # type: ignore[arg-type]


def _normalise_message_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared conventions for message-shaped frames."""
    for column in _MESSAGE_FILL_COLUMNS:
        frame[column] = frame[column].fillna("")
    frame["created_at"] = pd.to_datetime(
        frame["created_at"], format=_TIMESTAMP_FORMAT
    )
    frame["forwarded_count"] = (
        pd.to_numeric(frame["forwarded_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    return frame


def _message_dtypes() -> dict[str, str]:
    return {column: "string" for column in _MESSAGE_STR_COLUMNS}


def load_messages() -> pd.DataFrame:
    """Incoming messages that require a routing prediction."""
    frame = _read_csv("messages.csv", dtype=_message_dtypes())
    return _normalise_message_frame(frame)


def load_sample_messages() -> pd.DataFrame:
    """Solved example messages, including ground-truth label columns."""
    dtypes = _message_dtypes()
    dtypes.update(
        {
            "action": "string",
            "message_type": "string",
            "reason": "string",
            "evidence_message_ids": "string",
        }
    )
    frame = _read_csv("sample_messages.csv", dtype=dtypes)
    frame = _normalise_message_frame(frame)
    frame["evidence_message_ids"] = frame["evidence_message_ids"].fillna("none")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    return frame


def load_message_history() -> pd.DataFrame:
    """Past messages received by users, used as retrieval evidence."""
    frame = _read_csv("message_history.csv", dtype=_message_dtypes())
    return _normalise_message_frame(frame)


def load_message_events() -> pd.DataFrame:
    """How users reacted to historical messages.

    Keyed on ``(user_id, message_id)``; ``message_id`` alone is not unique
    across users. Flags stay nullable so an absent reaction reads as unknown
    rather than as a negative.
    """
    dtypes: dict[str, str] = {"user_id": "string", "message_id": "string"}
    dtypes.update({column: "Int64" for column in _EVENT_FLAG_COLUMNS})
    frame = _read_csv("message_events.csv", dtype=dtypes)
    frame["reaction_time_minutes"] = pd.to_numeric(
        frame["reaction_time_minutes"], errors="coerce"
    ).astype("Float64")
    return frame


def load_users() -> pd.DataFrame:
    """Per-user notification behaviour and quiet hours."""
    return _read_csv(
        "users.csv", dtype={"user_id": "string", "do_not_disturb_window": "string"}
    )


def load_groups() -> pd.DataFrame:
    """Group chat metadata."""
    frame = _read_csv(
        "groups.csv",
        dtype={"group_id": "string", "group_name": "string", "group_type": "string"},
    )
    frame["created_at"] = pd.to_datetime(frame["created_at"], format="mixed")
    return frame


def load_group_members() -> pd.DataFrame:
    """How each user relates to each group they belong to."""
    frame = _read_csv(
        "group_members.csv",
        dtype={"group_id": "string", "user_id": "string", "role": "string"},
    )
    frame["joined_at"] = pd.to_datetime(frame["joined_at"], format="mixed")
    return frame


def load_business_accounts() -> pd.DataFrame:
    """Business sender identity, verification and domain metadata."""
    return _read_csv(
        "business_accounts.csv",
        dtype={
            "business_id": "string",
            "display_name": "string",
            "brand_name": "string",
            "category": "string",
            "official_domain": "string",
            "domain_used_by_sender": "string",
        },
    )


def load_user_business_history() -> pd.DataFrame:
    """Whether a user has an existing relationship with a business."""
    return _read_csv(
        "user_business_history.csv",
        dtype={
            "user_id": "string",
            "business_id": "string",
            "why_user_knows_account": "string",
        },
    )


def load_images() -> pd.DataFrame:
    """Image IDs and their media file paths."""
    return _read_csv(
        "images.csv", dtype={"image_id": "string", "file_path": "string"}
    )


def load_voice_notes() -> pd.DataFrame:
    """Voice note IDs and their media file paths."""
    return _read_csv(
        "voice_notes.csv", dtype={"voice_note_id": "string", "file_path": "string"}
    )


def load_daily_notification_summary() -> pd.DataFrame:
    """Daily notification load per user."""
    frame = _read_csv(
        "daily_notification_summary.csv",
        dtype={"user_id": "string", "date": "string"},
    )
    frame["date"] = pd.to_datetime(frame["date"], format="mixed")
    return frame


def resolve_media_path(file_path: str) -> Path:
    """Turn a ``media/...`` path from images/voice_notes into an absolute path."""
    return DATASET_DIR / file_path
