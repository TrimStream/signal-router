"""Per-user retrieval of similar past messages.

For an incoming message, find the most lexically similar messages in the same
user's history and attach how that user reacted to each. Downstream stages read
the reaction signature, not the text, to decide the routing action.

Retrieval is scoped to one user because the evidence always is: every
``evidence_message_ids`` value in ``sample_messages.csv`` points at a message in
that same user's history. Cross-user matches would be noise.

Pure and side-effect free: no file I/O, no printing, no network calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_TOP_K: Final[int] = 3


@dataclass(frozen=True)
class RetrievedMessage:
    """A past message from the user's history, with their reaction to it."""

    message_id: str
    message_text: str
    similarity: float
    created_at: pd.Timestamp | None
    conversation_type: str
    opened: int | None
    replied: int | None
    dismissed: int | None
    muted: int | None
    reported: int | None
    reaction_time_minutes: float | None

    @property
    def has_event(self) -> bool:
        """Whether a reaction was recorded for this message at all."""
        return self.opened is not None

    @property
    def reaction_signature(self) -> str:
        """Compact reaction summary, e.g. ``opened,replied`` or ``no-event``."""
        if not self.has_event:
            return "no-event"
        flags = [
            name
            for name, value in (
                ("opened", self.opened),
                ("replied", self.replied),
                ("dismissed", self.dismissed),
                ("muted", self.muted),
                ("reported", self.reported),
            )
            if value == 1
        ]
        return ",".join(flags) if flags else "none-set"


def _to_optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _to_optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def retrieve_similar(
    message_row: pd.Series,
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedMessage]:
    """Return the ``top_k`` most similar past messages for this user.

    Returns an empty list when there is nothing usable to retrieve against:
    no history for the user, or a query with no text. An empty result means
    "no usable evidence" and should surface downstream as
    ``evidence_message_ids=none`` rather than as a low-scoring guess.
    """
    if top_k <= 0:
        return []

    user_id = str(message_row["user_id"])
    query_text = _clean_text(message_row.get("message_text"))

    user_history = history_df[history_df["user_id"] == user_id]

    # Guard against a message retrieving itself when history and query come
    # from the same frame.
    query_id = message_row.get("message_id")
    if query_id is not None and not pd.isna(query_id):
        user_history = user_history[user_history["message_id"] != str(query_id)]

    # Guard A: no history to retrieve from.
    if user_history.empty:
        return []

    # Guard B: no query text. Voice notes arrive with empty `message_text`,
    # and TF-IDF on an empty string yields a zero vector that scores 0.0
    # against everything. Bail out rather than emit a meaningless ranking;
    # these messages need their evidence from transcription, not from text.
    if not query_text:
        return []

    history_texts = [_clean_text(text) for text in user_history["message_text"]]

    # Fit on history plus the query together. Per-user corpora are tiny (a
    # median of 10 documents), so fitting on history alone would drop query
    # terms that never appear in it out of the vocabulary entirely.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([*history_texts, query_text])
    except ValueError:
        # Raised as "empty vocabulary" when every document reduces to stop
        # words or punctuation.
        return []

    history_matrix = matrix[:-1]
    query_vector = matrix[-1]

    similarities = cosine_similarity(query_vector, history_matrix).ravel()

    # Name avoids a leading underscore: itertuples renames such columns
    # positionally, which would break attribute access below.
    ranked = user_history.assign(similarity_score=similarities)
    # Tie-break on message_id so equal scores produce a stable order rather
    # than depending on row order.
    ranked = ranked.sort_values(
        by=["similarity_score", "message_id"], ascending=[False, True]
    ).head(top_k)

    return _attach_events(ranked, events_df, user_id)


def _attach_events(
    ranked: pd.DataFrame, events_df: pd.DataFrame, user_id: str
) -> list[RetrievedMessage]:
    """Join ranked history rows to their reaction events and build results."""
    # The composite (user_id, message_id) key is required: the same message_id
    # can appear for more than one recipient.
    user_events = events_df[events_df["user_id"] == user_id]
    events_by_id = {
        str(row.message_id): row for row in user_events.itertuples(index=False)
    }

    results: list[RetrievedMessage] = []
    for row in ranked.itertuples(index=False):
        event = events_by_id.get(str(row.message_id))
        created_at = getattr(row, "created_at", None)
        results.append(
            RetrievedMessage(
                message_id=str(row.message_id),
                message_text=_clean_text(row.message_text),
                similarity=float(row.similarity_score),
                created_at=None if created_at is None or pd.isna(created_at) else created_at,
                conversation_type=_clean_text(row.conversation_type),
                opened=_to_optional_int(getattr(event, "message_opened", None)),
                replied=_to_optional_int(getattr(event, "message_replied", None)),
                dismissed=_to_optional_int(
                    getattr(event, "notification_dismissed", None)
                ),
                muted=_to_optional_int(getattr(event, "muted_after_message", None)),
                reported=_to_optional_int(getattr(event, "message_reported", None)),
                reaction_time_minutes=_to_optional_float(
                    getattr(event, "reaction_time_minutes", None)
                ),
            )
        )
    return results
