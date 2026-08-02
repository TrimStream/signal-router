"""Entry point: dataset/messages.csv -> dataset/output.csv.

Runs the full pipeline per message — retrieve -> safety gate -> classify —
and writes one row per message_id in the exact required column order.

    uv run python code/main.py                 # the real, full run
    uv run python code/main.py --limit 8 --dry-run   # wiring smoke test, no API calls

Guarantees exactly one output row per message_id in messages.csv, even when a
step raises unexpectedly: any exception during retrieval, the safety gate, or
classification for one message becomes a safe fallback row for that message,
never a missing row and never a crashed run. The rest of the batch keeps going
regardless of what happened to any single message.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Modules live in <repo_root>/code/router/, imported as `router.*`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from router.classify import (  # noqa: E402
    Classification,
    ClassifyOutcome,
    build_fallback_classification,
    build_media_index,
    classify_message,
)
from router.data import (  # noqa: E402
    DATASET_DIR,
    OUTPUT_COLUMNS,
    load_business_accounts,
    load_images,
    load_message_events,
    load_message_history,
    load_messages,
    load_voice_notes,
)
from router.retrieval import retrieve_similar  # noqa: E402
from router.safety_gate import build_business_index, evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route dataset/messages.csv to dataset/output.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N rows of messages.csv (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="skip real LLM calls; use a deterministic zero-cost stub for the "
        "non-gate-triggered path, to test pipeline wiring and CSV output "
        "without spending API quota",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path override (default: dataset/output.csv for a full, "
        "non-limited, non-dry-run pass; a cache/ scratch path otherwise, so a "
        "test run can never accidentally overwrite the real deliverable)",
    )
    return parser.parse_args()


def _dry_run_classify(
    message_row: pd.Series,
    retrieved: list,
    verdict,
    business_row: pd.Series | None,
    media_index: dict,
) -> ClassifyOutcome:
    """Zero-API stand-in for classify_message.

    Delegates safety-gate-triggered rows to the real function — that path is
    already deterministic and free, so faking it would test less, not more.
    Only the real LLM call is replaced, with a fixed placeholder.
    """
    if verdict.triggered:
        return classify_message(message_row, retrieved, verdict, business_row, media_index)
    return ClassifyOutcome(
        classification=Classification(
            action="digest",
            message_type="unknown",
            reason="Dry-run stub output; no LLM call was made for this row.",
            confidence=0.80,
            evidence_message_ids="none",
        ),
        source="llm",
    )


def process_row(
    message_row: pd.Series,
    history_df: pd.DataFrame,
    events_df: pd.DataFrame,
    business_index: dict,
    business_by_id: pd.DataFrame,
    media_index: dict,
    classify_fn,
) -> tuple[Classification, str]:
    """Classify one message. Never raises: any failure becomes a fallback row.

    This is the boundary that makes "exactly one row per message_id" true
    regardless of what goes wrong. classify_message already retries and falls
    back internally for LLM-validation failures; this catches everything
    else — retrieval, the safety gate, or media resolution raising outright.
    """
    try:
        retrieved = retrieve_similar(message_row, history_df, events_df)
        verdict = evaluate(message_row, business_index)
        biz_id = str(message_row.get("business_id") or "")
        business_row = (
            business_by_id.loc[biz_id]
            if biz_id and biz_id in business_by_id.index
            else None
        )
        outcome = classify_fn(message_row, retrieved, verdict, business_row, media_index)
        return outcome.classification, outcome.source
    except Exception as exc:  # noqa: BLE001 - deliberate: guarantee a row no matter what fails
        print(
            f"    [ERROR] {message_row.get('message_id')}: unexpected "
            f"{type(exc).__name__}: {exc} — using fallback",
            file=sys.stderr,
        )
        return build_fallback_classification(message_row), "pipeline_error"


def run_pipeline(messages_df: pd.DataFrame, classify_fn) -> pd.DataFrame:
    history_df = load_message_history()
    events_df = load_message_events()
    business_accounts = load_business_accounts()
    business_index = build_business_index(business_accounts)
    business_by_id = business_accounts.set_index(business_accounts["business_id"].astype(str))
    media_index = build_media_index(load_images(), load_voice_notes())

    rows: list[dict[str, object]] = []
    source_counts: dict[str, int] = {}
    t_start = time.monotonic()
    n = len(messages_df)

    for i, (_, message_row) in enumerate(messages_df.iterrows(), start=1):
        classification, source = process_row(
            message_row, history_df, events_df, business_index, business_by_id, media_index, classify_fn
        )
        source_counts[source] = source_counts.get(source, 0) + 1
        rows.append(
            {
                "message_id": message_row["message_id"],
                "action": classification.action,
                "message_type": classification.message_type,
                "reason": classification.reason,
                "confidence": classification.confidence,
                "evidence_message_ids": classification.evidence_message_ids,
            }
        )
        elapsed = time.monotonic() - t_start
        print(
            f"[{i}/{n}] {message_row['message_id']} -> "
            f"{classification.action}/{classification.message_type} ({source})  "
            f"[{elapsed:.0f}s elapsed]"
        )

    print("\nsource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:<14s} {count}")

    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def verify_output(output_df: pd.DataFrame, messages_df: pd.DataFrame) -> None:
    """Exactly one row per message_id in messages.csv: no gaps, no duplicates,
    no rows for ids that were never in the input. Raises rather than writing
    a silently incomplete or malformed output.csv."""
    expected_ids = set(messages_df["message_id"].astype(str))
    got_ids = list(output_df["message_id"].astype(str))
    got_set = set(got_ids)

    missing = expected_ids - got_set
    extra = got_set - expected_ids
    seen: set[str] = set()
    duplicates = {mid for mid in got_ids if mid in seen or seen.add(mid)}

    problems = []
    if missing:
        problems.append(f"missing rows for {len(missing)} message_id(s): {sorted(missing)[:10]}")
    if extra:
        problems.append(f"rows for {len(extra)} unknown message_id(s): {sorted(extra)[:10]}")
    if duplicates:
        problems.append(f"duplicate rows for {len(duplicates)} message_id(s): {sorted(duplicates)[:10]}")
    if len(output_df) != len(messages_df):
        problems.append(f"row count mismatch: {len(output_df)} output rows vs {len(messages_df)} input rows")

    if problems:
        raise RuntimeError("output.csv coverage check failed: " + "; ".join(problems))


def main() -> int:
    args = parse_args()
    all_messages = load_messages()
    messages_df = all_messages.head(args.limit) if args.limit is not None else all_messages

    classify_fn = _dry_run_classify if args.dry_run else classify_message

    mode = "DRY RUN (no real LLM calls)" if args.dry_run else "LIVE RUN"
    scope = f"{len(messages_df)} message(s)" + (
        f" of {len(all_messages)} total (--limit active)" if args.limit is not None else ""
    )
    print(f"{mode}: {scope}")
    if not args.dry_run:
        from router import gemini_client

        print(f"model={gemini_client.MODEL_NAME}  rate_limit={gemini_client.REQUESTS_PER_MINUTE} req/min")
    print()

    output_df = run_pipeline(messages_df, classify_fn)

    is_full_real_run = args.limit is None and not args.dry_run
    if is_full_real_run:
        verify_output(output_df, messages_df)
        print("\ncoverage check passed: exactly one row per message_id, no gaps, no duplicates.")
    else:
        print(
            "\n(partial run: --limit and/or --dry-run active, so the full-coverage "
            "check against all of messages.csv is skipped)"
        )

    out_path = args.out
    if out_path is None:
        out_path = (
            DATASET_DIR / "output.csv"
            if is_full_real_run
            else Path(__file__).resolve().parents[1] / "cache" / "test_output.csv"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)
    print(f"wrote {len(output_df)} row(s) to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
