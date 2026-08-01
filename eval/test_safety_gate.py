"""Measure the safety gate against labels that already exist in the dataset.

Every case here is driven by a label already present in ``sample_messages.csv``
or by text constructed inline for the exemption tests. There is no curated
message_id list, so the numbers cannot be tuned by editing a fixture.

    uv run python eval/test_safety_gate.py

Read-only: prints a report, writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from router.data import (  # noqa: E402
    load_business_accounts,
    load_message_history,
    load_sample_messages,
)
from router.safety_gate import (  # noqa: E402
    SafetyVerdict,
    build_business_index,
    evaluate,
)

RULE = "=" * 78
THIN = "-" * 78

# Verbatim from the safety_gate module docstring, which cites these as the
# reason the exemption exists.
DOCSTRING_EXEMPTION_CASES: tuple[str, ...] = (
    "Please do not share OTP or card details in the family group",
    "Someone posted a maintenance quick-pay QR from a new number, admin "
    "please confirm before anyone scans it",
)

# Appended to each exemption case to confirm the exemption is conditional:
# a message that warns *and* solicits is still an attack.
SOLICITATION_SUFFIXES: tuple[str, ...] = (
    ". Confirm your OTP at secure-verify.in to keep your account active.",
    ". Pay now to this QR and share the card PIN to release the amount.",
)


def truncate(text: str, width: int = 88) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def as_row(text: str) -> pd.Series:
    """A minimal message row for text-only checks (no business sender)."""
    return pd.Series({"message_text": text, "business_id": ""})


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def report_scam_recall(
    sample: pd.DataFrame, index: dict[str, pd.Series]
) -> tuple[int, int]:
    section("1. RECALL — sample rows labelled message_type in {scam, spam}")
    rows = sample[sample["message_type"].isin(["scam", "spam"])]
    caught = 0
    misses: list[tuple[str, str, SafetyVerdict]] = []
    for _, row in rows.iterrows():
        verdict = evaluate(row, index)
        if verdict.triggered:
            caught += 1
            print(
                f"  TRIGGER  {row['message_id']:<16s} type={row['message_type']:<5s} "
                f"-> {verdict.message_type:<5s} {list(verdict.signals)}"
            )
        else:
            misses.append((str(row["message_id"]), str(row["message_text"]), verdict))

    print(f"\n  recall: {caught}/{len(rows)}  ({caught / len(rows):.1%})")
    if misses:
        print(f"\n  MISSED ({len(misses)}):")
        for message_id, text, verdict in misses:
            print(f"    {message_id}")
            print(f"      text:   {truncate(text)}")
            print(f"      reason: {verdict.rationale}")
    return caught, len(rows)


def report_false_positives(
    sample: pd.DataFrame, index: dict[str, pd.Series]
) -> tuple[int, int]:
    section("2. FALSE POSITIVES — sample rows with action in {notify, digest}")
    print("  These must never be muted. Any trigger here suppresses a message")
    print("  the user was supposed to see.\n")
    rows = sample[sample["action"].isin(["notify", "digest"])]
    fired: list[tuple[str, str, SafetyVerdict]] = []
    for _, row in rows.iterrows():
        verdict = evaluate(row, index)
        if verdict.triggered:
            fired.append((str(row["message_id"]), str(row["message_text"]), verdict))

    if fired:
        for message_id, text, verdict in fired:
            print(f"  FALSE POSITIVE  {message_id}  -> {verdict.message_type}")
            print(f"    text:    {truncate(text)}")
            print(f"    signals: {list(verdict.signals)}")
    else:
        print("  none — no notify/digest row was muted")

    rate = len(fired) / len(rows) if len(rows) else 0.0
    print(f"\n  false positive rate: {len(fired)}/{len(rows)}  ({rate:.1%})")
    return len(fired), len(rows)


def report_exemption(index: dict[str, pd.Series]) -> bool:
    section("3. META-DISCUSSION EXEMPTION")
    print("  A warning about fraud must pass; the same warning carrying a real")
    print("  solicitation must not.\n")
    ok = True

    print("  (a) warning alone — expect NO trigger")
    for text in DOCSTRING_EXEMPTION_CASES:
        verdict = evaluate(as_row(text), index)
        passed = not verdict.triggered
        ok &= passed
        print(f"    {'PASS' if passed else 'FAIL'}  {truncate(text, 72)}")
        if verdict.triggered:
            print(f"          unexpectedly triggered: {list(verdict.signals)}")

    print("\n  (b) same warning + direct solicitation — expect TRIGGER")
    for text in DOCSTRING_EXEMPTION_CASES:
        for suffix in SOLICITATION_SUFFIXES:
            combined = text + suffix
            verdict = evaluate(as_row(combined), index)
            passed = verdict.triggered
            ok &= passed
            print(f"    {'PASS' if passed else 'FAIL'}  ...{truncate(suffix, 66)}")
            if not passed:
                print(f"          failed to trigger on: {truncate(combined, 70)}")

    print(f"\n  exemption behaves correctly: {ok}")
    return ok


def report_history_scale(
    history: pd.DataFrame, index: dict[str, pd.Series]
) -> tuple[int, int]:
    section("4. SCALE CHECK — full message_history.csv")
    triggered = 0
    by_type: dict[str, int] = {}
    for _, row in history.iterrows():
        verdict = evaluate(row, index)
        if verdict.triggered:
            triggered += 1
            key = str(verdict.message_type)
            by_type[key] = by_type.get(key, 0) + 1

    total = len(history)
    print(f"  triggered: {triggered}/{total}  ({triggered / total:.1%})")
    for key, count in sorted(by_type.items()):
        print(f"    {key:<6s} {count}")
    print(
        "\n  Sanity band: the corpus is deliberately scam-heavy, but a rate far\n"
        "  above ~20% would mean the gate is over-firing on ordinary traffic."
    )
    return triggered, total


def main() -> int:
    sample = load_sample_messages()
    history = load_message_history()
    index = dict(build_business_index(load_business_accounts()))

    print(RULE)
    print("SAFETY GATE MEASUREMENT")
    print(RULE)

    caught, n_scam = report_scam_recall(sample, index)
    fps, n_clean = report_false_positives(sample, index)
    exemption_ok = report_exemption(index)
    fired, n_hist = report_history_scale(history, index)

    section("SUMMARY")
    print(f"  recall on labelled scam/spam      {caught}/{n_scam}  ({caught / n_scam:.1%})")
    print(f"  false positives on notify/digest  {fps}/{n_clean}  ({fps / n_clean:.1%})")
    print(f"  meta-discussion exemption         {'correct' if exemption_ok else 'BROKEN'}")
    print(f"  history trigger rate              {fired}/{n_hist}  ({fired / n_hist:.1%})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
