"""Inspect retrieval quality against the solved rows in sample_messages.csv.

Read-only: prints to stdout, writes no files, makes no LLM calls.

Sampling is stratified across notify/digest/mute rather than taking the first
N rows, because sample_messages.csv is grouped by action at the top of the file
and a head() sample would be all notify.

    uv run python scripts/verify_retrieval.py --n 9
    uv run python scripts/verify_retrieval.py --action mute --n 10
    uv run python scripts/verify_retrieval.py --n 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Modules live in <repo>/code/router/, imported as `router.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

# The Windows console defaults to cp1252, which mangles the box and ellipsis
# characters used below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from router.data import (  # noqa: E402
    ACTIONS,
    load_message_events,
    load_message_history,
    load_sample_messages,
)
from router.retrieval import RetrievedMessage, retrieve_similar  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify TF-IDF retrieval against solved sample messages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=9, help="number of sample rows to inspect"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="rotate the start position within each action stratum",
    )
    parser.add_argument(
        "--action",
        choices=ACTIONS,
        default=None,
        help="restrict to a single action stratum",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="candidates to retrieve per message"
    )
    parser.add_argument(
        "--text-width",
        type=int,
        default=88,
        help="characters of message text to display",
    )
    return parser.parse_args()


def stratified_sample(
    sample_df: pd.DataFrame, n: int, offset: int, action: str | None
) -> list[str]:
    """Pick message_ids round-robin across action strata, deterministically."""
    strata = [action] if action else list(ACTIONS)
    pools: dict[str, list[str]] = {}
    for name in strata:
        ids = sorted(
            sample_df.loc[sample_df["action"] == name, "message_id"].astype(str)
        )
        if ids:
            # Rotate rather than slice so a large offset still yields rows.
            shift = offset % len(ids)
            pools[name] = ids[shift:] + ids[:shift]

    picked: list[str] = []
    depth = 0
    while len(picked) < n and any(depth < len(p) for p in pools.values()):
        for name in strata:
            pool = pools.get(name)
            if pool and depth < len(pool) and len(picked) < n:
                picked.append(pool[depth])
        depth += 1
    return picked


def truth_ids(raw: object) -> set[str]:
    """Parse the ground-truth evidence_message_ids cell into a set."""
    text = "" if raw is None or pd.isna(raw) else str(raw).strip()
    if not text or text.lower() == "none":
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def implied_action(hit: RetrievedMessage) -> str | None:
    """Action suggested by a past reaction, per the CLAUDE.md signature table.

    Ignores the scam override, which is the safety gate's job and is not built
    yet. Scam rows are therefore expected to disagree here.
    """
    if not hit.has_event:
        return None
    if hit.reported == 1 or hit.muted == 1 or hit.dismissed == 1:
        return "mute"
    if hit.opened == 1 and hit.replied == 1:
        return "notify"
    if hit.opened == 1:
        return "digest"
    return None


def truncate(text: str, width: int) -> str:
    flat = text.replace("\n", "\\n").replace("\r", "")
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def flag(value: int | None) -> str:
    return "-" if value is None else str(value)


def main() -> int:
    args = parse_args()

    sample_df = load_sample_messages()
    history_df = load_message_history()
    events_df = load_message_events()

    picked = stratified_sample(sample_df, args.n, args.offset, args.action)
    if not picked:
        print("No rows matched the requested sample.", file=sys.stderr)
        return 1

    indexed = sample_df.set_index(sample_df["message_id"].astype(str))

    scored: list[tuple[str, str, bool, bool, float]] = []
    empty_query: list[str] = []
    truth_none: list[str] = []
    signature_rows: list[tuple[str, str, str | None, str]] = []
    hit_sims: list[float] = []
    miss_sims: list[float] = []

    print(RULE)
    scope = args.action or "stratified across " + "/".join(ACTIONS)
    print(f"RETRIEVAL VERIFICATION  |  n={args.n} offset={args.offset} top_k={args.top_k}")
    print(f"sample: {scope}  ({len(picked)} rows)")
    print(RULE)

    for message_id in picked:
        row = indexed.loc[message_id]
        truth = truth_ids(row["evidence_message_ids"])
        user_id = str(row["user_id"])
        history_size = int((history_df["user_id"] == user_id).sum())
        query = str(row["message_text"]).strip()

        hits = retrieve_similar(row, history_df, events_df, top_k=args.top_k)

        print()
        print(THIN)
        print(
            f"{message_id}  user={user_id}  truth_action={row['action']}  "
            f"truth_type={row['message_type']}"
        )
        print(
            f"  conv={row['conversation_type']}  media={str(row['media_type']) or '-'}  "
            f"forwarded={row['forwarded_count']}  history_size={history_size}"
        )

        if not query:
            print("  QUERY: [EMPTY QUERY TEXT — no text to retrieve on]")
        else:
            print(f"  QUERY: {truncate(query, args.text_width)}")

        if hits:
            print(f"  TOP-{args.top_k} RETRIEVED:")
            print(
                "    rk  message_id      sim     op rp di mu re   text"
            )
            for rank, hit in enumerate(hits, start=1):
                mark = "*" if hit.message_id in truth else " "
                print(
                    f"   {mark}{rank}.  {hit.message_id:<15s} {hit.similarity:.4f}  "
                    f"{flag(hit.opened):>2s} {flag(hit.replied):>2s} "
                    f"{flag(hit.dismissed):>2s} {flag(hit.muted):>2s} "
                    f"{flag(hit.reported):>2s}   "
                    f"{truncate(hit.message_text, args.text_width - 30)}"
                )
        else:
            print("  TOP-K RETRIEVED: (none — retrieval returned no candidates)")

        print(f"  TRUE EVIDENCE: {sorted(truth) if truth else ['none']}")

        # Verdict.
        if not query:
            empty_query.append(message_id)
            print("  VERDICT: N/A (empty query) — excluded from hit rates")
        elif not truth:
            truth_none.append(message_id)
            print("  VERDICT: N/A (ground truth is 'none') — excluded from hit rates")
        else:
            retrieved = [h.message_id for h in hits]
            hit1 = bool(retrieved) and retrieved[0] in truth
            hitk = any(mid in truth for mid in retrieved)
            covered = truth.issubset(set(retrieved))
            best = hits[0].similarity if hits else 0.0
            scored.append((message_id, str(row["action"]), hit1, hitk, best))
            (hit_sims if hitk else miss_sims).append(best)

            if hit1:
                verdict = "HIT@1"
            elif hitk:
                verdict = f"HIT@{args.top_k}"
            else:
                verdict = "MISS"
            extra = "" if covered else f"  (recall {len(truth & set(retrieved))}/{len(truth)})"
            print(f"  VERDICT: {verdict}{extra}")

        if hits:
            top = hits[0]
            signature_rows.append(
                (message_id, str(row["action"]), implied_action(top), top.reaction_signature)
            )

    # ---------------- summary ----------------
    print()
    print(RULE)
    print("SUMMARY")
    print(RULE)

    if scored:
        n_scored = len(scored)
        h1 = sum(1 for r in scored if r[2])
        hk = sum(1 for r in scored if r[3])
        print(f"\nScorable rows: {n_scored}")
        print(f"  hit@1        {h1}/{n_scored}  ({h1 / n_scored:.1%})")
        print(f"  hit@{args.top_k}        {hk}/{n_scored}  ({hk / n_scored:.1%})")

        print("\nPer action:")
        print("  action   n    hit@1          hit@k")
        for name in ACTIONS:
            rows = [r for r in scored if r[1] == name]
            if not rows:
                continue
            a1 = sum(1 for r in rows if r[2])
            ak = sum(1 for r in rows if r[3])
            print(
                f"  {name:<8s} {len(rows):<4d} {a1}/{len(rows)} ({a1 / len(rows):5.1%})   "
                f"{ak}/{len(rows)} ({ak / len(rows):5.1%})"
            )

        misses = [r[0] for r in scored if not r[3]]
        if misses:
            print(f"\nMissed rows: {', '.join(misses)}")

        if hit_sims:
            print(
                f"\nTop-1 similarity on hits:   min={min(hit_sims):.4f} "
                f"median={sorted(hit_sims)[len(hit_sims) // 2]:.4f} max={max(hit_sims):.4f}"
            )
        if miss_sims:
            print(
                f"Top-1 similarity on misses: min={min(miss_sims):.4f} "
                f"median={sorted(miss_sims)[len(miss_sims) // 2]:.4f} max={max(miss_sims):.4f}"
            )
    else:
        print("\nNo scorable rows in this sample.")

    print(f"\nExcluded — empty query text: {len(empty_query)}"
          + (f"  ({', '.join(empty_query)})" if empty_query else ""))
    print(f"Excluded — ground truth 'none': {len(truth_none)}"
          + (f"  ({', '.join(truth_none)})" if truth_none else ""))

    if signature_rows:
        agree = sum(1 for _, true_a, imp, _ in signature_rows if imp == true_a)
        total = len(signature_rows)
        print(
            f"\nTop-1 reaction signature implies the correct action: "
            f"{agree}/{total} ({agree / total:.1%})"
        )
        print("  (scam override not applied — safety gate is not built yet)")
        disagreements = [
            (mid, true_a, imp, sig)
            for mid, true_a, imp, sig in signature_rows
            if imp != true_a
        ]
        if disagreements:
            print("  disagreements:")
            for mid, true_a, imp, sig in disagreements:
                print(f"    {mid:<16s} truth={true_a:<7s} implied={str(imp):<7s} [{sig}]")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
