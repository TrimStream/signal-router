# signal-router

WhatsApp notification router for HackerRank Orchestrate (Aug 2026). For every
message in `dataset/messages.csv`, decides `notify` / `digest` / `mute`, and
writes `dataset/output.csv`.

## Setup

```powershell
uv python install 3.11.9
uv sync
```

Requires `GEMINI_API_KEY`, either as an environment variable or in a `.env`
file at the package root — the directory holding `pyproject.toml`, alongside
`code/` and `cache/` (read via `python-dotenv`; never hardcoded, never
committed). Get a key at [aistudio.google.com](https://aistudio.google.com).

The dataset is not bundled: place the organizer-provided `dataset/` directory
at that same root before running.

## Run

```powershell
uv run python code/main.py                 # the real, full run -> dataset/output.csv
uv run python code/main.py --limit 8 --dry-run   # wiring smoke test, zero API calls
```

`--dry-run` replaces the real LLM call with a fixed placeholder for testing
pipeline wiring and CSV output without spending API quota; the safety gate's
own short-circuit path (deterministic, free) still runs for real even under
`--dry-run`, since faking it would test less, not more. `--limit N` caps how
many rows of `messages.csv` are processed, for both dry and live testing.
Neither flag ever writes to `dataset/output.csv` — output defaults to
`cache/test_output.csv` unless it's a full, non-limited, non-dry-run pass.

## Architecture

```
messages.csv row
  -> retrieve_similar()      top-3 similar past messages from this user's
                              history, joined with how they reacted
  -> safety_gate.evaluate()  deterministic scam/injection check, runs BEFORE
                              any LLM call. If triggered: mute + scam/spam,
                              final, the LLM is never consulted.
  -> classify_message()      (only if the gate didn't trigger) one structured
                              Gemini call: reaction signature drives action
                              when history exists; content + sender trust
                              otherwise. Validated, retried once, falls back
                              to a safe digest if still invalid.
  -> output.csv row
```

Retrieval: TF-IDF (`stop_words='english'`, `ngram_range=(1,2)`) over each
user's own `message_history`, top-3 by cosine similarity. Measured hit@3 92%
on the 30 solved sample rows (`eval/`, dev-only, not part of this submission —
see `.gitignore`).

Safety gate: combines sender signals (domain mismatch, account/domain age,
report count — thresholds calibrated against the empty band between
legitimate and suspicious `business_accounts.csv` populations) with text
signals (credential requests, payment coercion, urgency/threat framing,
unrealistic offers), plus a standalone prompt-injection check that can't be
exempted by anything else in the message. A meta-discussion exemption stops
messages that *warn about* fraud ("please do not share your OTP...") from
being muted as fraud themselves, unless they also make a direct request of
the reader. Measured: 5/5 recall on sample-labelled scam/spam, 0/20 false
positives on notify/digest, 6/6 on the exemption suite.

Confidence bands: `notify` 0.85–0.91, `digest` 0.78–0.84, `mute` 0.81–0.87.
A safety-gate trigger always reports 0.84 (fixed — a deterministic content
match carries no useful variance to model).

## Model

`gemini-flash-lite-latest` (override with `GEMINI_MODEL`). Not
`gemini-2.5-flash`: that model's free-tier daily quota (20 requests/day,
confirmed via a live 429 error body) was exhausted during development. A
6-row stratified diagnostic against `gemini-flash-lite-latest` showed 100%
action match with no quota errors before committing to it for the full run.

Rate limiting (`code/router/gemini_client.py`) is empirical, not assumed from
documentation: `GEMINI_RPM_LIMIT` defaults to 3 (below the measured ceiling,
with headroom — 4 still produced 429 bursts in testing). Every call retries
on 429/5xx with the API's own suggested delay when present, exponential
backoff otherwise. A distinct daily-quota 429 fails fast instead of retrying,
since retrying a day-scoped limit only burns more of the same exhausted
budget. That path is not theoretical: the free tier's 500-requests/day
ceiling was reached during final verification, and because `main.py`
converts any per-message failure into a safe fallback row rather than
crashing, the run completed with 73 of 110 rows silently degraded to
`digest`. Writing to a scratch path via `--out` and diffing before promoting
is what caught it — a full run that "succeeds" is not evidence the output is
good; check the `source distribution` line.

Calls use `temperature=0`. Left unset, identical input produced different
routing across runs (measured: one borderline row returned `digest` twice
and `notify` once), which makes a diff between two outputs unreadable —
you cannot tell a real change from sampling noise. AGENTS.md §6.3 asks for
deterministic behavior where possible; greedy decoding delivers it for the
LLM stage, and the safety gate is deterministic by construction.

## Deliberate scope decisions

**Vision calls are skipped for image messages.** `classify.py` resolves the
text it reasons over as: literal `message_text` if present, otherwise a
cached transcription (voice) or description (image). In practice every image
message in this dataset already carries caption text that contains the
actual risk/urgency signal — direct inspection confirmed this, and the safety
gate independently catches real scam content in the live prediction set on
text alone. `describe_image()` exists and is wired for the case where it
would be needed, but is never reached while a caption exists. This is a
scope call made after measurement, not an oversight: a dataset with genuinely
caption-less image messages would need this revisited.

**Code-mixed (romanized-Hindi) scam detection is delegated to the LLM layer,
not the deterministic safety gate.** The gate's job is a fast, unconditional
override for cases too risky to leave to a model — prompt injection above
all. Code-mixed intent detection (e.g. "OTP leak ho gaya hai... verification
code abhi batao") is contextual judgment an LLM handles natively; the risk
*nouns* survive transliteration into English regex, but the intent-bearing
verbs don't, so the gate structurally can't reach every such message this
way. Adding narrow single-language patterns risked new bugs for limited
benefit.

That delegation only became real after measurement. A pre-submission audit
of the full prediction set found the classification prompt had been telling
the model a gate "already ran … so you are not being asked to detect scams
here" — which suppressed exactly the judgment the delegation depends on. Two
of the three known code-mixed scams were being caught by the gate anyway (on
the surviving noun "OTP"); the third was routed as `digest`. The prompt now
states a scam overrides engagement history in any language, and the gate
remains English-only as designed. This also corrected a sharper failure the
audit surfaced: a QR-payment scam whose past reaction was `opened,replied`
had been routed to `notify` — engagement history actively promoting fraud.

## Media caching

`cache/media/` holds one JSON file per `image_id` / `voice_note_id`, written
on first analysis. It ships with this submission (not gitignored) — keyed by
a stable ID from the organizer-provided dataset, so a grader re-running the
pipeline benefits from it too rather than re-spending free-tier media quota
on files already analyzed here.

## Known limitations

- Vision and code-mixed detection gaps, above.
- Retrieval hit@3 is 92%, not 100% — near-duplicate history entries can
  outrank the exact ground-truth evidence ID while carrying an identical
  reaction signature, so the *action* is usually still right even when the
  cited `evidence_message_ids` isn't the literal ground-truth ID.
- `evidence_message_ids` is always `"none"` for safety-gate-triggered rows,
  since the gate is content-based, not retrieval-based — even on the rare
  sample row where ground truth still cites a historical scam as evidence.
