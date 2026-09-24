# Experiment log

**Every run goes here, including the failures.** An unlogged run does not exist:
if it is not in this table we cannot reproduce it, cannot compare against it, and
cannot describe it in the methodology document that Amazon reviews.

Local F₀.₅ is always measured on the held-out 15% of Source 1 entities
(`VALIDATION_FRACTION` in `config.py`). Never quote a number from a different
split — if the split changes, every row above the change becomes meaningless and
must be marked as such.

## Reference points

| Fact | Value |
|---|---|
| All-empty submission (predict nothing for everyone) | **0.0558** |
| Singleton rate in training ground truth | 5.58% |
| Mean matches per Source 1 entity | 3.46 |
| Candidate recall target for blocking | ≥ 0.95 |

## Runs

| # | Date | Who | Phase | What changed | Cand. recall | Local F₀.₅ | Public LB | Uploaded? | Output file |
|---|---|---|---|---|---|---|---|---|---|
| — | 2026-09-25 | — | 0 | Baseline reference: predict empty for every entity | — | 0.0558 | — | no | — |

## Upload budget

15 total, 5 per day. The team leader is the only person who uploads.

| Day | Used | Remaining | Notes |
|---|---:|---:|---|
| 25 Sep | 0 | 5 | |
| 26 Sep | 0 | 5 | |
| 27 Sep | 0 | 5 | keep one in hand until the end |

## Open questions

- [ ] `test_source3.tsv` missing from the download — leader to fetch the second
      Google Drive part.
- [ ] How much does transliteration actually recover? Measure candidate recall on
      the Devanagari subset specifically, with and without it.
- [ ] Does the exclusive-assignment constraint (each S2/S3 record belongs to at
      most one S1) improve F₀.₅ in practice, and by how much?
