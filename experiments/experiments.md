# Experiment log

**Every run goes here, including the failures.** An unlogged run does not exist:
if it is not in this table we cannot reproduce it, cannot compare against it, and
cannot describe it in the methodology document that Amazon reviews.

Local F0.5 is always measured on the held-out 15% of Source 1 records
(`VALIDATION_FRACTION` in `config.py`), scored with `metric.py`. Never quote a
number from a different split. If the split changes, every row above the change
becomes meaningless and must be marked as such.

## Reference points

| Fact | Value |
|---|---|
| All-empty submission (predict nothing for everyone) | **0.0561** |
| Singleton rate in training ground truth | 5.58% |
| Mean matches per Source 1 record | 3.46 |
| Candidate recall target for blocking | 0.95 or better |
| Candidate recall we currently get | **0.7123** |

## Runs

| # | Date | Who | Phase | What changed | Cand. recall | Local F0.5 | Public LB | Uploaded? |
|---|---|---|---|---|---|---|---|---|
| 0 | 25 Sep | n/a | n/a | Reference: predict empty for everyone | n/a | 0.0561 | n/a | no |
| 1 | 25 Sep | Claude | 1 | TF-IDF cosine on core name, blocked by country, block-min 0.30, accept 0.92, exclusivity off | 0.7123 | **0.5659** | | not yet |
| 2 | 25 Sep | Claude | 1 | Same, blocking cutoff raised to 0.60 | 0.6554 | 0.5658 | | not yet |

Run 2 exists to show that tightening the blocking cutoff from 0.30 to 0.60 costs
nothing at this accept threshold (0.5658 against 0.5659) while halving the number
of candidate pairs held in memory. The test run uses 0.60 for that reason. It
does lower the reported candidate recall, which matters for phase 2, not for this
score.

### Threshold sweep, run 1, full held-out split

| Accept threshold | Exclusivity off | Exclusivity on |
|---|---|---|
| 0.45 | 0.3149 | 0.5234 |
| 0.55 | 0.3575 | 0.5336 |
| 0.65 | 0.4172 | 0.5415 |
| 0.75 | 0.4969 | 0.5544 |
| 0.85 | 0.5519 | 0.5622 |
| **0.92** | **0.5659** | 0.5583 |
| 0.95 | 0.5652 | 0.5529 |
| 0.97 | 0.5596 | 0.5456 |
| 0.99 | 0.5567 | 0.5421 |

Two things to read off this table:

- **The curve turns over at 0.92**, so that is a real optimum and not just the
  edge of the range we happened to test. We extended the sweep to 0.99 to check.
- **Exclusivity is worth a lot when the threshold is loose and slightly negative
  when it is tight.** At 0.45 it lifts the score from 0.3149 to 0.5234, because it
  throws out contested pairs that a loose threshold would otherwise accept. By
  0.92 the threshold has already removed those pairs, so exclusivity only costs a
  little recall. Keep it in the toolbox for phase 4, where a model score replaces
  the raw cosine and the useful operating point will move again.

## Method notes, read these before trusting any number above

**The validation candidate pool is constructed, not just held out.** Blocking
scans candidate to record: every Source 2 or 3 record proposes the few Source 1
records it most resembles. How selective that is depends on how many records it
is choosing between. At test time that number is 1,732,544.

An early version of this harness held out 20,000 records but still showed every
candidate all 10 million rows. Each candidate then dumped its nearest guess onto
that small query set. It reported 961 candidates per record and a score that
meant nothing. **Do not undo this.**

The harness now rebuilds the test set's own proportions: for N held-out records
it uses their true matches plus enough random distractors to reach 5.75
candidates per record, which is the test ratio (9,969,589 candidates over
1,732,544 records).

**This is still slightly optimistic and we should not pretend otherwise.** With
329k held-out records a candidate competes for attention against 329k of them,
not against 1.73 million, so true matches survive the top-k more often than they
will at test time. We saw this directly:

| Held-out records used | Candidate recall | Local F0.5 |
|---|---|---|
| 40,000 sample | 0.7971 | 0.6966 |
| Full held-out split | 0.7123 | 0.5659 |

**The score falls as the record count rises.** So expect the leaderboard to come
in below 0.5659. Treat the gap between the public score and this number as
information about how much the top-k blocking loses at full scale, and write it
back here once we have it.

## Upload budget

15 total, 5 per day. The team leader is the only person who uploads.

| Day | Used | Remaining | Notes |
|---|---:|---:|---|
| 25 Sep | 0 | 5 | baseline ready once the official validator passes |
| 26 Sep | 0 | 5 | |
| 27 Sep | 0 | 5 | keep one in hand until the end |

## Open questions

- [ ] **Candidate recall is 0.7123, so 29% of true matches are never even looked
      at.** This is the single biggest cap on our score and it is exactly what
      phase 2 exists to fix. The most promising fix is to add a record to
      candidate scan alongside the current candidate to record one and take the
      union, so a true match can be found from either direction.
- [ ] How much does transliteration actually recover? Measure candidate recall on
      the Devanagari subset specifically, with it and without it.
- [ ] Does exclusivity help once a trained model replaces the raw cosine? The
      sweep above suggests it depends entirely on where the operating point lands.
- [ ] How far below 0.5659 does the public leaderboard actually land? That gap
      measures what our validation harness cannot simulate.
