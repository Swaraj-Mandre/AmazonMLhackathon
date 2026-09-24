# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM_NAME}}
**Team Members:** {{TEAM_MEMBERS}}
**Submission Date:** {{DATE}}

---

## 1. Executive Summary

We treat the task as classical entity resolution: normalise aggressively, block
by country, then score candidate pairs. This submission is our calibrated
baseline, a fully rules-based pipeline with no learned model, scoring
**{{LOCAL_F05}} macro F_0.5** on a held-out split of the training data against
**0.0561** for an empty submission. Its purpose is to lock the output format and
give us a trustworthy reference point before a learned matcher is added. The work
that distinguishes it so far is in measurement rather than modelling: three
structural properties of the data (country-closed matching, one-owner-per-record,
and cross-script matches) were verified on the full training set and each one
changed the design.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings below come from `src/profile_data.py` run over the complete training
set, not from sampling intuition.

**Match structure.** 2,206,821 Source 1 records carry 7,638,365 links, a mean of
3.46 matches each, maximum 11. Only 5.58% are singletons, so an all-empty
submission scores 0.0561 and there is no cheap floor to exploit.

**Each Source 2/3 record has at most one owner.** Across all 7,638,365 links, no
Source 2 or Source 3 record is ever claimed by two different Source 1 records
(7,638,365 distinct records used, zero reuse). Source 1 is genuinely
deduplicated, so the target structure is a partition of the candidate side rather
than an arbitrary bipartite graph.

**Matches never cross country labels.** Over 1,038,755 sampled links we found
zero cross-country matches. Country is therefore a lossless blocking key. We
treat it as an open set of string labels, grouping by whatever values appear, so
the France block in the test set is handled by the same code path as any other
and no value is ever hard-coded.

**Roughly 10% of Indian matches cross writing systems.** Source 1 is always Latin
script, but 10.15% of Indian links point at a Devanagari Source 2/3 record.
Character and token overlap are identically zero across scripts, so these matches
are unreachable by any string similarity unless the scripts are unified first.

**Field quality.** Business names are never empty. Addresses are empty in about
3.3% of Source 2 and Source 3 rows and never in Source 1. Raw names carry leading
junk such as `--` and `<<` prefixes.

### 2.2 Solution Strategy

**Approach Type:** Blocking + threshold (rules baseline; blocking + classifier is
the planned successor)

**Core Innovation:** Two decisions that follow from the measurements above.
First, a self-written offline Devanagari-to-Latin transliterator, which recovers
the ~10% of Indian links that no string metric could otherwise see, without any
external service (the fair-play rules forbid lookups, so the mapping table is
ordinary source code). Second, a validation harness that reconstructs the test
set's own candidate-to-record ratio rather than scoring against the full training
pool, described in section 5.

Normalisation pipeline, applied identically to every record:

1. Transliterate Devanagari to Latin.
2. Strip leading and trailing non-word characters.
3. Unicode NFKD decomposition with combining marks removed, which folds the
   French accents so `Président` and `President` agree.
4. Case folding, punctuation to whitespace, `&` expanded to `and` before
   punctuation is stripped so it is not silently deleted.
5. Legal-form folding both ways: `corporation`/`corp`, `private`/`pvt`,
   `limited`/`ltd`, `incorporated`/`inc`, and similar.
6. A `core_name` variant with generic legal tokens removed entirely, since
   `pvt ltd` appears on a large share of Indian records and inflates similarity
   between unrelated businesses. This is the blocking key.
7. Address abbreviation folding (`road`/`rd`, `street`/`st`, directionals) and
   extraction of digit runs, longest first, so postcodes and house numbers become
   their own comparable field.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** country (exact, verified lossless), then TF-IDF cosine
  over whitespace tokens of the normalised `core_name`, fitted per country block
  so the IDF reflects that country's vocabulary. A token appearing in more than
  20% of a block's records is dropped as uninformative.
- **Candidate pairs generated:** {{CANDIDATE_PAIRS}} across {{S1_COUNT}} Source 1
  records ({{CANDIDATES_PER_ENTITY}} per record). Reduction ratio
  {{REDUCTION_RATIO}} against the full cross-product.
- **How we ensured true matches were not lost:**

The exhaustive comparison is 1,732,544 Source 1 records against 9,969,589 Source
2 and Source 3 records, about 1.7 x 10^13 pairs. Country blocking removes roughly
61% of that at zero measured recall cost. Within a block, the scan runs
**candidate to record**: each Source 2/3 record proposes the top 5 Source 1
records it most resembles above a cosine floor. That direction is chosen because
each candidate has at most one correct owner, so the search has a single target,
and because it bounds memory: each chunk of candidates produces a fixed-size
result regardless of how many records exist.

Recall is measured directly rather than assumed, using
`metric.candidate_recall()` on the held-out split. The honest figure for this
submission is **{{CAND_RECALL}}**, which is also the ceiling on any matcher placed
downstream.

We also measured how this number behaves with scale, because it is not
scale-invariant. Selectivity depends on how many records each candidate chooses
between: on a 40,000-record held-out sample recall was 0.7971, and on the full
329,000-record split it fell to 0.7123. At the test set's 1,732,544 records it
will be lower again. The single-direction scan is the cause, and the fix we are
implementing is a second scan in the opposite direction, each Source 1 record
proposing its own top candidates, with the two candidate sets unioned. We report
this rather than quote the flattering number, because the blocking recall claim
should be one a reviewer can reproduce.

---

## 4. Matching Model

**Features used:**

- Name features: TF-IDF cosine over token vectors of the normalised `core_name`,
  with IDF fitted per country block. Rare shared tokens therefore dominate the
  score, which is the intended behaviour: two records sharing an unusual token
  are far stronger evidence than two sharing `restaurant`.
- Address features: extracted but not yet used by this baseline. The normalised
  address and its ordered digit runs are computed and cached for every record.
- Other: a transliteration flag and an empty-address flag are cached per record
  for the learned model.

**Model type:** None in this submission. The decision rule is a single threshold
on the blocking cosine. This is deliberate: it establishes a verified reference
score and a working submission format before a learned matcher is introduced. The
planned successor is a gradient-boosted classifier (LightGBM) over pairwise
features, trained with hard negatives drawn from our own blocking output rather
than random pairs.

**Threshold selection method:** F_0.5 optimisation on the held-out split. The
accept threshold was swept from 0.45 to 0.99, crossed with an on/off
exclusivity constraint. The optimum is a genuine interior maximum at
**{{THRESHOLD}}**, not an endpoint; the curve rises to it and falls after it,
which is why the sweep was extended to 0.99 to confirm.

A second decision rule was tested and is reported here because its behaviour is
instructive. Since each candidate record has at most one owner, contested pairs
can be resolved by keeping only each candidate's highest-scoring record. At a
loose threshold this is worth a great deal (0.3149 to 0.5234 at 0.45), because it
removes exactly the pairs a loose threshold wrongly admits. At the optimal
threshold the cut has already been made, and the constraint only costs recall
(0.5659 against 0.5583 at 0.92). It is therefore **disabled** in this submission
and will be re-evaluated once a model score replaces the raw cosine, since the
useful operating point will move.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **{{LOCAL_F05}}** on the held-out split
  ({{VAL_ENTITIES}} Source 1 records, 15% of training, split by record). Empty
  submission reference: 0.0561.

**Threshold sweep, full held-out split:**

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

**On the validity of the validation split.** The split is by Source 1 record and
never by pair, since a pair-level split places other matches of the same record
on both sides and leaks the label. The split is seeded so every experiment scores
the same held-out records.

More importantly, the candidate pool is **reconstructed** rather than taken
wholesale. Because blocking scans candidate to record, its selectivity depends on
how many records a candidate chooses between. Showing a small held-out set the
entire 10-million-record pool makes every candidate deposit its nearest guess
onto that small set; an early version of our harness did exactly this and
reported 961 candidates per record and a meaningless score. The harness now draws
each held-out record's true matches plus enough random distractors to reach 5.75
candidates per record, which is the test set's own ratio (9,969,589 over
1,732,544). We expect the leaderboard to fall somewhat below our local figure for
the residual scale effect described in section 3, and we treat that gap as a
measurement rather than a surprise.

**Common false positives (wrong merges):** Businesses of the same chain or
franchise in the same city, where the name is identical and only the address
differs. The current baseline scores names alone, so it cannot separate them.
Generic names composed entirely of common tokens produce the same failure.
Address features exist in the cache specifically to fix this class.

**Common false negatives (missed matches):** Dominated by blocking loss rather
than threshold loss, {{MISSED_PCT}} of true matches never reach the scoring stage
at all. Within that, the recognisable groups are transliterated names where our
character mapping produces a form too distant from the English spelling (the
inherent-vowel convention means `aadity` for `aditya`), records whose names share
only high-frequency tokens, and abbreviations or trade names with no lexical
overlap with the registered name.

---

## 6. Conclusion

A carefully normalised, country-blocked TF-IDF baseline with a single tuned
threshold reaches {{LOCAL_F05}} macro F_0.5 without any learned component, an
order of magnitude above the empty-submission floor of 0.0561. The dominant
remaining loss is blocking recall rather than matching precision: {{MISSED_PCT}}
of true matches are never scored, which caps every downstream improvement. The
main lesson so far is that the evaluation harness deserved as much scrutiny as
the model, since our first version of it produced a confidently wrong number, and
that verifying structural claims about the data on the full set rather than
assuming them is what produced every design decision that mattered.

---

## Appendix

### A. Code Artefacts

The complete runnable pipeline ships under `code/business_entity_resolution/`,
with all source in `src/`, a `README.md` carrying exact run instructions, and a
`requirements.txt` pinning versions. It is CPU-only and uses no pretrained model,
so the MIT/Apache-2.0 and 8-billion-parameter constraints are satisfied.

| Module | Responsibility |
|---|---|
| `config.py` | Every path and shared constant, resolved from the package location so the pipeline runs identically from the repository or the extracted zip |
| `normalize.py` | Transliteration, accent and case folding, abbreviation folding, numeric extraction. Self-testing |
| `metric.py` | Official macro F_0.5, candidate recall, reduction ratio. Verified against the worked example in the problem statement (0.714286). Self-testing |
| `data_io.py` | Streaming TSV to Parquet cache with normalised columns, entity-level validation split, submission writers |
| `blocking.py` | Per-country TF-IDF candidate generation with chunked sparse products |
| `run_baseline.py` | End-to-end validation and prediction, threshold sweep |
| `profile_data.py` | Regenerates every dataset statistic quoted above |
| `make_submission.py` | Assembles this package |

Reproduce both output files:

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/data_io.py --build-cache
python code/business_entity_resolution/src/run_baseline.py --validate
python code/business_entity_resolution/src/run_baseline.py --predict \
    --threshold {{THRESHOLD}} --no-exclusive --block-min {{BLOCK_MIN}}
```

Modules with logic carry self-tests; run the file directly to execute them.

### B. Additional Results

**Dataset profile (measured):**

| File | Rows | Country distribution |
|---|---:|---|
| train_source1 | 2,206,821 | US 60.0%, India 40.0% |
| train_source2 | 5,034,616 | US 59.9%, India 40.1% |
| train_source3 | 5,285,603 | US 60.0%, India 40.0% |
| test_source1 | 1,732,544 | India 46.8%, US 38.3%, France 15.0% |
| test_source2 | 4,887,273 | India 47.3%, US 38.3%, France 14.4% |
| test_source3 | 5,082,316 | mixed |

**Match count distribution (training):**

| Matches | Records | Share |
|---:|---:|---:|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7+ | 87,424 | 3.96% |

**Fair play.** Every transformation in this pipeline derives solely from the
provided training data. No external database, API, geocoding service, or
internet-sourced data augmentation is used anywhere. The Devanagari-to-Latin
mapping and all abbreviation tables are hand-written source code in
`normalize.py`, not retrieved resources.
