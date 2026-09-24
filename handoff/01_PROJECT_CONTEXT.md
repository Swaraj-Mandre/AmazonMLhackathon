# Project context, paste this into a new AI chat first

You are helping a 4-person student team in the Amazon ML Challenge 2026. Read
this whole brief before answering anything. It is the accurate current state of
the project. Do not contradict the measured numbers below, they came from running
code over the full dataset.

---

## The task

Business Entity Resolution. Three sources of business records, no shared IDs.

Every record has four columns: `entity_id`, `business_name`, `business_address`,
`country`. The `entity_id` prefix says which source it came from: `S1-`, `S2-`,
`S3-`.

**Source 1 is a deduplicated master list.** For every Source 1 record, output
every Source 2 and Source 3 record that refers to the same real business. A
record may have zero, one, or many matches.

Output is a tab separated file, one row per Source 1 record:

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00003	
```

A second file, `candidate_pairs.tsv`, has the same shape and holds whatever our
blocking stage passed to the matcher. Amazon audits it.

## Scoring

**F0.5, computed per Source 1 record then averaged (macro).** Precision is
weighted twice as heavily as recall.

```
F_0.5 = (1.25 * P * R) / (0.25 * P + R)
```

Per-record outcomes, which drive every design decision:

| True matches | We predict | Score for that record |
|---|---|---|
| 1 | exactly it | 1.00 |
| 1 | it plus one wrong | 0.56 |
| 1 | nothing | 0.00 |
| none | nothing | 1.00 |
| none | anything | 0.00 |

So a wrong match costs more than a missed one. When unsure, predict nothing.

## Dataset size

| File | Rows |
|---|---:|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

About 2 GB of text. Training covers US and India only. **The test set also
contains France, which never appears in training,** and every test record
including the French ones must appear in the submission.

## What we measured (do not contradict these)

1. **Singletons are rare.** 5.58% of Source 1 records have no match. Predicting
   nothing for everyone scores 0.0561. Mean matches per record is 3.46, max 11.
2. **Every Source 2 or 3 record has at most one owner.** Checked across all
   7,638,365 links, zero exceptions. So predictions can be made mutually
   exclusive on the candidate side.
3. **Matches never cross country labels.** 1,038,755 sampled links, zero
   exceptions. Country is a free, lossless filter. Treat it as an open set of
   strings, never hard-code US and India, because France must flow through the
   same code path.
4. **About 10% of Indian matches are Latin to Devanagari.** Source 1 is always
   Latin script, Source 2 and 3 sometimes are not. Word and character overlap
   score exactly zero across two scripts, so we transliterate first.
5. Names are never empty. Addresses are empty in about 3.3% of Source 2 and 3
   rows. Raw names contain junk prefixes like `--` and `<<`.

## Where we are now

Phases 0 and 1 are finished.

**Current result: local F0.5 = 0.5659, candidate recall = 0.7123.** This is a
rules-only baseline, no machine learning: cleaned text, TF-IDF cosine on the
business name inside each country, accept above a threshold. Best settings are
accept threshold 0.92 with exclusivity off.

**The biggest problem right now is candidate recall of 0.7123.** That means 29%
of the true matches are never even looked at by the scoring stage, and no model
downstream can recover them. Raising it is worth more than any other single
change.

## How the pipeline works

1. **Normalise** (`normalize.py`): transliterate Devanagari to Latin, strip junk
   prefixes, remove accents (needed for French), fold case, expand `&` to `and`,
   fold legal forms so `Private Limited` equals `Pvt Ltd`, fold street types so
   `Road` equals `Rd`, and pull digit runs out of addresses. `core_name()` also
   drops generic tokens like `pvt` and `ltd`, and that stripped form is the
   blocking key.
2. **Cache** (`data_io.py`): each TSV becomes a Parquet file with the cleaned
   columns already attached, so the expensive normalisation is paid once.
3. **Block** (`blocking.py`): within one country at a time, TF-IDF over
   whitespace tokens of `core_name`, fitted per country block so the IDF reflects
   that country's vocabulary. Each Source 2/3 record proposes its top 5 Source 1
   records above a cosine floor.
4. **Match** (`run_baseline.py`): currently just a threshold on that cosine. A
   trained classifier is the planned replacement.

## How we validate, and why it looks strange

**Do not "simplify" this. It is deliberate.**

Blocking scans candidate to record: each Source 2/3 record proposes the few
Source 1 records it most resembles. How selective that is depends on how many
records it is choosing between. At test time that is 1,732,544.

Our first harness held out 20,000 records but still showed every candidate all
10 million rows. Every candidate then dumped its nearest guess onto that tiny
query set. It reported 961 candidates per record and a completely meaningless
score.

The harness now rebuilds the test set's own proportions: for N held-out records
it takes their true matches plus enough random distractors to reach 5.75
candidates per record, which is the real test ratio.

It is still slightly optimistic and we know it. A 40,000 record sample scored
0.6966 with 0.7971 recall; the full 329,000 split scored 0.5659 with 0.7123. The
score falls as the record count rises, so the leaderboard should land below
0.5659.

Other validation rules:
- Split by **Source 1 record, never by pair.** A pair-level split puts other
  matches of the same record on both sides and leaks the answer.
- The split is seeded so everyone scores the same held-out records.
- Never change the split. If it changes, every earlier number is void.

## Memory, this dataset will freeze a laptop

We develop on 16 GB machines. Two runs died. Rules:

1. Read Parquet with the Arrow dtype backend (`load_source()` does it). Without
   it the strings become millions of Python objects.
2. Never load a whole source then filter. Use `load_source_country(path,
   country)`, which pushes the filter into Parquet.
3. Never build the full ground truth if you need a slice. Use
   `load_ground_truth(keep=ids)`.
4. **Never hold millions of pairs as Python strings.** The test run makes about
   36 million pairs. As strings that is over 8 GB. Entity IDs are a prefix plus
   digits with no leading zeros, so `encode_ids()` turns them into int64 plus a
   source byte, about 21 bytes per pair. Write output with
   `write_id_lists_coded()`, never by building a `{record: set(ids)}` dict.
5. `CANDIDATE_CHUNK` in `blocking.py` is the main memory dial. At 20,000 one
   block reached 5.8 GB; at 6,000 the same block runs at 2.2 GB with identical
   output.

Also `del` big frames and `gc.collect()` between blocks.

## Rules that disqualify us

- **No external data lookup of any kind.** No entity resolution APIs, no business
  registries, **no geocoding APIs for addresses**, no internet data. Amazon
  reviews the code. Everything must derive from the provided training data.
  Hand-written mapping tables in our own source are fine, retrieved data is not.
- Final model must be **MIT or Apache 2.0 licensed and 8 billion parameters or
  fewer**.
- Five leaderboard uploads per day, fifteen total.

## How to help

- Give code that fits the existing modules and naming. Ask for
  `05_CODE_MAP.md` if you need exact signatures.
- Prefer approaches that raise candidate recall.
- Say when you are unsure rather than guessing a number.
- Every suggestion must be checkable by running
  `run_baseline.py --validate` and comparing F0.5 against 0.5659.
