# Amazon ML Challenge 2026: Business Entity Resolution

Team repo. **Read this file top to bottom before you write any code.**

Quick links:
- Approach explained in plain words: [`docs/Approach_Pitch.docx`](docs/Approach_Pitch.docx)
- Score log, every run we do: [`experiments/experiments.md`](experiments/experiments.md)
- Official problem statement: [`data/raw/provided/PROBLEM_STATEMENT.md`](data/raw/provided/PROBLEM_STATEMENT.md)

---

## 1. What we have to build

Three files of business records. Same shop can appear in all three, spelled
differently, with no ID linking them.

- **Source 1** is the clean master list.
- For every Source 1 record, output every **Source 2 and Source 3** record that is
  the same real business.
- A record can have **zero, one, or many** matches. Average is 3.46.

## 2. How we are scored

Metric is **F0.5**, worked out for each Source 1 record separately, then averaged.

**Precision counts twice as much as recall.** A wrong match hurts more than a
missed one.

| Situation | What we predict | Score for that record |
|---|---|---|
| 1 real match | exactly it | 1.00 |
| 1 real match | it plus one wrong | 0.56 |
| 1 real match | nothing | 0.00 |
| No real match | nothing | 1.00 |
| No real match | anything | 0.00 |

**Rule of thumb: when the model is not sure, predict nothing.**

## 3. What we found in the data

All measured by us. Rerun with
`python code/business_entity_resolution/src/profile_data.py`.

| File | Rows | Countries |
|---|---:|---|
| train_source1 | 2,206,821 | US 60%, India 40% |
| train_source2 | 5,034,616 | US 60%, India 40% |
| train_source3 | 5,285,603 | US 60%, India 40% |
| train_ground_truth | 2,206,821 | n/a |
| test_source1 | 1,732,544 | India 47%, US 38%, **France 15%** |
| test_source2 | 4,887,273 | India 47%, US 38%, France 14% |
| test_source3 | 5,082,316 | n/a |

Four findings that shape everything:

1. **Only 5.58% of records have no match.** Submitting nothing for everyone
   scores 0.0561. That is our floor, not a plan.
2. **Every Source 2 or 3 record has at most one owner.** We checked all 7,638,365
   links. Zero exceptions. So we can force our answers to be exclusive, which
   costs a bit of recall and buys precision. The metric pays for precision.
3. **Matches never cross countries.** 1,038,755 links checked, zero exceptions.
   Country is a free filter that cuts the work by about 2.6 times.
4. **About 10% of Indian matches are Latin to Devanagari.** Source 1 is always
   Latin. Word overlap scores zero across two scripts, so those matches are
   invisible unless we convert first. Our converter is in `normalize.py`.

## 4. Folder layout

Matches the submission zip Amazon wants, so packaging on day 3 is just a zip.

```
data/raw/train/        the 4 training files (not in git, too big)
data/raw/test/         the 3 test files (not in git)
data/raw/provided/     problem statement, doc template, official validator
data/interim/          parquet cache we build (not in git)
code/business_entity_resolution/src/    all our code
output/                the 2 files we submit
experiments/           score log
docs/                  the approach doc, source emails, PDFs
```

## 5. Setup, do this first

```bash
git clone https://github.com/Swaraj-Mandre/AmazonMLhackathon.git
cd AmazonMLhackathon
pip install -r code/business_entity_resolution/requirements.txt
```

The data is **not** in git, it is 2 GB. Get the files from the leader's download
link and put them in `data/raw/train/` and `data/raw/test/`. Then check it worked:

```bash
python code/business_entity_resolution/src/profile_data.py
```

Build the parquet cache once. Takes about 25 minutes, then everything after is
fast:

```bash
python code/business_entity_resolution/src/data_io.py --build-cache
```

## 6. Phases, claim one in the group chat before you start

| # | Phase | Status | Who | Time | Needs |
|---|---|---|---|---|---|
| 0 | Setup, profiling, metric, text cleaning | done | | | |
| 1 | Data loading, validation split, rules baseline | done | | | 0 |
| 2 | Better blocking | open | | 6 to 8 h | 1 |
| 3 | Pair features | open | | 5 to 7 h | 2 |
| 4 | Classifier plus exclusivity | open | | 6 to 8 h | 3 |
| 5 | Threshold tuning, France check | open | | 3 to 4 h | 4 |
| 6 | Packaging, methodology doc | open | | 3 h | 5 |

About 25 to 30 hours of work left. Split across 3 or 4 people that fits, **as
long as phase 2 does not slip**, because 3, 4 and 5 all wait on it.

---

### Phase 0, done

- `config.py` holds every file path. Nothing else hardcodes a path.
- `metric.py` is the official F0.5. **Checked against the example in the problem
  statement, gives 0.714286.** Also has `candidate_recall()` and
  `reduction_ratio()`, the two numbers Amazon uses to judge our blocking.
- `normalize.py` cleans text: Devanagari to Latin, accent stripping for French,
  abbreviation folding (Pvt Ltd equals Private Limited), number extraction.
- `profile_data.py` regenerates every number in section 3.

Run `python src/metric.py` and `python src/normalize.py` to check they pass.

### Phase 1, done

**Result: local F0.5 = 0.5659, candidate recall = 0.7123.** No machine learning
at all, just TF-IDF cosine on the cleaned name with a threshold. For comparison,
predicting nothing for everyone scores 0.0561.

- `data_io.py` turns each TSV into a parquet cache with the cleaned columns
  already added, streaming in chunks so it never blows memory.
- Validation split is **15% of Source 1 records, split by record, never by pair.**
  A pair split leaks the answer and makes every local score a lie. Seeded, so we
  all score on exactly the same held out set.
- `blocking.py` shortlists candidates using TF-IDF cosine on the name, inside one
  country at a time.
- `run_baseline.py` runs the whole thing end to end and sweeps the threshold.

```bash
python code/business_entity_resolution/src/run_baseline.py --validate
python code/business_entity_resolution/src/run_baseline.py --predict --threshold 0.92 --no-exclusive --block-min 0.60
```

Best settings found: accept threshold **0.92**, exclusivity **off**. The full
sweep is in the experiment log.

**Read the Method notes in [`experiments/experiments.md`](experiments/experiments.md)
before you change the validation harness.** It builds its own candidate pool on
purpose, sized to the test set's real ratio of 5.75 candidates per record. An
earlier version did not, and produced a completely meaningless score. That
section explains why.

### Phase 2, blocking, open

**This is the most important phase left, by a long way.** Our candidate recall is
**0.7123**, which means **29% of the real matches are never even looked at.** No
model downstream can recover them. Every point of recall we add here raises the
ceiling on everything in phases 3, 4 and 5.

The single most promising fix, in order:

1. **Scan in both directions and take the union.** Right now every Source 2 or 3
   record proposes its top 5 Source 1 records. That direction alone loses recall
   as the record count grows, and we measured it: a 40k sample gave 0.7971 recall,
   the full 329k split gave 0.7123. At the real 1.73 million it will be worse. Add
   the opposite scan, each Source 1 record proposing its top k candidates, and
   merge the two candidate sets.
2. **Add more ways to find candidates.** Character n-grams on the name, word level
   TF-IDF, and an address number key (two records sharing a rare long number like
   a PIN code plus a house number).
3. **Print `candidate_recall()` and `reduction_ratio()` every single time.** Target
   recall 0.95 or better while keeping candidates per record in the low tens.

Keep memory in check: one country at a time, chunk the sparse multiply, keep only
top k per record. See section 9 before you write loading code.

### Phase 3, pair features, open

For every shortlisted pair, build a row of numbers:
- TF-IDF cosine on name, on core name, on address
- `rapidfuzz` token set ratio, partial ratio, Jaro-Winkler on both fields
- Jaccard over name words
- **Rare word overlap, weighted by IDF.** Sharing "Zephay" means far more than
  sharing "Restaurant". Usually the strongest feature in this kind of task.
- Address numbers: does the longest number match (that is usually the PIN code),
  how many numbers are shared, does either side have none
- Length ratios, word counts, was the record transliterated, was the address empty
  (3.3% of Source 2 and 3 rows have no address)

**Do not one-hot the country.** It encodes US and India and will do something
unpredictable on France.

### Phase 4, classifier, open

- LightGBM on those features.
- Positives come from the ground truth.
- **Negatives must be the near misses our own blocking produced**, not random
  pairs. Random pairs are too easy and teach the model nothing.
- Then apply finding 2: if two Source 1 records claim the same partner, keep only
  the stronger claim.

### Phase 5, thresholds, open

- Sweep the accept threshold against F0.5 on the held out set. Expect the best one
  to sit high.
- **This sweep usually gains more score than another model idea.** Do not skip it.
- **France check:** train on US records only, test on India only. If the score
  falls apart, our pipeline learned country specific habits and 15% of the test
  set will fail. Cheap to run, catches the biggest trap.

### Phase 6, packaging, open

- Zip: `output/`, `code/business_entity_resolution/`, and the filled in
  `Documentation_template.md` (it is in `data/raw/provided/`).
- **No page limit** on the methodology doc. Amazon reads the top teams' packages
  in detail.
- Regenerate both TSVs in the same run so the matches really are a subset of the
  candidates.

---

## 7. Rules, read these once

Things that get us disqualified:
- **No outside data.** No entity resolution APIs, no business registries, **no
  geocoding APIs for addresses**, no data pulled from the internet. They review
  the code.
- Final model must be **MIT or Apache 2.0 licensed and 8B parameters or less**.
- **5 uploads per day, 15 total.** Only the team leader can access the portal.

Before anything is uploaded:
1. It must beat our best local F0.5.
2. The official validator must pass:

```bash
python data/raw/provided/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

Ties on the leaderboard go to whoever submitted earlier, so upload a good result
as soon as we have one.

## 8. Working rules

- **Log every run** in `experiments/experiments.md`, including failures. If it is
  not logged, it did not happen and we cannot write it up later.
- **Nobody changes the validation split.** If it changes, every earlier number
  becomes meaningless.
- **Only the leader uploads.** Two people uploading burns the daily cap by
  accident.
- **Say you are stuck after 30 minutes**, not after 3 hours.
- Settle disagreements by running both and comparing F0.5, not by arguing.

## 9. Memory, read this before writing loading code

This dataset punishes careless loading. The machine we developed on has 16 GB with
about 6.5 GB free, and the first validation run reached **8.7 GB and went to swap.**
Three rules came out of fixing it:

1. **Read parquet with the Arrow dtype backend.** `load_source()` already does.
   Without it the string columns become millions of individual Python objects and
   a 5 million row source costs several GB instead of a few hundred MB.
2. **Never load a whole source and then filter it.** Use
   `load_source_country(path, country)`, which pushes the filter down into parquet
   so the rows you are going to throw away are never built at all.
3. **Never build the full ground truth if you only need a slice.** Use
   `load_ground_truth(keep=some_ids)`. All 2.2 million records with their 7.6
   million IDs as Python sets costs a couple of GB on its own.

Also: free big frames with `del` and call `gc.collect()` between country blocks.
`run_baseline.py` shows the pattern.

Two more that cost us a full run each on the test set:

4. **Never hold millions of pairs as Python strings.** The test run produces
   about 36 million candidate pairs. As strings that is over 8 GB and it killed
   the machine. Entity IDs are a `S1-`/`S2-`/`S3-` prefix plus digits with no
   leading zeros (checked on all 11,702,133 test IDs), so `encode_ids()` turns
   them into int64 plus a source byte. Measured cost: **21 bytes per pair.**
   Write output with `write_id_lists_coded()`, which sorts the coded arrays and
   emits rows without ever building a `{record: set(ids)}` dictionary.
5. **`CANDIDATE_CHUNK` in `blocking.py` is the main memory dial.** Each chunk
   makes a sparse product of (chunk x vocabulary) against (vocabulary x records),
   and that intermediate scales linearly with the chunk size. At 20,000 the first
   block alone reached 5.8 GB; at 6,000 the same block runs at 2.2 GB with the
   same output and the same runtime. If you hit memory trouble, lower this first.

## 10. Help files for working with an AI assistant

We each use different assistants, and every new chat starts knowing nothing about
this project. Explaining it from scratch each time wastes effort and the answers
come back wrong, because the assistant guesses at things we already measured.

So we have put a set of briefing files in [`handoff/`](handoff/). We can paste
these into our individual chatbots to continue from wherever the last person
stopped.

**How to use them:**

1. Open a new chat with whatever assistant you use.
2. Paste [`handoff/01_PROJECT_CONTEXT.md`](handoff/01_PROJECT_CONTEXT.md) first.
   That is the shared background: the problem, the scoring, everything we
   measured, what is already built, and the traps.
3. Paste the file for the phase you claimed, for example
   [`handoff/02_PHASE2_BLOCKING.md`](handoff/02_PHASE2_BLOCKING.md).
4. Ask your question normally.

| File | Use it for |
|---|---|
| [`00_START_HERE.md`](handoff/00_START_HERE.md) | How this works, read once |
| [`01_PROJECT_CONTEXT.md`](handoff/01_PROJECT_CONTEXT.md) | Always, at the start of every chat |
| [`02_PHASE2_BLOCKING.md`](handoff/02_PHASE2_BLOCKING.md) | Candidate generation |
| [`03_PHASE3_FEATURES.md`](handoff/03_PHASE3_FEATURES.md) | Pair features |
| [`04_PHASE4_MODEL.md`](handoff/04_PHASE4_MODEL.md) | Training the classifier |
| [`05_CODE_MAP.md`](handoff/05_CODE_MAP.md) | Exact function names and signatures |
| [`06_RULES_AND_TRAPS.md`](handoff/06_RULES_AND_TRAPS.md) | Rules, deadlines, output format |

**Two things to be careful about:**

- **Never paste the dataset or any rows from it** into a chatbot. The rules
  forbid taking our data outside the challenge, and the files are gigabytes
  anyway. The context file describes the columns, which is all an assistant
  needs.
- **Check whatever it writes.** Assistants sound confident even when they are
  wrong. Run `run_baseline.py --validate` and compare the F0.5 against
  [`experiments/experiments.md`](experiments/experiments.md). If a change does
  not move that number, it did not help.
