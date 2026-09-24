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
   scores 0.0558. That is our floor, not a plan.
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

- `data_io.py` turns each TSV into a parquet cache with the cleaned columns
  already added, streaming in chunks so it never blows memory.
- Validation split is **15% of Source 1 records, split by record, never by pair.**
  A pair split leaks the answer and makes every local score a lie. Seeded, so we
  all score on exactly the same held out set.
- `blocking.py` shortlists candidates using TF-IDF cosine on the name, inside one
  country at a time.
- `run_baseline.py` runs the whole thing end to end, no machine learning.

```bash
python code/business_entity_resolution/src/run_baseline.py --validate
python code/business_entity_resolution/src/run_baseline.py --predict --threshold 0.75
```

### Phase 2, blocking, open

**This is the most important phase left.** Whatever blocking misses is gone for
good, no model downstream can get it back.

What to do:
- Add more ways to find candidates, not just one. Character n-grams on the name,
  word level TF-IDF, and an address number key (two records sharing a rare long
  number like a PIN code plus a house number).
- Print `candidate_recall()` and `reduction_ratio()` **every single time.** Target
  recall 0.95 or better while keeping candidates per record in the low tens.
- Keep memory in check: one country at a time, chunk the sparse multiply, keep
  only top k per record.

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
