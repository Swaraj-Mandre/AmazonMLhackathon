# Business Entity Resolution — Amazon ML Challenge 2026

Team working repository. **Read this file first, then check
[`experiments/experiments.md`](experiments/experiments.md) for the latest scores.**

Whoever picks the work up next: the phase table below says what is done, what is
in progress, and what to start. Claim a phase in the team chat before you begin
so two people don't build the same thing.

---

## 1. The task in one paragraph

Three sources of business records (`business_name`, `business_address`,
`country`) with no shared identifiers. **Source 1 is the deduplicated reference
list.** For every Source 1 entity, output every Source 2 and Source 3 record that
refers to the same real-world business — zero, one, or many.

Scored on **F₀.₅ (precision-weighted 2×), macro-averaged per Source 1 entity**,
singletons included. Predicting an empty list for a true singleton scores a full
1.0; predicting anything for it scores 0.0.

---

## 2. What the data actually looks like

Measured, not assumed — regenerate with
`python code/business_entity_resolution/src/profile_data.py`.

| File | Rows | Countries |
|---|---:|---|
| `train_source1.tsv` | 2,206,821 | US 60.0%, India 40.0% |
| `train_source2.tsv` | 5,034,616 | US 59.9%, India 40.1% |
| `train_source3.tsv` | 5,285,603 | US 60.0%, India 40.0% |
| `train_ground_truth.tsv` | 2,206,821 | — |
| `test_source1.tsv` | 1,732,544 | India 46.8%, US 38.3%, **France 15.0%** |
| `test_source2.tsv` | 4,887,273 | India 47.3%, US 38.3%, France 14.4% |
| `test_source3.tsv` | **NOT DELIVERED** | — |

### Five findings that drive the whole design

1. **Singletons are only 5.58%** of Source 1 entities. An all-empty submission
   scores exactly **0.0558** — that is the floor, not a strategy. But false
   merges on those 123k entities still cost a full point each.
2. **Mean 3.46 matches per entity**, and 91% of entities have between 1 and 6.
   Max observed is 11. This is a multi-match problem, not a 1:1 linkage.
3. **Every Source 2/3 record belongs to at most one Source 1 entity.** Verified
   across all 7,638,365 match links — zero reuse. This is a hard structural
   constraint we can exploit: predictions can be made mutually exclusive on the
   right-hand side, which converts a thresholding problem into an assignment
   problem and should buy precision for free.
4. **Matches never cross country labels.** 1,038,755 sampled links, 0 exceptions.
   Country is therefore a lossless blocking key that cuts the pair space ~2.6×.
   Group by whatever label appears — never hard-code `{US, India}`, since 15% of
   the test set is France and must still be matched.
5. **~10% of Indian match links are Latin → Devanagari.** Source 1 is always
   Latin; Source 2/3 sometimes are not. Character n-grams score exactly zero
   across scripts, so transliteration is mandatory, not a nicety. Handled in
   `normalize.py`.

---

## 3. Blocking issue: `test_source3.tsv` is missing

The download we have contains `test_source1` and `test_source2` but **not
`test_source3`** — the zip is named `...-1-001.zip`, which is how Google Drive
names the *first part* of a split download. The missing file is almost certainly
in a second part.

**Action for the team leader:** go back to the portal download link and fetch the
remaining part(s), then drop `test_source3.tsv` into `data/raw/test/`.

This does **not** block development. Every phase below is built and scored on the
training data. It only blocks generating the final submission, because a Source 1
entity's matches can come from Source 3 and we would silently emit none.

---

## 4. Repository layout

This mirrors the required submission zip, so packaging on day 3 is a zip command
and not a reorganisation.

```
.
├── data/
│   ├── raw/train/            provided TSVs (gitignored — download separately)
│   ├── raw/test/             provided TSVs (gitignored)
│   ├── raw/provided/         problem statement, doc template, official validator
│   └── interim/              cached parquet + blocking artefacts (gitignored)
├── code/business_entity_resolution/
│   ├── src/                  all pipeline code
│   ├── README.md             how to reproduce end-to-end
│   └── requirements.txt      pinned dependencies
├── output/                   matching_results.tsv + candidate_pairs.tsv
├── experiments/              the experiment log — every run goes here
└── docs/                     source emails, problem statement PDFs, notes
```

## 5. Setup

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Put the provided data in `data/raw/train/` and `data/raw/test/`, then confirm:

```bash
python code/business_entity_resolution/src/profile_data.py
```

---

## 6. Phases

Each phase ends with something runnable and a number in the experiment log.
**Do not start a later phase before its dependency is green** — a feature built
on a broken validation split wastes the whole day.

| # | Phase | Status | Owner | Est. | Depends on |
|---|---|---|---|---|---|
| 0 | Setup, profiling, metric, normalisation | ✅ **done** | — | — | — |
| 1 | Data loading + validation split + rules baseline | ⬜ open | | 4–6 h | 0 |
| 2 | Blocking / candidate generation | ⬜ open | | 6–8 h | 1 |
| 3 | Pairwise features | ⬜ open | | 5–7 h | 2 |
| 4 | Classifier + exclusive assignment | ⬜ open | | 6–8 h | 3 |
| 5 | Threshold tuning + generalisation check | ⬜ open | | 3–4 h | 4 |
| 6 | Packaging + methodology document | ⬜ open | | 3 h | 5 |

Roughly 30–36 person-hours. Across three or four people working in parallel that
fits the 72-hour window with real margin — **provided phases 1 and 2 are not
allowed to slip**, because everything downstream is blocked on them.

### Phase 0 — Setup and profiling ✅ done

- `src/config.py` — every path in one place, derived from the file's own
  location so the pipeline runs identically from the repo or the extracted zip.
- `src/metric.py` — official F₀.₅, macro-averaged. **Verified against the
  problem statement's worked example: 0.714286.** Also provides
  `candidate_recall()` and `reduction_ratio()`, the two numbers Amazon uses to
  audit blocking quality.
- `src/normalize.py` — Devanagari→Latin transliteration, accent folding (needed
  for France), legal-form and street-type abbreviation folding, numeric token
  extraction.
- `src/profile_data.py` — regenerates every statistic in section 2.

Run the self-tests: `python src/metric.py` and `python src/normalize.py`.

### Phase 1 — Data loading, validation split, rules baseline ⬜

**Goal: a scored submission on the leaderboard.**

1. `src/data_io.py` — convert each TSV to parquet once in `data/interim/`
   (columnar reads are far faster and the machine only has ~6.6 GB free RAM;
   the raw text is ~2 GB). Provide chunked iteration for anything that cannot
   fit.
2. Hold out **15% of Source 1 entities** (`VALIDATION_FRACTION` in `config.py`).
   **Split by entity, never by pair** — a pair-level split leaks the answer and
   makes every local number a lie.
3. `src/run_baseline.py` — normalise, group by country, TF-IDF character
   n-grams on `core_name`, top-k nearest neighbours, accept above a fixed cosine
   threshold. No ML.
4. Write both output TSVs, run the official validator, hand to the leader.

**Done when:** a local F₀.₅ is recorded in the experiment log and the official
validator exits 0.

### Phase 2 — Blocking / candidate generation ⬜

**This sets the ceiling on recall. A true match not in the candidate set can
never be recovered downstream — no model fixes it.**

- Block within `country` (lossless, verified) and generate candidates per source.
- Combine several recall channels rather than one: character n-gram TF-IDF on
  name, token-level TF-IDF, and an address-number key (records sharing a rare
  long number such as a postcode plus a house number).
- Report `candidate_recall()` and `reduction_ratio()` on the held-out split
  **every time**. Target recall ≥ 0.95 while keeping candidates per entity in the
  low tens.
- Memory: process one country at a time, chunk the sparse matrix multiply, and
  keep only top-k per Source 1 entity.

### Phase 3 — Pairwise features ⬜

For each candidate pair, build a feature row. Start with:

- TF-IDF cosine on name, on `core_name`, and on address
- `rapidfuzz` token-set ratio, partial ratio, Jaro-Winkler on both fields
- Jaccard over name tokens
- **IDF-weighted rare-token overlap** — sharing "Zephay" means vastly more than
  sharing "Restaurant". Usually one of the strongest features in this task.
- Address number overlap: exact match on the longest number (postcode), count of
  shared numbers, whether either side has none
- Length ratios, token counts, a flag for whether the source record was
  transliterated, and whether the address was empty (3.3% of S2/S3 rows)

Do **not** add a raw `country` one-hot — it encodes the training countries and
will behave unpredictably on France. A same-country boolean is fine (and is
always true after Phase 2 blocking, so it carries no information anyway).

### Phase 4 — Classifier and exclusive assignment ⬜

- LightGBM on the pair features. Positives from the ground truth; negatives are
  the **hard** ones — candidates your own blocking produced that are not true
  matches. Random negatives teach the model nothing useful.
- Then exploit finding #3: each Source 2/3 record may belong to **at most one**
  Source 1 entity. Resolve conflicts where two Source 1 entities both claim the
  same record, keeping the higher-scoring claim. Free precision, and precision is
  what F₀.₅ pays for.

### Phase 5 — Threshold tuning and generalisation ⬜

- Sweep the accept threshold against F₀.₅ on the held-out split. Expect the
  optimum well above 0.5 — the metric punishes false merges twice as hard.
  **This sweep is usually worth more than another modelling idea.**
- Treat "predict empty" as a real decision, not a fallback.
- **The France check:** train on US records only, validate on India only. If the
  score collapses, the pipeline has learned country-specific patterns and 15% of
  the test set will fail. Cheap to run, catches the single biggest trap.

### Phase 6 — Packaging and methodology ⬜

- `<team_name>_submission.zip` with `output/`, `code/business_entity_resolution/`,
  and the filled-in `Documentation_template.md`.
- The template lives at `data/raw/provided/Documentation_template.md`. **No page
  limit** — the problem statement asks for technical depth, and the top teams'
  packages are reviewed in detail before final rankings.
- Regenerate both TSVs in the same run so `matching_results` is genuinely a
  subset of `candidate_pairs`. The validator warns when it is not.

---

## 7. Rules that get you disqualified

- **No external data lookup.** No entity-resolution APIs, no business registries,
  **no geocoding APIs for address normalisation**, no internet augmentation. Code
  pipelines are reviewed. Everything in this repo derives from the provided data.
- Final model must be **MIT/Apache-2.0 licensed and ≤ 8B parameters**.
- Max **5 leaderboard uploads per day**, 15 total. Only the registered team
  leader can access the portal.
- Nothing goes up unless it beats our best local F₀.₅ **and** the official
  validator exits 0:

```bash
python data/raw/provided/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

Ties on the leaderboard are broken by **earlier submission time**, so upload a
good result as soon as it exists rather than sitting on it.
