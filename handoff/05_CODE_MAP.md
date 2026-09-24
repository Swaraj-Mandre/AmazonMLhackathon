# Code map: what exists and how to call it

Paste this when you need exact function names instead of guesses.

Everything lives in `code/business_entity_resolution/src/`. Modules with logic
have self-tests, so running the file directly checks it still works.

## config.py

Every path, resolved from the file location, so nothing breaks when the folder
moves.

```python
REPO_ROOT, DATA_DIR, RAW_DIR, INTERIM_DIR, TRAIN_DIR, TEST_DIR, PROVIDED_DIR
OUTPUT_DIR, EXPERIMENTS_DIR
GROUND_TRUTH                  # train_ground_truth.tsv
TRAIN_SOURCES = {"S1": Path, "S2": Path, "S3": Path}
TEST_SOURCES  = {"S1": Path, "S2": Path, "S3": Path}
MATCHING_RESULTS, CANDIDATE_PAIRS
RESULT_COLUMNS, CANDIDATE_COLUMNS, SOURCE_COLUMNS
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 20260925
```

## normalize.py

```python
transliterate(text) -> str          # Devanagari to Latin, other scripts untouched
has_devanagari(text) -> bool
normalize_name(raw) -> str          # full cleaned name, legal forms folded
core_name(raw) -> str               # same, with pvt/ltd/inc/the stripped. BLOCKING KEY
normalize_address(raw) -> str       # rd/road, st/street, directionals folded
address_numbers(raw) -> list[str]   # digit runs, longest first (postcode leads)
name_tokens(raw) -> frozenset[str]
```

## metric.py

The only place F0.5 is computed. Do not reimplement it anywhere else.

```python
entity_f_beta(truth: set, pred: set, beta=0.5) -> float
macro_f_beta(truth: dict[str, set], pred: dict[str, set], beta=0.5) -> float
candidate_recall(truth: dict[str, set], candidates: dict[str, set]) -> float
reduction_ratio(candidates: dict, n_source2: int, n_source3: int) -> float
parse_id_list(cell) -> set[str]
load_id_map(path, key_col, value_col) -> dict[str, set[str]]
```

Verified against the worked example in the problem statement: 0.714286.

## data_io.py

```python
build_cache(tsv_path, force=False) -> Path        # TSV to normalised Parquet
build_all_caches(force=False)
load_source(tsv_path, columns=None) -> DataFrame  # Arrow backed, memory safe
load_source_country(tsv_path, country, columns=None) -> DataFrame   # USE THIS
list_countries(tsv_path) -> list[str]             # discovered, never hard-coded
iter_source_chunks(tsv_path, columns=None, batch_rows=500_000)
load_ground_truth(keep: set | None = None) -> dict[str, set[str]]
split_entities(entity_ids, fraction=0.15, seed=RANDOM_SEED) -> (train, val)
encode_ids(ids) -> (num: int64 array, src: uint8 array)
write_id_lists_coded(path, columns, left_num, right_num, right_src, order_num) -> int
write_id_lists_from_pairs(path, columns, pairs, id_col, order) -> int
write_matching_results(path, matches: dict, order: list)
write_candidate_pairs(path, candidates: dict, order: list)
```

Cached columns added to every row: `core_name`, `norm_addr`, `addr_nums`,
`was_transliterated`, alongside the four original columns.

## blocking.py

```python
CANDIDATE_CHUNK = 6_000        # main memory dial, lower it if memory is tight
fit_vectorizer(texts, sample=1_000_000, seed=0) -> TfidfVectorizer
generate_candidates(queries, candidates, top_k=8, min_score=0.30,
                    vectorizer=None, verbose=True) -> DataFrame
    # columns: source1_entity_id, candidate_entity_id, cos_name
to_id_lists(pairs, id_col="candidate_entity_id") -> dict[str, set[str]]
```

`queries` and `candidates` are DataFrames needing `entity_id` and `core_name`.

## run_baseline.py

```python
BLOCK_MIN_SCORE = 0.30
TOP_K_PER_CANDIDATE = 5
ACCEPT_THRESHOLDS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.95, 0.97, 0.99]
CANDIDATES_PER_ENTITY = 5.75           # the real test ratio

build_pairs(queries, tags, load_block, verbose=True, min_score=...)
iter_block_pairs(queries, tags, load_block, ...)   # yields one block at a time
apply_exclusivity(pairs) -> DataFrame              # one owner per candidate
build_validation_pool(truth, seed) -> dict[str, DataFrame]
evaluate(pairs, truth, entity_order, pool_size)
run_validate(sample, min_score)
run_predict(threshold, exclusive, min_score)
```

Command line:

```bash
python src/run_baseline.py --validate                 # full held-out split
python src/run_baseline.py --validate --sample 40000  # faster, optimistic
python src/run_baseline.py --predict --threshold 0.92 --no-exclusive --block-min 0.60
```

## make_submission.py

```bash
python src/make_submission.py --team "sahara" --members "..."
```

Runs the official validator, refuses to write if it fails, then puts
`matching_results.tsv` and the zip into `submission/`.

## profile_data.py

Regenerates every dataset statistic quoted in the README and context file.
