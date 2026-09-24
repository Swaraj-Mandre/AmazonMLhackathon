# Business Entity Resolution — pipeline

Self-contained pipeline for the Amazon ML Challenge 2026 Business Entity
Resolution task. Regenerates both required output files from the provided
training and test data using only what is in this folder.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.11+. No GPU required — the pipeline is CPU-only by design. No
pretrained model is used, so the MIT/Apache-2.0 and 8B-parameter constraints are
satisfied trivially.

## Data layout

Paths are resolved in `src/config.py` relative to this package, so nothing needs
editing. Expected structure at the repository root:

```
data/raw/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
data/raw/test/{test_source1,test_source2,test_source3}.tsv
```

## Reproducing the outputs end to end

```bash
# 1. Verify the environment and print dataset statistics
python src/profile_data.py

# 2. Normalise, block, match, and write both output files
python src/run_baseline.py          # phase 1 — rules only
```

Both files are written to `output/` at the repository root:

- `matching_results.tsv` — final matches (uploaded to the portal)
- `candidate_pairs.tsv` — the candidate set fed to the matching model

## Validating before submission

```bash
python ../../data/raw/provided/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../data/raw/test
```

## Modules

| File | Role |
|---|---|
| `config.py` | Every path and shared constant, resolved from this file's location |
| `metric.py` | Official F₀.₅ (macro per Source 1 entity), candidate recall, reduction ratio |
| `normalize.py` | Devanagari→Latin transliteration, accent folding, abbreviation folding, numeric extraction |
| `profile_data.py` | Dataset statistics: singleton rate, match distribution, country spread |

Each module with logic carries self-tests; run the file directly to execute them.
`metric.py` checks itself against the worked example in the problem statement.

## Design notes

Three measured properties of the data shape the pipeline:

1. **Matches never cross country labels** (0 exceptions in 1,038,755 sampled
   links), so blocking groups by whatever country label is present. The label is
   treated as an open set — the test data contains France, which never appears
   in training.
2. **Each Source 2/3 record belongs to at most one Source 1 entity** (0 reuse
   across all 7,638,365 training links), so final predictions can be made
   mutually exclusive on the right-hand side to gain precision.
3. **About 10% of Indian match links are Latin→Devanagari.** Source 1 is always
   Latin script. Character n-gram similarity is exactly zero across scripts, so
   `normalize.py` transliterates before any comparison happens.

All transformations derive solely from the provided training data. No external
databases, APIs, geocoding services, or internet data augmentation are used
anywhere in this pipeline.
