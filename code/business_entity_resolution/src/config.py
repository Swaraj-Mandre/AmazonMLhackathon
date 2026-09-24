"""Every path the pipeline uses, resolved once.

Nothing else in the codebase hard-codes a path. Paths are derived from this
file's own location, so the pipeline runs the same from the repository or from
the extracted submission zip, on any machine, without editing anything.

Layout::

    <repo root>/
    ├── data/raw/{train,test}/*.tsv      provided data, never modified
    ├── data/interim/                    cached artefacts we regenerate
    ├── code/business_entity_resolution/src/
    └── output/                          the two TSVs we submit
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SRC_DIR.parent                  # code/business_entity_resolution
REPO_ROOT = PACKAGE_DIR.parent.parent         # repository root

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
PROVIDED_DIR = RAW_DIR / "provided"
OUTPUT_DIR = REPO_ROOT / "output"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TRAIN_SOURCES = {
    "S1": TRAIN_DIR / "train_source1.tsv",
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}
TEST_SOURCES = {
    "S1": TEST_DIR / "test_source1.tsv",
    "S2": TEST_DIR / "test_source2.tsv",
    "S3": TEST_DIR / "test_source3.tsv",  # not yet delivered; see README
}

SOURCE_FILES = {
    **{f"train {k}": v for k, v in TRAIN_SOURCES.items()},
    **{f"test {k}": v for k, v in TEST_SOURCES.items()},
}

MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"

# Column names are fixed by the problem statement; the validator rejects anything else.
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
RESULT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

# Held-out fraction of Source 1 entities used for local scoring. Splitting by
# entity (never by pair) is what keeps the local number comparable to the
# leaderboard, so this constant is shared rather than chosen per script.
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 20260925

for _d in (INTERIM_DIR, OUTPUT_DIR, EXPERIMENTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
