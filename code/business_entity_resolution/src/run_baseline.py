"""Phase 1, rules-only baseline, end to end.

No machine learning: normalise, block by country, score candidate pairs by
TF-IDF cosine over business names, accept everything above a threshold. The
point is not the score but the scaffolding, it proves the output format, gives
the leaderboard a real number to calibrate against, and establishes the local
validation harness every later phase is measured on.

Two modes::

    python src/run_baseline.py --validate            # score on held-out entities
    python src/run_baseline.py --predict             # write the submission files

Why validation builds its own candidate pool
--------------------------------------------
Blocking scans candidate to entity: each Source 2/3 record proposes the few
Source 1 entities it looks most like. How selective that is depends entirely on
how many entities the record has to choose between. At test time a record picks
its best few out of 1,732,544 entities. If validation holds out only 20,000
entities but still shows every record all 10 million rows, each record dumps its
nearest guess onto that small set and precision collapses for reasons that have
nothing to do with the model. The first version of this script did exactly that
and reported around 960 candidates per entity.

So validation reconstructs the test set's own proportions instead: for N
held-out entities it uses their true matches plus enough randomly drawn
distractors to reach the same candidates-per-entity ratio the test data has.
That keeps the difficulty honest and, as a side effect, makes validation fast
enough to run on the full held-out split rather than a sample.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import fit_vectorizer, generate_candidates, to_id_lists  # noqa: E402
from config import (  # noqa: E402
    CANDIDATE_COLUMNS,
    CANDIDATE_PAIRS,
    MATCHING_RESULTS,
    RANDOM_SEED,
    RESULT_COLUMNS,
    TEST_SOURCES,
    TRAIN_SOURCES,
)
from data_io import (  # noqa: E402
    encode_ids,
    write_id_lists_coded,
    load_ground_truth,
    write_id_lists_from_pairs,
    load_source,
    load_source_country,
    split_entities,
    write_candidate_pairs,
    write_matching_results,
)
from metric import candidate_recall, macro_f_beta, reduction_ratio  # noqa: E402

# Blocking keeps anything this similar or better; the accept threshold applied
# afterwards is always higher. Keeping the two separate is what lets us sweep
# the accept threshold without regenerating candidates.
BLOCK_MIN_SCORE = 0.30
TOP_K_PER_CANDIDATE = 5
ACCEPT_THRESHOLDS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.95, 0.97, 0.99]

COLUMNS = ["entity_id", "core_name", "country"]

# Measured from the test files: 1,732,544 Source 1 entities against 4,887,273 +
# 5,082,316 Source 2 and Source 3 records. Validation reproduces this ratio so
# that a record on the validation side faces the same competition it will face
# at prediction time.
TEST_S1_ROWS = 1_732_544
TEST_CANDIDATE_ROWS = 4_887_273 + 5_082_316
CANDIDATES_PER_ENTITY = TEST_CANDIDATE_ROWS / TEST_S1_ROWS  # about 5.75

BlockLoader = Callable[[str, str], pd.DataFrame]


def build_pairs(queries: pd.DataFrame, tags: list[str], load_block: BlockLoader,
                verbose: bool = True, min_score: float = BLOCK_MIN_SCORE) -> pd.DataFrame:
    """Generate candidate pairs for every (source, country) block.

    ``load_block(tag, country)`` supplies one block of candidate records, which
    lets prediction read straight from Parquet while validation serves frames it
    has already sub-sampled. Countries come from whatever labels are present, so
    the unseen France block flows through without a special case.
    """
    frames = []
    for tag in tags:
        for country in sorted(queries["country"].unique()):
            q = queries[queries["country"] == country]
            if q.empty:
                continue
            c = load_block(tag, country)
            if c is None or c.empty:
                continue
            if verbose:
                print(f"  {tag} / {country}: {len(q):,} entities x {len(c):,} records",
                      flush=True)
            t0 = time.time()
            # Fitted per block so IDF reflects that country's own vocabulary:
            # "nagar" is common in India and rare in the US, and the weighting
            # should say so.
            vec = fit_vectorizer(pd.concat([q["core_name"], c["core_name"]],
                                           ignore_index=True))
            pairs = generate_candidates(
                q, c,
                top_k=TOP_K_PER_CANDIDATE,
                min_score=min_score,
                vectorizer=vec,
                verbose=False,
            )
            if verbose:
                print(f"    -> {len(pairs):,} pairs in {time.time()-t0:.0f}s", flush=True)
            frames.append(pairs)
            del c, vec
            gc.collect()

    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "cos_name"])
    return pd.concat(frames, ignore_index=True)


def iter_block_pairs(queries: pd.DataFrame, tags: list[str], load_block: BlockLoader,
                     verbose: bool = True, min_score: float = BLOCK_MIN_SCORE):
    """Same work as :func:`build_pairs`, yielding each block instead of concatenating.

    Prediction consumes blocks one at a time and compacts each to integer codes,
    so the full string-valued pair set never exists at once.
    """
    for tag in tags:
        for country in sorted(queries["country"].unique()):
            q = queries[queries["country"] == country]
            if q.empty:
                continue
            c = load_block(tag, country)
            if c is None or c.empty:
                continue
            if verbose:
                print(f"  {tag} / {country}: {len(q):,} entities x {len(c):,} records",
                      flush=True)
            t0 = time.time()
            vec = fit_vectorizer(pd.concat([q["core_name"], c["core_name"]],
                                           ignore_index=True))
            pairs = generate_candidates(q, c, top_k=TOP_K_PER_CANDIDATE,
                                        min_score=min_score, vectorizer=vec,
                                        verbose=False)
            if verbose:
                print(f"    -> {len(pairs):,} pairs in {time.time()-t0:.0f}s", flush=True)
            del c, vec
            gc.collect()
            yield pairs


def apply_exclusivity(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep only each candidate record's single best Source 1 entity.

    Justified by a measured property of the training data: across all 7,638,365
    match links, no Source 2 or Source 3 record is ever claimed by two different
    Source 1 entities. Enforcing it drops contested pairs, costing a little
    recall and buying precision, and F_0.5 weights precision twice as heavily.
    """
    if pairs.empty:
        return pairs
    order = pairs["cos_name"].to_numpy().argsort()[::-1]
    return pairs.iloc[order].drop_duplicates("candidate_entity_id", keep="first")


def build_validation_pool(truth: dict[str, set[str]], seed: int = RANDOM_SEED
                          ) -> dict[str, pd.DataFrame]:
    """Candidate pool for validation, sized to the test set's own ratio.

    Contains every true match of the held-out entities, plus randomly drawn
    records that belong to nobody in the held-out set, until the pool reaches
    ``CANDIDATES_PER_ENTITY`` records per held-out entity. Without the second
    part the task would be far too easy; without the first it would be
    unscoreable.
    """
    rng = np.random.default_rng(seed)
    true_ids = set().union(*truth.values()) if truth else set()
    target_total = int(len(truth) * CANDIDATES_PER_ENTITY)
    n_distractors = max(0, target_total - len(true_ids))

    print(f"  building validation pool: {len(truth):,} entities, "
          f"{len(true_ids):,} true matches, "
          f"{n_distractors:,} distractors "
          f"(target {CANDIDATES_PER_ENTITY:.2f} records per entity)", flush=True)

    pool: dict[str, pd.DataFrame] = {}
    sizes = {tag: 0 for tag in ("S2", "S3")}
    for tag in ("S2", "S3"):
        df = load_source(TRAIN_SOURCES[tag], columns=COLUMNS)
        sizes[tag] = len(df)
        pool[tag] = df

    total_rows = sum(sizes.values())
    for tag in ("S2", "S3"):
        df = pool[tag]
        is_true = df["entity_id"].isin(true_ids).to_numpy()
        share = int(n_distractors * sizes[tag] / total_rows)
        others = np.flatnonzero(~is_true)
        take = rng.choice(others, min(share, len(others)), replace=False)
        keep = np.concatenate([np.flatnonzero(is_true), take])
        pool[tag] = df.iloc[np.sort(keep)].reset_index(drop=True)
        print(f"    {tag}: {len(pool[tag]):,} records "
              f"({int(is_true.sum()):,} true + {len(take):,} distractors)", flush=True)
        del df, is_true
        gc.collect()
    return pool


def evaluate(pairs: pd.DataFrame, truth: dict[str, set[str]],
             entity_order: list[str], pool_size: int) -> tuple[float, float, bool]:
    """Report blocking quality, then sweep accept thresholds against F_0.5."""
    candidates = to_id_lists(pairs)
    for eid in entity_order:
        candidates.setdefault(eid, set())

    recall = candidate_recall(truth, candidates)
    kept = sum(len(v) for v in candidates.values())
    rr = reduction_ratio(candidates, pool_size, 0)
    print(f"\n  candidate recall   : {recall:.4f}   <- ceiling on any model downstream")
    print(f"  reduction ratio    : {rr:.6f}")
    print(f"  candidates kept    : {kept:,} "
          f"({kept/max(len(entity_order),1):.1f} per entity)")

    print(f"\n  {'threshold':>10} {'exclusive':>10} {'F_0.5':>8} {'predicted':>12}")
    print("  " + "-" * 46)
    best = (0.0, ACCEPT_THRESHOLDS[0], True)
    for exclusive in (False, True):
        working = apply_exclusivity(pairs) if exclusive else pairs
        for thr in ACCEPT_THRESHOLDS:
            sel = working[working["cos_name"] >= thr]
            pred = to_id_lists(sel, "candidate_entity_id")
            score = macro_f_beta(truth, pred)
            n_pred = sum(len(v) for v in pred.values())
            flag = ""
            if score > best[0]:
                best = (score, thr, exclusive)
                flag = "  <- best"
            print(f"  {thr:>10.2f} {str(exclusive):>10} {score:>8.4f} {n_pred:>12,}{flag}")

    print(f"\n  BEST local F_0.5   : {best[0]:.4f} "
          f"(threshold {best[1]}, exclusivity {best[2]})")
    print(f"  all-empty baseline : {macro_f_beta(truth, {}):.4f}")
    return best


def run_validate(sample: int, min_score: float) -> None:
    print("Loading training data and ground truth...", flush=True)
    s1 = load_source(TRAIN_SOURCES["S1"], columns=COLUMNS)
    _, val_ids = split_entities(s1["entity_id"].tolist())

    if sample and sample < len(val_ids):
        rng = np.random.default_rng(RANDOM_SEED)
        val_ids = [val_ids[i] for i in rng.choice(len(val_ids), sample, replace=False)]
        print(f"  sampling {sample:,} of the held-out entities", flush=True)

    val_set = set(val_ids)
    queries = s1[s1["entity_id"].isin(val_set)].reset_index(drop=True)
    del s1
    gc.collect()

    truth = load_ground_truth(keep=val_set)
    print(f"  {len(queries):,} held-out Source 1 entities, "
          f"{sum(len(v) for v in truth.values()):,} true links", flush=True)

    pool = build_validation_pool(truth)
    pool_size = sum(len(v) for v in pool.values())

    def load_block(tag: str, country: str) -> pd.DataFrame:
        df = pool[tag]
        return df[df["country"] == country]

    pairs = build_pairs(queries, ["S2", "S3"], load_block, min_score=min_score)
    evaluate(pairs, truth, val_ids, pool_size)


def run_predict(threshold: float, exclusive: bool, min_score: float) -> None:
    print("Loading test data...", flush=True)
    s1 = load_source(TEST_SOURCES["S1"], columns=COLUMNS)
    order_num, _ = encode_ids(s1["entity_id"])
    print(f"  {len(order_num):,} Source 1 entities to produce matches for", flush=True)

    def load_block(tag: str, country: str) -> pd.DataFrame | None:
        path = TEST_SOURCES[tag]
        if not path.exists():
            print(f"  !! {path.name} missing, its matches cannot be produced")
            return None
        return load_source_country(path, country, columns=COLUMNS)

    # Each block's pairs are converted to integer codes and the string frame is
    # dropped straight away. Holding all 36 million pairs as Python strings is
    # what exhausted memory on the first attempt; as int64 plus a source byte
    # the same pairs cost well under a gigabyte.
    left, right, rsrc, cos = [], [], [], []
    for block in iter_block_pairs(s1, ["S2", "S3"], load_block, min_score=min_score):
        if exclusive:
            block = apply_exclusivity(block)
        ln, _ = encode_ids(block["source1_entity_id"])
        rn, rs = encode_ids(block["candidate_entity_id"])
        left.append(ln)
        right.append(rn)
        rsrc.append(rs)
        cos.append(block["cos_name"].to_numpy().astype(np.float32))
        del block
        gc.collect()

    left_num = np.concatenate(left) if left else np.empty(0, np.int64)
    right_num = np.concatenate(right) if right else np.empty(0, np.int64)
    right_src = np.concatenate(rsrc) if rsrc else np.empty(0, np.uint8)
    scores = np.concatenate(cos) if cos else np.empty(0, np.float32)
    del left, right, rsrc, cos
    gc.collect()
    print(f"\n  {len(left_num):,} candidate pairs total "
          f"({left_num.nbytes + right_num.nbytes + right_src.nbytes + scores.nbytes:,} bytes)",
          flush=True)

    # candidate_pairs.tsv must be exactly what the matching stage scored, so it
    # is written from the same frame the threshold is applied to, never from an
    # earlier and wider blocking pass.
    #
    # Written straight from the pair arrays rather than via {entity: set(ids)}.
    # At tens of millions of pairs the dictionary form costs more memory than
    # everything else in the run put together.
    n_cand = write_id_lists_coded(CANDIDATE_PAIRS, CANDIDATE_COLUMNS,
                                  left_num, right_num, right_src, order_num)
    keep = scores >= threshold
    n_match = write_id_lists_coded(MATCHING_RESULTS, RESULT_COLUMNS,
                                   left_num[keep], right_num[keep], right_src[keep],
                                   order_num)
    n_singleton = len(order_num) - len(np.unique(left_num[keep]))
    print(f"\n  wrote {CANDIDATE_PAIRS.name}: {n_cand:,} candidates")
    print(f"  wrote {MATCHING_RESULTS.name}: {n_match:,} matches, "
          f"{n_singleton:,} entities predicted as singletons "
          f"({n_singleton/len(order_num):.1%})")
    print("\n  Now run the official validator before uploading:")
    print("    python data/raw/provided/validate_submission.py \\\n"
          "        --matching output/matching_results.tsv \\\n"
          "        --candidate output/candidate_pairs.tsv \\\n"
          "        --test-dir data/raw/test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="score on held-out training entities")
    ap.add_argument("--predict", action="store_true",
                    help="write matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--sample", type=int, default=0,
                    help="held-out entities to score on (0 = all of them)")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="accept threshold for --predict")
    ap.add_argument("--block-min", type=float, default=BLOCK_MIN_SCORE,
                    help="blocking cutoff; higher keeps fewer candidates and less memory")
    ap.add_argument("--no-exclusive", action="store_true",
                    help="skip the one-owner-per-record constraint")
    args = ap.parse_args()

    if not (args.validate or args.predict):
        ap.error("choose --validate or --predict")

    t0 = time.time()
    if args.validate:
        run_validate(args.sample, args.block_min)
    if args.predict:
        run_predict(args.threshold, not args.no_exclusive, args.block_min)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
