"""Phase 1, rules-only baseline, end to end.

No machine learning: normalise, block by country, score candidate pairs by
TF-IDF cosine over business names, accept everything above a threshold. The
point is not the score but the scaffolding, it proves the output format, gives
the leaderboard a real number to calibrate against, and establishes the local
validation harness every later phase is measured on.

Two modes::

    python src/run_baseline.py --validate            # score on held-out entities
    python src/run_baseline.py --predict             # write the submission files

``--validate`` samples the held-out entities by default (``--sample``), because
the candidate side is the full 10 million records either way and query count is
what drives runtime. The sample only affects the precision of the estimate, not
its correctness.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blocking import fit_vectorizer, generate_candidates, to_id_lists  # noqa: E402
from config import (  # noqa: E402
    CANDIDATE_PAIRS,
    MATCHING_RESULTS,
    RANDOM_SEED,
    TEST_SOURCES,
    TRAIN_SOURCES,
)
from data_io import (  # noqa: E402
    load_ground_truth,
    load_source,
    split_entities,
    write_candidate_pairs,
    write_matching_results,
)
from metric import candidate_recall, macro_f_beta  # noqa: E402

# Blocking keeps anything this similar or better; the accept threshold applied
# afterwards is always higher. Keeping the two separate is what lets us sweep
# the accept threshold without regenerating candidates.
BLOCK_MIN_SCORE = 0.30
TOP_K_PER_CANDIDATE = 5
ACCEPT_THRESHOLDS = [0.45, 0.55, 0.65, 0.75, 0.85, 0.92]

COLUMNS = ["entity_id", "core_name", "country"]


def build_pairs(queries: pd.DataFrame, sources: dict[str, Path],
                verbose: bool = True) -> pd.DataFrame:
    """Generate candidate pairs for every country block, across every source file.

    Countries are taken from whatever labels appear in the data rather than any
    fixed list, so an unseen country (France in the test set) forms its own block
    like any other.
    """
    frames = []
    for tag, path in sources.items():
        if tag == "S1":
            continue
        if not path.exists():
            print(f"  !! {path.name} missing, its matches cannot be produced")
            continue
        candidates = load_source(path, columns=COLUMNS)
        for country in sorted(queries["country"].unique()):
            q = queries[queries["country"] == country]
            c = candidates[candidates["country"] == country]
            if verbose:
                print(f"  {tag} / {country}: {len(q):,} entities x {len(c):,} records")
            t0 = time.time()
            # The vectorizer is fitted per block so that IDF reflects that
            # country's own vocabulary, "nagar" is common in India and rare in
            # the US, and the weighting should say so.
            vec = fit_vectorizer(pd.concat([q["core_name"], c["core_name"]],
                                           ignore_index=True))
            pairs = generate_candidates(
                q, c,
                top_k=TOP_K_PER_CANDIDATE,
                min_score=BLOCK_MIN_SCORE,
                vectorizer=vec,
                verbose=verbose,
            )
            if verbose:
                print(f"    -> {len(pairs):,} pairs in {time.time()-t0:.0f}s")
            frames.append(pairs)
        del candidates

    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "cos_name"])
    return pd.concat(frames, ignore_index=True)


def apply_exclusivity(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep only each candidate record's single best Source 1 entity.

    Justified by a measured property of the training data: across all 7,638,365
    match links, no Source 2 or Source 3 record is ever claimed by two different
    Source 1 entities. Enforcing it removes contested pairs, which costs a
    little recall and buys precision, and F_0.5 weights precision twice as
    heavily.
    """
    if pairs.empty:
        return pairs
    order = pairs["cos_name"].to_numpy().argsort()[::-1]
    return pairs.iloc[order].drop_duplicates("candidate_entity_id", keep="first")


def evaluate(pairs: pd.DataFrame, truth: dict[str, set[str]],
             entity_order: list[str]) -> None:
    """Report candidate recall, then sweep accept thresholds against F_0.5."""
    candidates = to_id_lists(pairs)
    for eid in entity_order:
        candidates.setdefault(eid, set())

    recall = candidate_recall(truth, candidates)
    kept = sum(len(v) for v in candidates.values())
    print(f"\n  candidate recall   : {recall:.4f}   <- ceiling on any model downstream")
    print(f"  candidates kept    : {kept:,} "
          f"({kept/max(len(entity_order),1):.1f} per entity)")

    print(f"\n  {'threshold':>10} {'exclusive':>10} {'F_0.5':>8} {'predicted':>12}")
    print("  " + "-" * 44)
    best = (0.0, None)
    for exclusive in (False, True):
        working = apply_exclusivity(pairs) if exclusive else pairs
        for thr in ACCEPT_THRESHOLDS:
            sel = working[working["cos_name"] >= thr]
            pred = to_id_lists(sel, "candidate_entity_id")
            score = macro_f_beta(truth, pred)
            n_pred = sum(len(v) for v in pred.values())
            flag = ""
            if score > best[0]:
                best = (score, (thr, exclusive))
                flag = "  <- best"
            print(f"  {thr:>10.2f} {str(exclusive):>10} {score:>8.4f} {n_pred:>12,}{flag}")

    thr, exclusive = best[1]
    print(f"\n  BEST local F_0.5   : {best[0]:.4f} "
          f"(threshold {thr}, exclusivity {exclusive})")
    print(f"  all-empty baseline : {macro_f_beta(truth, {}):.4f}")


def run_validate(sample: int) -> None:
    print("Loading training data and ground truth...")
    s1 = load_source(TRAIN_SOURCES["S1"], columns=COLUMNS)
    truth_all = load_ground_truth()
    _, val_ids = split_entities(s1["entity_id"].tolist())

    if sample and sample < len(val_ids):
        rng = np.random.default_rng(RANDOM_SEED)
        val_ids = [val_ids[i] for i in
                   rng.choice(len(val_ids), sample, replace=False)]
        print(f"  sampling {sample:,} of the held-out entities for speed")

    val_set = set(val_ids)
    queries = s1[s1["entity_id"].isin(val_set)].reset_index(drop=True)
    truth = {k: truth_all[k] for k in val_ids}
    print(f"  {len(queries):,} held-out Source 1 entities, "
          f"{sum(len(v) for v in truth.values()):,} true links")

    pairs = build_pairs(queries, TRAIN_SOURCES)
    evaluate(pairs, truth, val_ids)


def run_predict(threshold: float, exclusive: bool) -> None:
    print("Loading test data...")
    s1 = load_source(TEST_SOURCES["S1"], columns=COLUMNS)
    order = s1["entity_id"].tolist()
    print(f"  {len(order):,} Source 1 entities to produce matches for")

    pairs = build_pairs(s1, TEST_SOURCES)
    if exclusive:
        pairs = apply_exclusivity(pairs)

    # candidate_pairs.tsv must be exactly what the matching stage scored, so it
    # is written from the same frame the threshold is applied to, never from an
    # earlier, wider blocking pass.
    candidates = to_id_lists(pairs)
    matches = to_id_lists(pairs[pairs["cos_name"] >= threshold])

    write_candidate_pairs(CANDIDATE_PAIRS, candidates, order)
    write_matching_results(MATCHING_RESULTS, matches, order)

    n_match = sum(len(v) for v in matches.values())
    n_singleton = sum(1 for e in order if not matches.get(e))
    print(f"\n  wrote {CANDIDATE_PAIRS.name}: "
          f"{sum(len(v) for v in candidates.values()):,} candidates")
    print(f"  wrote {MATCHING_RESULTS.name}: {n_match:,} matches, "
          f"{n_singleton:,} entities predicted as singletons "
          f"({n_singleton/len(order):.1%})")
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
    ap.add_argument("--sample", type=int, default=40_000,
                    help="held-out entities to score on (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="accept threshold for --predict")
    ap.add_argument("--no-exclusive", action="store_true",
                    help="skip the one-owner-per-record constraint")
    args = ap.parse_args()

    if not (args.validate or args.predict):
        ap.error("choose --validate or --predict")

    t0 = time.time()
    if args.validate:
        run_validate(args.sample)
    if args.predict:
        run_predict(args.threshold, not args.no_exclusive)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
