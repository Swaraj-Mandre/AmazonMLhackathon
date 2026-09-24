"""Official F_0.5 metric for the Business Entity Resolution challenge.

Scoring is defined per Source 1 entity and then macro-averaged, so every entity
counts the same regardless of how many matches it has. Singletons are included:
an entity with no true matches scores 1.0 for a correctly predicted empty list
and 0.0 for any prediction at all.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

This module is the single source of truth for every score the team quotes.
Nothing else in the pipeline should compute F_0.5 independently.
"""

from __future__ import annotations

import csv
from pathlib import Path


def parse_id_list(cell: str | None) -> set[str]:
    """Split a ``matched_entity_ids`` cell into a set of entity IDs.

    Handles the empty/missing cell (a singleton) and strips incidental
    whitespace around the commas.
    """
    if not cell:
        return set()
    return {part.strip() for part in cell.split(",") if part.strip()}


def entity_f_beta(truth: set[str], pred: set[str], beta: float = 0.5) -> float:
    """F_beta for a single Source 1 entity.

    Both empty means the singleton was predicted correctly, which the problem
    statement scores as a full 1.0. Any other mismatch involving an empty set
    scores 0.0, since precision or recall is undefined-and-worthless there.
    """
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0

    hits = len(truth & pred)
    if hits == 0:
        return 0.0

    precision = hits / len(pred)
    recall = hits / len(truth)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_f_beta(
    truth: dict[str, set[str]],
    pred: dict[str, set[str]],
    beta: float = 0.5,
) -> float:
    """Macro-average of :func:`entity_f_beta` over every Source 1 entity in ``truth``.

    ``truth`` defines the evaluation set. A Source 1 entity missing from ``pred``
    is treated as an empty prediction rather than skipped, which mirrors how the
    leaderboard treats a submission that omits rows (there, it is a rejection -
    here it keeps the local score honest instead of silently flattering us).
    """
    if not truth:
        raise ValueError("empty ground truth: nothing to score")

    total = sum(
        entity_f_beta(matches, pred.get(s1_id, set()), beta)
        for s1_id, matches in truth.items()
    )
    return total / len(truth)


def load_id_map(path: str | Path, key_col: str, value_col: str) -> dict[str, set[str]]:
    """Read a two-column TSV of ``entity_id -> comma-separated ID list``.

    Works for ``train_ground_truth.tsv``, ``matching_results.tsv`` and
    ``candidate_pairs.tsv`` alike, since all three share that shape.
    """
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        missing = {key_col, value_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {sorted(missing)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            out[row[key_col].strip()] = parse_id_list(row.get(value_col))
    return out


def candidate_recall(
    truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
) -> float:
    """Share of true matches that survive blocking, our ceiling on recall.

    Anything lost here cannot be recovered by the matching model, so this is the
    number to watch while tuning candidate generation.
    """
    found = total = 0
    for s1_id, matches in truth.items():
        total += len(matches)
        found += len(matches & candidates.get(s1_id, set()))
    return found / total if total else 1.0


def reduction_ratio(
    candidates: dict[str, set[str]],
    n_source2: int,
    n_source3: int,
) -> float:
    """Fraction of the full cross-product that blocking discarded.

    1.0 means everything was pruned, 0.0 means we kept every possible pair.
    Reported alongside :func:`candidate_recall`, the two trade off.
    """
    full = len(candidates) * (n_source2 + n_source3)
    if full == 0:
        return 1.0
    kept = sum(len(c) for c in candidates.values())
    return 1.0 - kept / full


def _self_test() -> None:
    """Check the implementation against the worked example in the problem statement."""
    # Problem statement: predicting [S2-00047, S2-00193, S3-00812] against a
    # ground truth of [S2-00047, S3-00812] gives precision 2/3, recall 1.0,
    # and F_0.5 = 0.714.
    got = entity_f_beta({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(got - 0.714) < 5e-4, f"PS worked example: expected 0.714, got {got:.6f}"

    assert entity_f_beta(set(), set()) == 1.0, "correct singleton must score 1.0"
    assert entity_f_beta(set(), {"S2-1"}) == 0.0, "false merge on singleton must score 0.0"
    assert entity_f_beta({"S2-1"}, set()) == 0.0, "missing a real match must score 0.0"
    assert entity_f_beta({"S2-1"}, {"S2-1"}) == 1.0, "exact match must score 1.0"
    assert entity_f_beta({"S2-1"}, {"S2-2"}) == 0.0, "wrong match must score 0.0"

    # One right plus one wrong costs ~0.44; firing on a singleton costs the
    # whole point. This asymmetry is the reason the pipeline stays conservative.
    one_wrong_added = entity_f_beta({"S2-1"}, {"S2-1", "S2-9"})
    assert abs(one_wrong_added - 0.5556) < 5e-4, one_wrong_added

    # Recall is cheaper to lose than precision: dropping one of two true matches
    # must score higher than adding one false one to a complete prediction.
    missed_one = entity_f_beta({"S2-1", "S2-2"}, {"S2-1"})
    added_one = entity_f_beta({"S2-1", "S2-2"}, {"S2-1", "S2-2", "S2-9"})
    assert missed_one > added_one, (missed_one, added_one)

    macro = macro_f_beta(
        {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-3"}},
        {"S1-1": {"S2-1"}, "S1-2": set()},  # S1-3 omitted -> treated as empty
    )
    assert abs(macro - 2 / 3) < 1e-9, macro

    assert parse_id_list("") == set()
    assert parse_id_list(None) == set()
    assert parse_id_list("S2-1, S3-2 ,") == {"S2-1", "S3-2"}

    assert candidate_recall({"S1-1": {"S2-1", "S2-2"}}, {"S1-1": {"S2-1", "S2-9"}}) == 0.5

    print("metric.py: all checks passed (PS worked example = %.6f)" % got)


if __name__ == "__main__":
    _self_test()
