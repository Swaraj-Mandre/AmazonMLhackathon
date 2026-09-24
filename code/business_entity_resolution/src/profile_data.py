"""One-off exploration of the provided data.

Answers the questions that decide the architecture before any modelling starts:
how many Source 1 entities are singletons (which sets the floor a do-nothing
submission would score), how matches are distributed across Source 2 and
Source 3, whether an S2/S3 record can belong to more than one S1 entity, and
how the country labels are spread across train and test.

Run from the repository root:

    python code/business_entity_resolution/src/profile_data.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GROUND_TRUTH, SOURCE_FILES  # noqa: E402

csv.field_size_limit(1 << 24)


def profile_ground_truth() -> None:
    """Match-count distribution, singleton rate, and the S2/S3 reuse question."""
    n_entities = 0
    n_singletons = 0
    match_counts = Counter()
    per_source = Counter()
    # A Source 2 / Source 3 record appearing under two different Source 1 entities
    # would mean the reference source is not truly deduplicated. If it never
    # happens, "each S2/S3 record belongs to at most one S1" is a constraint we
    # can exploit at prediction time.
    seen_rhs: set[str] = set()
    reused = 0
    total_matches = 0

    with open(GROUND_TRUTH, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for row in reader:
            n_entities += 1
            ids = [x for x in (row[1].split(",") if len(row) > 1 and row[1] else []) if x]
            match_counts[len(ids)] += 1
            total_matches += len(ids)
            if not ids:
                n_singletons += 1
            for i in ids:
                per_source[i[:2]] += 1
                if i in seen_rhs:
                    reused += 1
                else:
                    seen_rhs.add(i)

    print("=" * 62)
    print("GROUND TRUTH")
    print("=" * 62)
    print(f"Source 1 entities           : {n_entities:,}")
    print(f"Singletons (no match)       : {n_singletons:,}  ({n_singletons/n_entities:6.2%})")
    print(f"  -> an all-empty submission would score exactly this as macro F_0.5")
    print(f"Total match links           : {total_matches:,}")
    print(f"Mean matches per entity     : {total_matches/n_entities:.2f}")
    print(f"Links from Source 2         : {per_source['S2']:,}")
    print(f"Links from Source 3         : {per_source['S3']:,}")
    print(f"Distinct S2/S3 records used : {len(seen_rhs):,}")
    print(f"S2/S3 records claimed twice : {reused:,}"
          f"{'   <- many-to-one is possible' if reused else '   <- each RHS record belongs to at most ONE S1'}")
    print("\nMatches per entity:")
    for k in sorted(match_counts)[:12]:
        bar = "#" * int(60 * match_counts[k] / n_entities)
        print(f"  {k:>3} matches : {match_counts[k]:>10,} ({match_counts[k]/n_entities:6.2%}) {bar}")
    tail = sum(v for k, v in match_counts.items() if k >= 12)
    if tail:
        print(f"  12+ matches: {tail:>10,} ({tail/n_entities:6.2%})")
    print(f"  max observed: {max(match_counts)}")


def profile_sources() -> None:
    """Country spread and field emptiness for every source file we have."""
    print("\n" + "=" * 62)
    print("SOURCE FILES")
    print("=" * 62)
    for label, path in SOURCE_FILES.items():
        if not path.exists():
            print(f"\n{label:14} MISSING -> {path}")
            continue
        countries = Counter()
        rows = empty_name = empty_addr = 0
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                rows += 1
                countries[row.get("country", "")] += 1
                if not (row.get("business_name") or "").strip():
                    empty_name += 1
                if not (row.get("business_address") or "").strip():
                    empty_addr += 1
        spread = "  ".join(f"{c}={n:,} ({n/rows:.1%})" for c, n in countries.most_common())
        print(f"\n{label:14} {rows:>10,} rows")
        print(f"  countries    {spread}")
        print(f"  empty name   {empty_name:,}   empty address {empty_addr:,}")


if __name__ == "__main__":
    profile_ground_truth()
    profile_sources()
