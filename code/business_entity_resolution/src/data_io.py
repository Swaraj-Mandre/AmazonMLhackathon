"""Loading, caching and writing.

The raw TSVs are about 2 GB of text across 24 million rows, which does not fit
comfortably in this machine's free memory once Python turns it into objects. So
every file is converted once into a Parquet cache that carries the normalised
columns alongside the originals, and everything downstream reads that instead.
The conversion streams in chunks and never holds a whole file in memory.

Normalising during conversion rather than at use time matters: `normalize_name`
runs at a few tens of microseconds per record, which is minutes across 24
million rows. Paying that once and caching it keeps every later experiment fast,
and every experiment then sees exactly the same normalised text.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CANDIDATE_COLUMNS,
    GROUND_TRUTH,
    INTERIM_DIR,
    RANDOM_SEED,
    RESULT_COLUMNS,
    SOURCE_COLUMNS,
    TEST_SOURCES,
    TRAIN_SOURCES,
    VALIDATION_FRACTION,
)
from normalize import address_numbers, core_name, has_devanagari, normalize_address  # noqa: E402

csv.field_size_limit(1 << 24)

CHUNK_ROWS = 200_000

# Columns added by the conversion. `core_name` is the blocking key (legal
# suffixes stripped), `addr_nums` holds the digit runs longest-first, and
# `was_transliterated` records whether the original was non-Latin, useful both
# as a model feature and for measuring how much transliteration recovers.
DERIVED_COLUMNS = ["core_name", "norm_addr", "addr_nums", "was_transliterated"]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the normalised columns to a raw chunk."""
    names = df["business_name"].fillna("")
    addrs = df["business_address"].fillna("")
    df["core_name"] = [core_name(n) for n in names]
    df["norm_addr"] = [normalize_address(a) for a in addrs]
    df["addr_nums"] = [" ".join(address_numbers(a)) for a in addrs]
    df["was_transliterated"] = [has_devanagari(n) for n in names]
    return df


def parquet_path(tsv_path: Path) -> Path:
    """Cache location for a given raw file."""
    return INTERIM_DIR / (tsv_path.stem + ".parquet")


def build_cache(tsv_path: Path, force: bool = False) -> Path:
    """Convert one source TSV to normalised Parquet, streaming in chunks.

    Returns the cache path. Re-running is a no-op unless ``force`` is set, so
    this is safe to call at the top of any script.
    """
    out = parquet_path(tsv_path)
    if out.exists() and not force:
        return out

    tmp = out.with_suffix(".parquet.tmp")
    writer = None
    rows = 0
    try:
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            chunksize=CHUNK_ROWS,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
        )
        for chunk in reader:
            chunk = _derive(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            print(f"  {tsv_path.name}: {rows:,} rows", end="\r", flush=True)
    finally:
        if writer is not None:
            writer.close()

    tmp.replace(out)
    print(f"  {tsv_path.name}: {rows:,} rows -> {out.name} "
          f"({out.stat().st_size/1e6:.0f} MB)")
    return out


def build_all_caches(force: bool = False) -> None:
    """Convert every source file we have. Missing files are reported, not fatal."""
    for label, path in {**{f"train_{k}": v for k, v in TRAIN_SOURCES.items()},
                        **{f"test_{k}": v for k, v in TEST_SOURCES.items()}}.items():
        if not path.exists():
            print(f"  {label}: MISSING ({path.name}), skipped")
            continue
        build_cache(path, force=force)


def load_source(tsv_path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a source file from cache, building the cache first if needed.

    Uses the Arrow-backed dtypes so the string columns stay in Arrow buffers
    rather than becoming millions of individual Python objects. On the 5 million
    row sources that is the difference between a few hundred MB and several GB.
    """
    return pd.read_parquet(build_cache(tsv_path), columns=columns,
                           dtype_backend="pyarrow")


def list_countries(tsv_path: Path) -> list[str]:
    """Distinct country labels in a source file.

    Reads only that one column, so it costs almost nothing. The labels are
    discovered rather than hard-coded, which is what lets France flow through
    the pipeline without a special case.
    """
    col = pd.read_parquet(build_cache(tsv_path), columns=["country"],
                          dtype_backend="pyarrow")["country"]
    return sorted(col.dropna().unique().tolist())


def load_source_country(tsv_path: Path, country: str,
                        columns: list[str] | None = None) -> pd.DataFrame:
    """Read only one country's rows, pushing the filter down into Parquet.

    Loading a whole 5 million row source and then subsetting it peaks at several
    GB before the subset is taken, which on a 16 GB machine means swapping. This
    never materialises the rows we are going to discard.
    """
    return pd.read_parquet(build_cache(tsv_path), columns=columns,
                           filters=[("country", "==", country)],
                           dtype_backend="pyarrow")


def iter_source_chunks(
    tsv_path: Path,
    columns: list[str] | None = None,
    batch_rows: int = 500_000,
) -> Iterator[pd.DataFrame]:
    """Stream a cached source in row batches, for files too large to hold at once."""
    pf = pq.ParquetFile(build_cache(tsv_path))
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def load_ground_truth(keep: set[str] | None = None) -> dict[str, set[str]]:
    """Read ``train_ground_truth.tsv`` into ``{source1_id: {matched ids}}``.

    Pass ``keep`` to load only those Source 1 entities. The full file is 2.2
    million entities holding 7.6 million IDs, which as Python sets costs a
    couple of GB; when only a held-out slice is being scored, filtering while
    reading avoids building the rest at all.
    """
    truth: dict[str, set[str]] = {}
    with open(GROUND_TRUTH, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for row in reader:
            if keep is not None and row[0] not in keep:
                continue
            ids = row[1] if len(row) > 1 else ""
            truth[row[0]] = {x for x in ids.split(",") if x}
    return truth


def split_entities(
    entity_ids: list[str],
    fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """Split Source 1 entity IDs into (train, validation).

    **By entity, never by pair.** A pair-level split puts other matches of the
    same Source 1 entity on both sides, which leaks the answer and makes every
    local score optimistic. Deterministic given the seed, so all of us score on
    exactly the same held-out entities.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ids = np.array(sorted(entity_ids))
    mask = rng.random(len(ids)) < fraction
    return ids[~mask].tolist(), ids[mask].tolist()


def _write_id_lists(path: Path, columns: list[str], rows: dict[str, set[str]],
                    order: list[str]) -> None:
    """Write a two-column TSV of ``id -> comma-separated id list``.

    ``order`` fixes the row set and the row order: every entity in it gets
    exactly one row, whether or not it has any IDs. That is the rule the
    official validator enforces most strictly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for eid in order:
            ids = rows.get(eid) or set()
            fh.write(f"{eid}\t{','.join(sorted(ids))}\n")


def encode_ids(ids) -> tuple["np.ndarray", "np.ndarray"]:
    """Split entity IDs into a numeric part and a source number.

    Every ID in the data is a ``S1-``/``S2-``/``S3-`` prefix followed by digits
    with no leading zeros (verified across all 11,702,133 test IDs), so this
    round-trips exactly. It matters because the prediction run produces tens of
    millions of pairs: as Python strings those cost gigabytes, as int64 plus a
    source byte they cost about 9 bytes each.
    """
    import numpy as np
    import pandas as pd

    s = pd.Series(ids, dtype="string").astype(str)
    src = s.str.slice(1, 2).to_numpy().astype(np.uint8)
    num = s.str.slice(3).to_numpy().astype(np.int64)
    return num, src


def write_id_lists_coded(path: Path, columns: list[str],
                         left_num, right_num, right_src, order_num) -> int:
    """Write an ID-list TSV from integer-coded pairs, in the given row order.

    Sorts the pairs once, records where each left-hand ID's run begins, then
    walks ``order_num`` so the output keeps the test file's own row order. Never
    materialises the ID strings for more than one row at a time.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.argsort(left_num, kind="stable")
    left_num = left_num[idx]
    right_num = right_num[idx]
    right_src = right_src[idx]

    if len(left_num):
        starts = np.flatnonzero(np.r_[True, left_num[1:] != left_num[:-1]])
        ends = np.r_[starts[1:], len(left_num)]
        spans = dict(zip(left_num[starts].tolist(), zip(starts.tolist(), ends.tolist())))
    else:
        spans = {}

    written = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for eid in order_num.tolist():
            span = spans.get(eid)
            if span is None:
                fh.write(f"S1-{eid}\t\n")
                continue
            s, e = span
            # Duplicates inside one list are a rejection, so collapse them here.
            ids = sorted({f"S{src}-{num}" for num, src
                          in zip(right_num[s:e].tolist(), right_src[s:e].tolist())})
            written += len(ids)
            fh.write(f"S1-{eid}\t{','.join(ids)}\n")
    return written


def write_id_lists_from_pairs(path: Path, columns: list[str], pairs,
                              id_col: str, order: list[str]) -> int:
    """Write an ID-list TSV straight from a pair frame, without building dicts.

    ``{entity: set(ids)}`` is the obvious intermediate, but a Python set costs
    roughly 60 bytes per element and the test run produces tens of millions of
    pairs, so the dictionary alone runs to gigabytes. Sorting the pair arrays and
    walking them in order needs one index of 1.7 million offsets instead.

    Returns the number of IDs written.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    left = pairs["source1_entity_id"].to_numpy()
    right = pairs[id_col].to_numpy()

    order_index = np.argsort(left, kind="stable")
    left, right = left[order_index], right[order_index]
    # Start offset of each run of equal source1 IDs.
    starts = np.flatnonzero(np.r_[True, left[1:] != left[:-1]]) if len(left) else np.empty(0, int)
    ends = np.r_[starts[1:], len(left)] if len(starts) else np.empty(0, int)
    spans = {left[s]: (s, e) for s, e in zip(starts, ends)}

    written = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for eid in order:
            span = spans.get(eid)
            if span is None:
                fh.write(f"{eid}\t\n")
                continue
            s, e = span
            # Duplicates inside one list are a rejection, so de-duplicate here.
            ids = sorted(set(right[s:e].tolist()))
            written += len(ids)
            fh.write(f"{eid}\t{','.join(ids)}\n")
    return written


def write_matching_results(path: Path, matches: dict[str, set[str]],
                           order: list[str]) -> None:
    """Write ``matching_results.tsv``, the file scored on the leaderboard."""
    _write_id_lists(path, RESULT_COLUMNS, matches, order)


def write_candidate_pairs(path: Path, candidates: dict[str, set[str]],
                          order: list[str]) -> None:
    """Write ``candidate_pairs.tsv``, the blocking set Amazon audits."""
    _write_id_lists(path, CANDIDATE_COLUMNS, candidates, order)


def _self_test() -> None:
    import tempfile

    tr, va = split_entities([f"S1-{i}" for i in range(10000)])
    assert len(tr) + len(va) == 10000
    assert 0.12 < len(va) / 10000 < 0.18, len(va) / 10000
    assert not (set(tr) & set(va)), "splits must be disjoint"
    assert split_entities([f"S1-{i}" for i in range(10000)])[1] == va, "must be deterministic"

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.tsv"
        write_matching_results(p, {"S1-2": {"S2-9", "S3-1"}}, ["S1-1", "S1-2"])
        text = p.read_text(encoding="utf-8")
        assert text.startswith("source1_entity_id\tmatched_entity_ids\n"), text
        # An entity with no matches still gets a row, with an empty list.
        assert "S1-1\t\n" in text, text
        assert "S1-2\tS2-9,S3-1\n" in text, text

    assert SOURCE_COLUMNS[0] == "entity_id"
    print("data_io.py: all checks passed")


if __name__ == "__main__":
    if "--build-cache" in sys.argv:
        build_all_caches(force="--force" in sys.argv)
    else:
        _self_test()
