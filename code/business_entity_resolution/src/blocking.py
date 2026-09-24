"""Candidate generation.

Blocking sets the ceiling on recall: a true match that never enters the
candidate set cannot be recovered by any model downstream, so this is the stage
worth the most careful attention.

The search is 1.7 million Source 1 entities against 10 million Source 2 and
Source 3 records, 17 trillion pairs if compared exhaustively. Two measured
properties of the data cut that down:

* **Matches never cross country labels** (0 exceptions in 1,038,755 sampled
  training links), so each country is blocked independently. The label is
  treated as an open set: whatever values appear get their own block, which is
  why the unseen France records are handled without a special case.
* **Each Source 2/3 record belongs to at most one Source 1 entity** (0 reuse
  across all 7,638,365 training links). That makes the natural search direction
  *candidate to entity* rather than the reverse, every candidate has at most
  one correct answer to find, and scanning that way keeps memory bounded, since
  each chunk of candidates produces a small fixed-size result.

Similarity is TF-IDF cosine over whitespace tokens of `core_name`. Token-level
rather than character-level because the vocabulary stays sparse enough for the
chunked sparse product to fit in memory; the IDF weighting is what does the real
work, since sharing a rare token like "Zephay" is far stronger evidence than
sharing "Restaurant".
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tokens are already normalised by `normalize.py`, so the analyser only has to
# split on whitespace, no further lowercasing or stripping.
VECTORIZER_KWARGS = dict(
    analyzer="word",
    token_pattern=r"\S+",
    lowercase=False,
    sublinear_tf=True,
    dtype=np.float32,
    min_df=1,
    # A token in more than 20% of records carries no signal and would make the
    # sparse product explode. Dropping it is both a quality and a memory measure.
    max_df=0.2,
)

CANDIDATE_CHUNK = 20_000


def fit_vectorizer(texts: pd.Series, sample: int = 1_000_000,
                   seed: int = 0) -> TfidfVectorizer:
    """Learn the vocabulary and IDF weights.

    Fitted on a random sample rather than the full corpus: IDF is a smooth
    statistic that a million documents estimate perfectly well, and fitting on
    all 12 million would cost several GB of vocabulary for no measurable gain.
    """
    if len(texts) > sample:
        rng = np.random.default_rng(seed)
        texts = texts.iloc[rng.choice(len(texts), sample, replace=False)]
    vec = TfidfVectorizer(**VECTORIZER_KWARGS)
    vec.fit(texts)
    return vec


def _top_k_per_row(sim: sp.csr_matrix, k: int, min_score: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of a sparse similarity matrix, the k highest-scoring columns.

    Returns parallel arrays of (row, column, score). Rows with nothing above
    ``min_score`` contribute nothing, which is what keeps the output small.
    """
    rows, cols, scores = [], [], []
    indptr, indices, data = sim.indptr, sim.indices, sim.data
    for r in range(sim.shape[0]):
        lo, hi = indptr[r], indptr[r + 1]
        if lo == hi:
            continue
        d = data[lo:hi]
        keep = d >= min_score
        if not keep.any():
            continue
        c = indices[lo:hi][keep]
        d = d[keep]
        if len(d) > k:
            sel = np.argpartition(-d, k)[:k]
            c, d = c[sel], d[sel]
        rows.append(np.full(len(d), r, dtype=np.int64))
        cols.append(c)
        scores.append(d)
    if not rows:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(scores)


def generate_candidates(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    top_k: int = 8,
    min_score: float = 0.30,
    vectorizer: TfidfVectorizer | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Candidate pairs between one country's Source 1 entities and its S2/S3 records.

    Both frames need ``entity_id`` and ``core_name``. ``top_k`` is applied per
    *candidate*, each Source 2/3 record proposes its best few Source 1 entities
   , which follows the at-most-one-owner structure of the data and bounds
    memory regardless of how large the entity side grows.

    Returns a frame of ``source1_entity_id, candidate_entity_id, cos_name``.
    """
    if len(queries) == 0 or len(candidates) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "cos_name"])

    if vectorizer is None:
        vectorizer = fit_vectorizer(pd.concat([queries["core_name"],
                                               candidates["core_name"]],
                                              ignore_index=True))

    q_matrix = vectorizer.transform(queries["core_name"])
    q_matrix = q_matrix.T.tocsc()  # (vocab x n_queries), ready for chunk @ q_matrix
    q_ids = queries["entity_id"].to_numpy()
    c_ids = candidates["entity_id"].to_numpy()

    out_q, out_c, out_s = [], [], []
    n = len(candidates)
    for start in range(0, n, CANDIDATE_CHUNK):
        stop = min(start + CANDIDATE_CHUNK, n)
        c_matrix = vectorizer.transform(candidates["core_name"].iloc[start:stop])
        sim = (c_matrix @ q_matrix).tocsr()
        rows, cols, scores = _top_k_per_row(sim, top_k, min_score)
        if len(rows):
            out_c.append(c_ids[start:stop][rows])
            out_q.append(q_ids[cols])
            out_s.append(scores)
        if verbose:
            print(f"    candidates {stop:,}/{n:,}", end="\r", flush=True)

    if not out_q:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "cos_name"])

    return pd.DataFrame({
        "source1_entity_id": np.concatenate(out_q),
        "candidate_entity_id": np.concatenate(out_c),
        "cos_name": np.concatenate(out_s),
    })


def to_id_lists(pairs: pd.DataFrame, id_col: str = "candidate_entity_id"
                ) -> dict[str, set[str]]:
    """Collapse a pair frame into ``{source1_entity_id: {ids}}``."""
    out: dict[str, set[str]] = defaultdict(set)
    for s1, other in zip(pairs["source1_entity_id"].to_numpy(),
                         pairs[id_col].to_numpy()):
        out[s1].add(other)
    return dict(out)


def _self_test() -> None:
    queries = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "core_name": ["zephay labs", "prime money"],
    })
    candidates = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3"],
        "core_name": ["zephay labs", "prime money", "totally unrelated bakery"],
    })
    vec = TfidfVectorizer(**{**VECTORIZER_KWARGS, "max_df": 1.0})
    vec.fit(pd.concat([queries["core_name"], candidates["core_name"]]))

    pairs = generate_candidates(queries, candidates, top_k=3, min_score=0.1,
                                vectorizer=vec, verbose=False)
    lists = to_id_lists(pairs)
    assert "S2-1" in lists["S1-1"], lists
    assert "S2-2" in lists["S1-2"], lists
    # The unrelated record shares no token, so it reaches nobody.
    assert all("S2-3" not in v for v in lists.values()), lists

    empty = generate_candidates(queries.iloc[:0], candidates, verbose=False)
    assert len(empty) == 0 and list(empty.columns)[0] == "source1_entity_id"

    print("blocking.py: all checks passed")


if __name__ == "__main__":
    _self_test()
