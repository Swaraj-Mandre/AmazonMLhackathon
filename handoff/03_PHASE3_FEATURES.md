# Phase 3: pairwise features

Paste `01_PROJECT_CONTEXT.md` first, then this.

## The job

For every candidate pair that blocking produced, build a row of numbers
describing how alike the two records are. These become the input to the
classifier in phase 4. No model yet, just the feature table.

## Features to build

Each row is one pair: a Source 1 record and a candidate record.

**Name features**

- TF-IDF cosine on `core_name`. Blocking already computes this, so carry it
  through rather than recomputing.
- TF-IDF cosine on the full normalised name, which keeps the legal suffixes.
- Token Jaccard over the name words.
- `rapidfuzz` token set ratio, partial ratio, and Jaro-Winkler.
- **IDF-weighted rare token overlap.** Two records sharing an unusual word are
  far stronger evidence than two sharing "restaurant". This is usually the
  strongest single feature in this kind of task, so do not skip it.
- Word count on each side, and the ratio between them.

**Address features**

- TF-IDF cosine on `norm_addr`.
- Token Jaccard on the address.
- Whether the longest number matches, which is usually the PIN code or postcode.
- How many numbers the two share, and how many each side has.
- Flags for whether either address is empty. About 3.3% of Source 2 and 3 rows
  have none.

**Record features**

- `was_transliterated` on either side.
- Name and address string lengths, and their ratios.

## What not to do

- **Do not one-hot the country.** It encodes US and India and will behave
  unpredictably on France, which is 15% of the test set. After blocking, both
  sides are the same country anyway, so the feature carries no information.
- Do not add features that need outside data. No geocoding, no lookups. That is
  an instant disqualification.

## Practical notes

- `rapidfuzz` is the fast C++ implementation. Pure Python string distance will be
  far too slow across tens of millions of pairs.
- Compute features in blocks and keep dtypes small, `float32` rather than
  `float64`. The feature table is large.
- The cleaned columns `core_name`, `norm_addr`, `addr_nums` and
  `was_transliterated` are already cached on every row by `data_io.py`, so
  nothing needs recomputing from the raw text.

## Definition of done

A function that takes a block of candidate pairs and returns a numeric feature
matrix with a documented column order, fast enough to run over the full candidate
set, plus a quick sanity check that true pairs score higher than random ones on
the obvious features.
