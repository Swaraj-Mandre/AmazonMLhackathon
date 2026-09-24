# Phase 2: better candidate generation (blocking)

Paste `01_PROJECT_CONTEXT.md` first, then this.

## The job

Raise candidate recall from **0.7123** towards **0.95**, without letting the
number of candidates per record grow beyond the low tens.

Candidate recall is the share of true matches that appear anywhere in our
candidate set. It is a hard ceiling: a match that blocking never proposes cannot
be recovered by any model later. This is the single most valuable phase left.

## Why recall is low right now

Blocking currently runs in one direction only. Every Source 2/3 record proposes
its top 5 Source 1 records. A true match is lost whenever the correct record does
not make that candidate's top 5, and that gets more likely as the number of
records grows. We measured it directly:

| Held-out records | Candidate recall |
|---|---|
| 40,000 | 0.7971 |
| 329,348 | 0.7123 |

At the real 1,732,544 records it will be worse again.

## What to try, in order of expected payoff

1. **Scan both directions and union the results.** Add the opposite pass where
   each Source 1 record proposes its own top k candidates, then merge the two
   candidate sets. A true match then only has to survive one of the two
   directions. This should be the biggest single win.
2. **Add a second similarity channel.** Character n-grams of 3 to 4 characters
   catch typos and word-order changes that whole-word TF-IDF misses. Run it as a
   separate channel and union the results, rather than replacing the word
   channel.
3. **Add an address-number key.** Two records sharing a rare long number, a PIN
   code or postcode, plus a house number, are strong candidates even when the
   names look different. `addr_nums` is already cached on every row, longest
   number first. Cheap, and it catches a class the name channels cannot.
4. **Raise `TOP_K_PER_CANDIDATE`** from 5. Simple, and worth measuring before
   anything clever. Watch the candidates-per-record count and memory.
5. **Check the transliterated subset separately.** Measure recall on rows where
   `was_transliterated` is true. If it is much worse than the rest, the
   transliteration is the bottleneck for that group rather than the blocking.

## How to measure, every single time

```python
from metric import candidate_recall, reduction_ratio
```

- `candidate_recall(truth, candidates)` is the number to maximise.
- `reduction_ratio(candidates, n_source2, n_source3)` is how much of the full
  cross product was pruned. Amazon audits both numbers.
- Then run the whole thing and check F0.5 actually moved:
  `python code/business_entity_resolution/src/run_baseline.py --validate`

Record every attempt in `experiments/experiments.md`, including the ones that
made things worse.

## Constraints

- Work one country at a time. Country blocking is lossless and verified, and it
  keeps each block small enough to handle.
- Never hard-code the country list. France is in the test set but not in
  training, and must flow through the same code path.
- Keep `CANDIDATE_CHUNK` at 6,000 or lower unless there is memory to spare.
- More candidates costs memory and time. Recall of 0.95 at 200 candidates per
  record is not useful. Aim for the low tens.

## Definition of done

Candidate recall meaningfully above 0.7123 on the full held-out split, candidates
per record still in the low tens, logged in the experiment file, and F0.5
rechecked.
