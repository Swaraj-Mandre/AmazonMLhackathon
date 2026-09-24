# Rules, deadlines and traps

Paste this when the question is about submissions, rules, or what we are allowed
to do.

## Deadlines

| Event | When |
|---|---|
| Challenge window | 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST |
| Top 50 announced | 2 Oct 2026 |
| Grand finale | 7 Oct 2026 |

The portal timer does not stop once started. Ties on the leaderboard go to
whoever submitted earlier, so a good result should go up as soon as it exists
rather than being sat on.

## Uploads

- **Five per day, fifteen in total.** The button disables after the fifth.
- **Only the registered team leader can access the portal.** Everyone else works
  offline and hands files over.
- Each upload takes two files: `matching_results.tsv`, and a zip of the code.

Nothing goes up unless both are true:

1. It beats our best local F0.5.
2. The official validator passes:

```bash
python data/raw/provided/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/raw/test
```

`make_submission.py` runs that validator automatically and refuses to build if it
fails.

## Instant disqualification

- **Any external data lookup.** No entity resolution APIs, no business registries,
  **no geocoding APIs for addresses**, no scraping, no internet data
  augmentation. Amazon reviews the code pipelines.
  - Hand-written mapping tables in our own source, like the Devanagari table and
    the abbreviation lists in `normalize.py`, are fine. They are code, not
    retrieved data.
- **Final model must be MIT or Apache 2.0 licensed and 8 billion parameters or
  fewer.** Check the licence before adding any pretrained model.
- Multiple accounts, one person on two teams, sharing code or predictions between
  teams.
- Simultaneous logins on the same participant account.

**Keep the GitHub repo private.** A public repo during a live competition looks
like code sharing.

## Output format traps

These cause rejection, and a rejected upload still costs one of the five.

- Every Source 1 record in the test set needs **exactly one row**, including the
  French ones and including records with no matches.
- Records with no matches get an **empty** `matched_entity_ids`, not a missing
  row.
- **Tab separated**, with exactly the column names
  `source1_entity_id` and `matched_entity_ids`. The ID list inside a cell is
  comma separated with no quoting. That is why the files are TSV: addresses and
  ID lists both contain commas.
- No duplicate IDs inside one list, and no duplicate `source1_entity_id` rows.
- Only Source 2 and Source 3 IDs, never Source 1, and only IDs that exist in the
  test set.
- Matches must be a **subset of the candidates** in `candidate_pairs.tsv`. The
  validator warns when they are not, and it means a pipeline bug.

## Reading the data

Always `sep="\t"`:

```python
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Without it pandas silently returns one column containing the whole line.

## Traps we already hit

1. **Validation that lies.** Our first harness scored a small held-out set
   against the full 10 million candidates and produced a meaningless number. See
   the Method notes in `experiments/experiments.md`. Do not undo that design.
2. **Memory.** Two full runs died. Millions of pairs as Python strings will
   exhaust a 16 GB machine. Use `encode_ids()` and `write_id_lists_coded()`, and
   keep `CANDIDATE_CHUNK` small.
3. **The France trap.** Training has only US and India. The test set is 15%
   France. Anything that hard-codes, filters, or one-hot encodes the country set
   will fail on that 15%. Cheap check: train on US only, validate on India only.
   If the score collapses, the pipeline has learned country specific habits.
