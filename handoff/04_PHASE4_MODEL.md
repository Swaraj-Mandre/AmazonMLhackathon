# Phase 4: the matching classifier

Paste `01_PROJECT_CONTEXT.md` first, then this.

## The job

Replace the single cosine threshold with a trained classifier over the phase 3
features, then decide the cut-off.

## Training data

- **Positives:** candidate pairs that appear in `train_ground_truth.tsv`.
- **Negatives: the near misses our own blocking produced**, meaning candidate
  pairs that are not true matches. This matters a lot. Random pairs drawn from
  the dataset are trivially easy and teach the model nothing useful, because at
  prediction time the model only ever sees blocking output.
- Build both from the held-out split machinery already in `data_io.py`, so
  training and validation records never mix.

## Model

LightGBM, binary classification, is the sensible default. It handles mixed
feature scales, trains fast on CPU, and there is no GPU here. It is MIT licensed
and has no parameter count issue, so both challenge constraints are satisfied
without argument.

Start simple. Get a working model with default settings and a logged F0.5 before
tuning anything.

## Then the decision rules

Two separate decisions, both worth real score.

**1. The threshold.** Sweep the model probability against F0.5 on the held-out
split, exactly as `ACCEPT_THRESHOLDS` does now. Expect the best cut-off to sit
high, because F0.5 punishes a false match twice as hard as a miss. This sweep is
usually worth more than further model tuning.

**2. Exclusivity.** Every Source 2/3 record has at most one owner in the data, so
when two Source 1 records claim the same candidate we can keep only the stronger
claim. `apply_exclusivity()` already does this. Measured with the raw cosine it
is worth a lot at loose thresholds and slightly negative at tight ones: 0.3149 to
0.5234 at threshold 0.45, but 0.5659 down to 0.5583 at 0.92. **Re-measure it with
the model score**, because the useful operating point will move.

## Things to watch

- Precision counts double. A model with a great AUC can still score badly if the
  threshold is wrong, so always report F0.5, not AUC alone.
- Check the predicted singleton rate against the 5.58% seen in training. A wildly
  different number means something is off.
- Keep the feature order identical between training and prediction, and save it
  alongside the model.

## Definition of done

A trained model whose best swept F0.5 beats 0.5659 on the full held-out split,
logged in `experiments/experiments.md`, with the chosen threshold and the
exclusivity decision recorded.
