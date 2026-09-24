# Start here

These files exist so any of us can work with our own AI chatbot and pick up
exactly where the others left off. Whatever assistant you use, it starts each
chat knowing nothing about this project. Pasting the right file at the start of
a chat gives it the context in one go, so you do not have to explain the problem
from scratch and it does not invent things that are wrong.

## How to use them

1. Open a new chat with whatever assistant you use.
2. Paste the whole of `01_PROJECT_CONTEXT.md` first. That is the shared
   background: the problem, the scoring, what we measured, what is already built.
3. Then paste the file for the phase you claimed, for example
   `02_PHASE2_BLOCKING.md`.
4. Then ask your question normally.

If your assistant has a limit and the paste is too long, send
`01_PROJECT_CONTEXT.md` in two messages and tell it to wait before answering.

## What NOT to paste

- **Never paste the dataset, or rows from it.** Not train, not test, not the
  ground truth. The rules forbid taking our data outside the challenge, and the
  files are gigabytes anyway. Describe the columns instead. The context file
  already does this for you.
- No login details, no portal links, no account information.

## Before you trust any code it writes

Assistants are confident even when wrong, and on this project two bugs cost us
whole runs. So:

1. **Score it.** Run `run_baseline.py --validate` and compare the F0.5 against
   what is in `experiments/experiments.md`. If a change does not move that
   number, it did not help.
2. **Do not let it rewrite the validation harness.** It looks odd on purpose.
   `01_PROJECT_CONTEXT.md` explains why, and there is a "do not undo this" note
   in the experiment log.
3. **Watch memory.** This dataset is big enough to freeze a laptop. The memory
   rules are in the README section 9 and repeated in the context file.
4. **Log it** in `experiments/experiments.md`, including things that failed. An
   unlogged run does not exist.

## Files here

| File | Paste this when |
|---|---|
| `01_PROJECT_CONTEXT.md` | Always, at the start of every new chat |
| `02_PHASE2_BLOCKING.md` | You are working on candidate generation |
| `03_PHASE3_FEATURES.md` | You are working on pair features |
| `04_PHASE4_MODEL.md` | You are training the classifier |
| `05_CODE_MAP.md` | You need the exact function names and signatures |
| `06_RULES_AND_TRAPS.md` | Anything about submissions, rules, or deadlines |
