# Origin story (short)

The full chronological account, with beliefs, evidence, decisions and results at each step,
lives in one master document: [moe-live-learning-tools/docs/ORIGIN_STORY.md](https://github.com/MaverickKB/moe-live-learning-tools/blob/main/docs/ORIGIN_STORY.md).
Project-wide research lessons: [moe-live-learning-tools/docs/LESSONS.md](https://github.com/MaverickKB/moe-live-learning-tools/blob/main/docs/LESSONS.md).

## How this repository fits in the story

1. **Belief:** extra neurons at the last layer, or closed-form weight edits, could store
   facts. **Result:** last-layer pockets 3/20 (coding) and 7/24 (simple facts) vs 16/20 and
   24/24 with the docs in the prompt; MEMIT-style edits 0/12; derivative-free fitting 0/20.
2. **Evidence that redirected us:** attention-readable memory trained against the model's own
   docs-in-prompt behaviour worked (CPU prototype 24/24; native 96 rows 34/36).
3. **Decision:** use that memory as the teacher and distil it into one added block on the
   always-on shared expert at a middle layer, with real gradients and frozen base weights.
   An earlier attempt with an approximate backward failed (0/12; measured derivative
   disagreement), and a second model load for training did not fit next to serving.
4. **Decision:** train inside the serving process. That needed the paired sealed start,
   nested serving between microbatches, streamed full-vocabulary loss, a free-memory floor
   with trainer-only abort, and a frozen-weight fingerprint around every job.
5. **Result:** 0/24 to 16/24 dev strict, 11/24 rewordings, pocket off; 14/14 controls
   unchanged; fingerprint unchanged; two sibling leaks unsolved (see `docs/RESULTS.md`).

Dates, sources and unverified items are marked in the master document. One setup, no
independent replication; treat this as a starting point.
