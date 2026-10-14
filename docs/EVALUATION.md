# Evaluation protocol

The question is whether facts are in the **weights**, not whether the system can answer them.

## Sets
- **Development facts:** the facts trained on, asked in the trained phrasing (origin: 24).
- **Rewordings:** the same facts asked differently (origin: 24).
- **Near-miss pairs:** facts that differ from a sibling in one detail; check for leaks.
- **Controls:** unrelated questions whose answers must not change (origin: 14).
- **General chat:** generic prompts; the expert-on answer must not be worse than expert-off.
- **Held-out set:** sealed; never read during development.

## Rules
1. Score with the retrieval pocket **off**. Pocket-on scores measure retrieval.
2. **Strict** credit: the grader passes the answer *and* the expected value appears in it as a
   whole token (`recall_scoring.value_hit`).
3. Count **sibling leaks** on failed answers (`recall_scoring.sibling_leaks`).
4. Compare controls and general chat **expert on versus off**; a changed answer counts as a drop
   only if graded worse.
5. Check the **frozen fingerprint** before and after every job; any change invalidates the run.
6. Report the baseline (expert off) for every number.
7. Report every attempt, including failed ones.

## What not to do
- Do not tune on rewordings or the held-out set.
- Do not report the best of many seeds without saying so.
