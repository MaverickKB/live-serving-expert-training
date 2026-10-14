# Why this exists and how it works

**Why.** Every attempt to teach the served model new facts outside its running process ran
into the same walls: a second model load did not fit next to serving, taking serving down
was not acceptable, and approaches that avoided real gradients (last-layer pockets,
derivative-free fitting, closed-form edits) did not move generated recall. Docs-in-prompt
answers were always far better, which suggested distilling the model's own docs-in-prompt
behaviour into a small added block with real gradients.

**How.**
- One added block of shared-expert neurons at a middle layer (layer 35 on the origin model);
  base weights are frozen and fingerprinted before and after each job.
- The trainer runs inside the server between scheduler iterations, reusing loaded weights.
- Both tensor-parallel ranks start the job at the same sealed message (`thin_job`), follow
  nested serving until an explicit resume op, and skip the post-job doorbell if chats are live.
- Loss is exact full-vocabulary cross-entropy computed in streamed tiles (`thin_native`).
- A free-memory floor aborts only the trainer, on all ranks together (`thin_budget`).
- Code changes are hot-loaded as exact hashed bytes (`hotload`).
- Recall is scored with the memory pocket off; sibling leaks are counted (`recall_scoring`).

Architecture: `ARCHITECTURE.md`. Porting: `PORTING.md`. Evaluation: `EVALUATION.md`.
Results and limits: `RESULTS.md`. Incident lessons: `../LESSONS.md`. Configuration is via
settings and environment variables (`src/resident_trainer/settings.py`), never hard-coded paths.
