# Architecture walkthrough

What happens during one training job, from the moment a job is admitted to the moment both
ranks are back to plain serving. Origin setup: two machines, one tensor-parallel rank each,
an MoE model already loaded for serving.

## 0. What is trained

`fit.AddedSharedNeurons` is a small gate/up/down block (SwiGLU, clamped) attached to the
residual stream at one middle layer (`RESIDENT_TRAINER_LAYER`, 35 at the origin). Its width is
split across the two ranks like a tensor-parallel expert; each rank holds half the neurons and
their Adam moments. `down` starts at zero, so a fresh block changes nothing. Nothing else has
`requires_grad`.

## 1. Admission

A job record (sealed JSON, hashed) names the mode (`probe` or `update`), the number of
microbatches, a deadline and the operator authority string. The request that carries it wakes
the scheduler. The head waits until no chats are live (`attach.finish`).

## 2. Paired sealed start

The head emits `THIN_START` with the job identity into the sealed serving stream, then `IDLE`,
and flushes. The worker sees it at the same message number. Both put `join` records on the
trainer channel (rendezvous store). The head checks the worker's message number and readiness:
`go` or `cancel`. A cancel affects only the job. (Diagrams: companion repo
`tensor-parallel-side-jobs`.)

## 3. Inside the job (`thin_job.run`)

1. `ThinPassBudget` admits the pass only if host free memory exceeds the floor (1 GB) plus the
   private workspace (2 GiB), the deadline is ahead and the authority still matches.
2. `frozen_sample` + `fingerprint` hash the frozen tensors near the training site.
3. The added block and optimizer are loaded once from a named checkpoint and kept resident.
4. For each microbatch: load the captured layer-input rows for the example, run the added block,
   replay the frozen suffix layer by layer (`LayerReplay`), compute exact full-vocabulary
   cross-entropy in tiles over the existing classifier shard (`StreamedCrossEntropy`),
   backpropagate to the added block only.
5. Between microbatches `guard()` runs: budget check on both ranks (flags exchanged on the
   trainer channel), then both ranks agree whether chats are waiting; if so, one nested serving
   iteration runs and ends with `THIN_RESUME`.
6. Gradients are clipped globally across ranks; Adam steps on each rank's half.
7. A per-rank checkpoint is written; the frozen fingerprint is taken again and must match.
8. A per-rank receipt records phases, memory, losses, hashes and how many chats were served.

`probe` mode instead checks that eager and replay gradients agree (relative error < 1e-3)
before any update is trusted.

## 4. After the job

`_thin_running` clears. If chats are still live, the worker skips its idle doorbell (the head
is not idle and will not ring). Serving continues as normal.

## 5. Failure paths

- Budget breach, deadline, revoked authority, or a garbage flag word on any rank:
  `PassAborted` on both ranks at the same boundary; the pending update is discarded.
- A frozen-hash mismatch: the job fails loudly.
- Serving crash: the single-owner recovery lock (`pair_recovery`) owns any restart.

## 6. Changing code

Stage modules can be hot-loaded through a sealed `hotload` job. Restart-only files (hook
registration, bootstrap, the autograd and budget modules, settings) need a planned restart.
