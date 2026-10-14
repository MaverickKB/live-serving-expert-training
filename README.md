# live-serving-expert-training

Train a small set of **added shared-expert neurons** at one middle layer of a large
mixture-of-experts model **inside the live serving process**, reusing the weights the
server already has loaded, with no model downtime.

> **Teaching example, not a product.** Adapt the ideas; do not copy it 1:1.
>
> **About this work.** This is an honest, ongoing investigation into what is possible, not a
> claim of perfection. We make mistakes and backfill transparency as we go, in the hope that
> others benefit and can skip the hard lessons we hit.

> **Status: reference extract, not a maintained product.** This code was pulled out of a
> research project so that people (and their coding agents) can read it, learn from it and
> adapt it. It ran on one specific setup. It has not been run anywhere else. Expect to
> rewrite the engine integration for your own stack.

## What it does

- The base model stays frozen. Training touches only an added gate/up/down block
  (`fit.AddedSharedNeurons`) whose output is added to the residual stream at one layer.
- The loss is exact full-vocabulary cross-entropy, computed in vocabulary tiles over the
  server's existing classifier storage (`thin.StreamedCrossEntropy`), and the frozen
  suffix of layers is replayed one layer at a time in the backward pass
  (`thin.LayerReplay`) so no full activation graph is kept.
- Jobs are small (a few microbatches) and are scheduled between chat requests. Between
  microbatches, both ranks pause together and serve any waiting chats.

## How it fits into a live tensor-parallel server

| Concern | Mechanism | Where |
|---|---|---|
| Both ranks start the same job at the same serving message | Paired sealed start: the head writes a `THIN_START` op into the sealed serving sequence; the worker confirms the message number on a trainer-only channel before anything runs. If the worker does not join in time, only the job is cancelled. | `thin_job.lead`, `thin_job.join`, `thin_job.attach` |
| Trainer traffic never mixes with serving collectives | Trainer-only channel over the rendezvous key-value store, every record tagged with job, sequence, name and rank | `thin_job.TrainerChannel` |
| Serving chats during a job | Both ranks agree at each boundary whether to serve; the head closes the nested iteration with an explicit `THIN_RESUME` marker | `thin_job.serve_waiting`, `thin_job.resume` |
| No stall after a job | Worker skips its doorbell wait if the job's last resume left chats live | `attach.await_bell` |
| Memory safety | Free-memory floor (default 1 GB) and private workspace ceiling sampled continuously; abort decisions are exchanged so both ranks abort the **trainer only** at the same boundary | `thin_budget.ThinPassBudget` |
| Proof the base never changes | Device-side fingerprint of frozen tensors before and after every job; a mismatch fails the job | `thin_job.fingerprint`, `thin_job.frozen_sample` |
| Crash recovery | One exclusive lock owns any restart of the pair; re-running recovery on a healthy pair is a no-op | `pair_recovery` |
| Code changes without a restart | Both ranks hash the staged files, agree, then execute exactly the hashed bytes into the existing module objects | `hotload` |

## Results, limits and replication

See `docs/RESULTS.md`. In short, on one two-machine setup: 0/24 to 16/24 development facts
(strict) and 11/24 rewordings with the retrieval pocket off; 14 controls unchanged; frozen
weights unchanged across every job. Two near-miss facts leaked a sibling's value and three
fixes failed. **Replication on other hardware and models is wanted** (see `docs/RESULTS.md`).

## Documentation

| Doc | What |
|---|---|
| `docs/ARCHITECTURE.md` | Walkthrough of one job across both ranks |
| `docs/PORTING.md` | Engine porting guide |
| `docs/EVALUATION.md` | Evaluation protocol (pocket-off, controls, held-out) |
| `docs/RESULTS.md` | Measured results, limits, replication wanted |
| `LESSONS.md` | Failures, causes and fixes |
| `AGENTS.md` | Map, invariants, adaptation steps for coding agents |

## Run the CPU tests and toy example

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'
pytest                      # CPU only, no GPU or server needed
python examples/toy_train.py              # single process
python examples/two_rank_training_sim.py  # two processes, torch.distributed gloo, CPU
```

The two-rank simulation splits the vocabulary head and the added neurons across two gloo
processes, trains with the same functions as the live trainer, and checks that both ranks see
the same losses and that the frozen fingerprint is unchanged.

The tests cover the streamed loss and layer replay against plain autograd, the paired
start and cancellation protocol with fake decoders and stores, the memory budget and
paired aborts, the recovery lock, hot-load agreement and refusal, and offline recall
scoring. The toy trains added neurons on a frozen 6-layer model and checks that the
frozen fingerprint is unchanged.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RESIDENT_TRAINER_AUTHORITY` | empty (nothing runs) | Exact string a job must carry to run |
| `RESIDENT_TRAINER_RECEIPTS` | `./receipts` | Per-rank receipts, phase logs, stack dumps |
| `RESIDENT_TRAINER_CHECKPOINTS` | `./checkpoints` | Per-rank added-neuron checkpoints |
| `RESIDENT_TRAINER_STATE_DIR` | `./state` | Recovery lock location |
| `RESIDENT_TRAINER_MASTER_PORT` | `29551` | Recorded in the recovery lock |
| `RESIDENT_TRAINER_LAYER` / `_WATCH_LAYER` | `35` / `44` | Trained layer; extra layer included in the frozen hash |

A `STOP` file in the job root aborts a running job at its next boundary.

## Related

The live-serving coordination (paired start, resume marker, doorbell skip, hot-load) is covered
in depth with diagrams and a two-process demo in the companion repository
`tensor-parallel-side-jobs` (URL to be filled in at publish time).

## Files

See `AGENTS.md` for a map of the code, the invariants and how to adapt it.
