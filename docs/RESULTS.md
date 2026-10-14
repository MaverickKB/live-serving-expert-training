# Results, limits and replication wanted

## Setup
GLM-5.3-Flash (MoE) in EXL3 quantisation, tensor-parallel across two machines, TensorFold-derived
engine. Added block at layer 35. One private fact set.

## Measured (pocket off)

| Run | Dev strict (/24) | Rewordings (/24) | Sibling leaks | Notes |
|---|---|---|---|---|
| Baseline (no added block) | 0 | 0 | n/a | |
| Best expert | **16** | **11** | 2 | controls unchanged (14/14) |
| Longer contrast training | 14 | 8 | 2 | worse |
| Gain scaled to 0.7 | 16 | 10 | 2 | leaks unchanged |
| Gain scaled to 0.5 | 15 | 7 | 2 | leaks unchanged |

The frozen-weight fingerprint never changed across jobs. Chats were served between microbatches.
Contrast training with "unknown" and cross-pocket examples (140 and 160 examples) also left the
leaks in place.

## Limits
- One model, one engine, one cluster shape, one small private fact set.
- No seed variance reported; no independent replication.
- Rewordings lag the trained phrasing (11 vs 16): limited generalisation.
- Sibling-value leaks are unsolved. The added block has no gate, so it cannot decide when not to
  fire. Exact, easily confused facts belong in retrieval memory, not weights.
- The engine glue (`src/resident_trainer/engine/`) is reference-only.

## Replication wanted
We would like to hear from anyone who tries this on:
- another MoE (or dense) model, another engine (for example a common open serving engine), or
  another world size;
- a public fact set, so numbers can be compared;
- several seeds, with variance;
- a gated variant of the added block aimed at the sibling-leak problem.

Please report baselines, pocket-off scores, controls, leaks and whether frozen weights stayed
unchanged, including runs that failed.
