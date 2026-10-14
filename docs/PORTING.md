# Engine porting guide

Port in this order. Each step has a check you can run before moving on.

## 1. Run the math on CPU
`pytest`, then `examples/toy_train.py` and `examples/two_rank_training_sim.py`. These use the
same autograd functions as the live trainer.

## 2. Choose the training site
Pick a middle layer. Capture its residual input for your training prompts (only that layer;
capturing many layers at once tripped a memory guard at the origin). Set
`RESIDENT_TRAINER_LAYER`. Update the allowed-layer checks in `thin_native.py` and `answer.py`.

## 3. Head and suffix adapters
- **Head:** an object with `rows`, `hidden`, `offset`, `logits(inputs, start, stop)` and
  `input_vjp(inputs, gradient, start, stop)` over your frozen classifier shard. `thin.TensorHead`
  is the reference; `thin_native.ResidentHead` is the origin's view over quantised storage.
- **Suffix:** callables for each frozen layer after the site, usable in forward and in
  `torch.autograd.grad` (the replay rebuilds one layer's graph at a time). For quantised
  kernels you need input VJPs (`engine/resident_ops.py` shows the origin's).
- **Check:** run `probe` mode and require eager-versus-replay gradient agreement.

## 4. Residual streams and world size
The origin model has four residual streams and two ranks. Search for `streams=4`, `world=2`
and `hidden * 4` and adapt. `shard_added_checkpoint` assumes both.

## 5. Coordination
Wire the paired start, nested serving, resume marker and doorbell skip into your scheduler.
Follow `tensor-parallel-side-jobs` (its porting guide lists the exact decoder surface).

## 6. Safety
- Fingerprint the right tensors (`frozen_sample(layers=...)`): include the large quantised
  weights near the site.
- Tune `FLOOR_BYTES` and `WORKSPACE_BYTES` per host.
- Set `RESIDENT_TRAINER_AUTHORITY`; leave it empty to make all jobs refuse.
- Take the recovery lock in every restart path.

## 7. First live run
A `probe` job while idle, then a one-microbatch `update` while sending chats, then compare
controls before and after. Keep rollback checkpoints.
