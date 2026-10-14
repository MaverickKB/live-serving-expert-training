"""Two-process CPU simulation of tensor-parallel added-neuron training (torch.distributed gloo).

Two OS processes act as the two TP ranks. Each holds:
- the same frozen toy model (a stack of residual blocks),
- HALF of the vocabulary head (rows split by rank, like a TP classifier shard),
- HALF of the added shared-expert neurons (width split by rank, like a TP expert).

Training uses the same functions as the live trainer: `ReplicatedInput` and `GatherBranches`
for the split added block, `replay_suffix` through the frozen layers, `streamed_answer_loss`
over the sharded vocabulary, and `accumulated_update` with a global gradient clip across ranks.
Each rank fingerprints its frozen tensors before and after.

It is a simulation of the math and the collective order, not of a serving engine
(see tensor-parallel-side-jobs for the live-serving coordination).

Run:  python examples/two_rank_training_sim.py
"""

import socket
import sys
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from resident_trainer.fit import AddedSharedNeurons
from resident_trainer.thin import (
    GatherBranches,
    ReplicatedInput,
    TensorHead,
    accumulated_update,
    replay_suffix,
    streamed_answer_loss,
)
from resident_trainer.thin_job import fingerprint

HIDDEN, VOCAB, WIDTH, LAYERS, MIDDLE = 32, 101, 8, 6, 3
PROMPTS, ANSWERS = [3, 17, 42, 77], [9, 55, 12, 90]


class GlooComm:
    """The `all_gather(flat_input, flat_output)` surface the trainer expects."""

    def all_gather(self, inputs, output):
        parts = [torch.empty_like(inputs) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, inputs.contiguous())
        output.copy_(torch.cat(parts))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return x + torch.tanh(self.proj(x))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def rank_main(rank, port, steps, results):
    store = dist.TCPStore('127.0.0.1', port, 2, rank == 0, timeout=timedelta(seconds=30))
    dist.init_process_group('gloo', store=store, rank=rank, world_size=2)
    try:
        torch.manual_seed(0)                       # identical frozen model on both ranks
        blocks = nn.ModuleList(Block() for _ in range(LAYERS)).requires_grad_(False)
        embed = torch.randn(VOCAB, HIDDEN)
        unembed = torch.randn(VOCAB, HIDDEN) / HIDDEN ** 0.5
        half = (VOCAB + 1) // 2
        start, stop = (0, half) if rank == 0 else (half, VOCAB)
        head = TensorHead(unembed[start:stop], offset=start)
        frozen = list(blocks.named_parameters()) + [('embed', embed), ('unembed', unembed)]
        before = fingerprint(frozen)

        torch.manual_seed(100 + rank)              # each rank owns its half of the added width
        added = AddedSharedNeurons(HIDDEN, WIDTH // 2)
        optimizer = torch.optim.Adam(added.parameters(), lr=3e-2)
        weights = SimpleNamespace(rank=rank, world=2, device='cpu', comm=GlooComm())
        prompts, answers = torch.tensor(PROMPTS), torch.tensor(ANSWERS)

        def prefix(tokens):
            x = embed[tokens]
            with torch.no_grad():
                for block in blocks[:MIDDLE]:
                    x = block(x)
            return x

        def with_added(x):
            branch = ReplicatedInput.apply(x, weights)
            return x + GatherBranches.apply(added(branch), weights).sum(0)

        def loss_for(index):
            x = with_added(prefix(prompts[index:index + 1]))
            x = replay_suffix(x, blocks[MIDDLE:])
            return streamed_answer_loss(x, answers[index:index + 1], head, 16, weights)

        def correct():
            with torch.no_grad():
                x = with_added(prefix(prompts))
                for block in blocks[MIDDLE:]:
                    x = block(x)
                return int(((x @ unembed.T).argmax(1) == answers).sum())

        start_correct = correct()
        losses = []
        for _ in range(steps):
            report = accumulated_update(added, optimizer, list(range(len(PROMPTS))), loss_for,
                                        weights=weights)
            losses.append(round(report['loss'], 6))
        result = {'rank': rank, 'correct_before': start_correct, 'correct_after': correct(),
                  'total': len(PROMPTS), 'final_loss': losses[-1],
                  'frozen_unchanged': fingerprint(frozen) == before, 'losses': losses}
    finally:
        dist.destroy_process_group()
    results.put(result)


def run(steps=60):
    port = free_port()
    context = mp.get_context('spawn')
    results = context.Queue()
    procs = [context.Process(target=rank_main, args=(r, port, steps, results)) for r in range(2)]
    for p in procs:
        p.start()
    try:
        out = sorted((results.get(timeout=120) for _ in procs), key=lambda r: r['rank'])
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
    return out


if __name__ == '__main__':
    for result in run():
        print({k: v for k, v in result.items() if k != 'losses'})
    sys.stdout.flush()
