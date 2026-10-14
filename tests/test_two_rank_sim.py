"""Two-process gloo simulation of TP added-neuron training, end to end on CPU."""
import importlib
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


def test_two_ranks_learn_identically_and_keep_the_base_frozen():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    sim = importlib.import_module('two_rank_training_sim')
    zero, one = sim.run(steps=40)
    assert zero['losses'] == one['losses']          # both ranks see the same global loss
    assert zero['frozen_unchanged'] and one['frozen_unchanged']
    assert zero['correct_after'] > zero['correct_before'] == 0
