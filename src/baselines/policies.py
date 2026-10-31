"""Baseline policy names and action-space helpers."""
from itertools import combinations_with_replacement


def enumerate_actions(candidate_ids, subtasks=2):
    """Unordered destination tuples with repetition, matching the report."""
    return list(combinations_with_replacement(candidate_ids, subtasks))


def encode_action(pair, candidate_ids):
    """Encode an unordered pair using its index in the canonical action list."""
    actions = enumerate_actions(candidate_ids, len(pair))
    normalized = tuple(sorted(pair, key=lambda x: candidate_ids.index(x)))
    return actions.index(normalized)


def decode_action(index, candidate_ids, subtasks=2):
    actions = enumerate_actions(candidate_ids, subtasks)
    if not 0 <= index < len(actions):
        raise IndexError("Action index outside the action space")
    return actions[index]
