"""User-owned penalty rule for incorrect placements.

Replace the function body with a nonnegative penalty calculation. Placements
are tuples of directions from the root, for example (), ("L",), or ("R", "L").
"""

from dataclasses import dataclass
from typing import Callable

from .tree import Placement


def _shared_prefix_length(attempted: Placement, answer: Placement) -> int:
    shared = 0
    for attempted_step, answer_step in zip(attempted, answer):
        if attempted_step != answer_step:
            break
        shared += 1
    return shared


def _tree_distance(attempted: Placement, answer: Placement) -> int:
    shared = _shared_prefix_length(attempted, answer)
    return len(attempted) + len(answer) - 2 * shared


def punishment(attempted: Placement, answer: Placement) -> float:
    """Penalize more when the robot takes the wrong branch near the root."""
    if attempted == answer:
        return 0.0
    return 1.0 / (_shared_prefix_length(attempted, answer) + 1)


def _constant(attempted: Placement, answer: Placement) -> float:
    return 1.0


def _root_exponential(attempted: Placement, answer: Placement) -> float:
    depth = _shared_prefix_length(attempted, answer)
    return 0.5**depth


def _root_linear(attempted: Placement, answer: Placement) -> float:
    depth = _shared_prefix_length(attempted, answer)
    return max(0.1, 1.0 - 0.1 * depth)


def _inverse_square(attempted: Placement, answer: Placement) -> float:
    depth = _shared_prefix_length(attempted, answer)
    return 1.0 / ((depth + 1) ** 2)


def _depth_linear(attempted: Placement, answer: Placement) -> float:
    return float(_shared_prefix_length(attempted, answer) + 1)


def _distance(attempted: Placement, answer: Placement) -> float:
    return float(_tree_distance(attempted, answer))


def _distance_squared(attempted: Placement, answer: Placement) -> float:
    return float(_tree_distance(attempted, answer) ** 2)


def _distance_root_weighted(attempted: Placement, answer: Placement) -> float:
    depth = _shared_prefix_length(attempted, answer)
    return _tree_distance(attempted, answer) / (depth + 1)


@dataclass(frozen=True)
class PunishmentRule:
    name: str
    formula: str
    function: Callable[[Placement, Placement], float]


PUNISHMENT_RULES = (
    PunishmentRule("Current: inverse divergence depth", "1 / (d + 1)", punishment),
    PunishmentRule("Constant penalty", "1", _constant),
    PunishmentRule("Root focused: exponential", "2^(-d)", _root_exponential),
    PunishmentRule("Root focused: linear", "max(0.1, 1 - 0.1d)", _root_linear),
    PunishmentRule("Root focused: inverse square", "1 / (d + 1)^2", _inverse_square),
    PunishmentRule("Deeper mistakes cost more", "d + 1", _depth_linear),
    PunishmentRule("Tree edge distance", "D", _distance),
    PunishmentRule("Squared tree distance", "D^2", _distance_squared),
    PunishmentRule("Distance with root weighting", "D / (d + 1)", _distance_root_weighted),
)
