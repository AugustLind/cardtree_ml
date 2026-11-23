from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .learner import Experience, QLearner
from .tree import BinarySearchTree, Placement, RANKS

PenaltyFunction = Callable[[Placement, Placement], float]


@dataclass(frozen=True)
class AttemptResult:
    rank: int
    attempted: Placement
    answer: Placement
    correct: bool
    penalty: float
    attempt_number: int


class TrainingRound:
    def __init__(
        self,
        learner: QLearner,
        penalty_function: PenaltyFunction,
        rng: random.Random | None = None,
    ) -> None:
        self.learner = learner
        self.penalty_function = penalty_function
        self.rng = rng or random.Random()
        self.tree = BinarySearchTree()
        self.deck = list(RANKS)
        self.rng.shuffle(self.deck)
        self.seed_count = self.rng.randint(4, 10)
        self.seed_cards = self.deck[: self.seed_count]
        self.cards_to_place = self.deck[self.seed_count :]
        for rank in self.seed_cards:
            self.tree.insert(rank, self.tree.correct_placement(rank))
        self.attempts = 0
        self.correct_count = 0
        self.total_penalty = 0.0

    @property
    def finished(self) -> bool:
        return self.attempts >= len(self.cards_to_place)

    @property
    def perfect(self) -> bool:
        return self.finished and self.correct_count == len(self.cards_to_place)

    def attempt_next(self) -> AttemptResult:
        if self.finished:
            raise StopIteration("this round has ended")

        rank = self.cards_to_place[self.attempts]
        answer = self.tree.correct_placement(rank)
        attempted, experience = self._choose_placement(rank)
        correct = attempted == answer

        penalty = 0.0
        if correct:
            reward = 0.2
            self.correct_count += 1
        else:
            penalty = float(self.penalty_function(attempted, answer))
            if not math.isfinite(penalty) or penalty < 0:
                raise ValueError("punishment must return a finite, nonnegative number")
            reward = -penalty
            self.total_penalty += penalty

        self.learner.learn(experience, reward)
        if correct:
            self.tree.insert(rank, attempted)
        self.attempts += 1
        if self.finished:
            self.learner.record_round_result(self.perfect)
        return AttemptResult(
            rank, attempted, answer, correct, penalty, self.attempts
        )

    def _choose_placement(self, rank: int) -> tuple[Placement, list[Experience]]:
        path: list[str] = []
        experience: list[Experience] = []
        node = self.tree.root
        while node is not None:
            action = self.learner.choose(rank, node.rank, self.rng)
            experience.append(((rank, node.rank), action))
            path.append(action)
            node = node.left if action == "L" else node.right
        return tuple(path), experience
