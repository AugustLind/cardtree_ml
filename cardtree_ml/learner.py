from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable

from .tree import Placement

State = tuple[int, int]  # (card rank, rank at the tree node being compared)
Experience = tuple[State, str]
PERFECT_STREAK_TARGET = 3


class QLearner:
    """Tabular Q-learning over the robot's local rank comparisons."""

    def __init__(
        self,
        values: dict[str, dict[str, float]] | None = None,
        epsilon: float = 0.25,
        iterations: int = 0,
        consecutive_perfect_rounds: int = 0,
        completion_iteration: int | None = None,
    ) -> None:
        self.values = values or {}
        self.epsilon = epsilon
        self.iterations = iterations
        self.consecutive_perfect_rounds = consecutive_perfect_rounds
        self.completion_iteration = completion_iteration
        self.learning_rate = 0.2
        self.discount = 0.9
        self.minimum_epsilon = 0.05

    @staticmethod
    def _key(state: State) -> str:
        return f"{state[0]}:{state[1]}"

    def choose(self, card_rank: int, node_rank: int, rng: random.Random) -> str:
        key = self._key((card_rank, node_rank))
        actions = self.values.get(key, {})
        if rng.random() < self.epsilon:
            return rng.choice(("L", "R"))
        left, right = actions.get("L", 0.0), actions.get("R", 0.0)
        if left == right:
            return rng.choice(("L", "R"))
        return "L" if left > right else "R"

    def learn(self, experience: Iterable[Experience], reward: float) -> None:
        """Propagate the final placement reward back along the chosen route."""
        self.iterations += 1
        route = list(experience)
        future_return = reward
        for state, action in reversed(route):
            key = self._key(state)
            actions = self.values.setdefault(key, {"L": 0.0, "R": 0.0})
            old_value = actions.get(action, 0.0)
            actions[action] = old_value + self.learning_rate * (
                future_return - old_value
            )
            future_return *= self.discount
        self.epsilon = max(self.minimum_epsilon, self.epsilon * 0.995)

    def record_round_result(self, perfect: bool) -> None:
        if perfect:
            self.consecutive_perfect_rounds = min(
                PERFECT_STREAK_TARGET, self.consecutive_perfect_rounds + 1
            )
            if (
                self.consecutive_perfect_rounds == PERFECT_STREAK_TARGET
                and self.completion_iteration is None
            ):
                self.completion_iteration = self.iterations
        else:
            self.consecutive_perfect_rounds = 0


class LearnerStore:
    """Persists the learned policy under the user's local data directory."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".cardtree_ml" / "policy.json"

    def load(self) -> QLearner:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return QLearner()
        if data.get("version") != 1 or not isinstance(data.get("values"), dict):
            raise ValueError(f"Unsupported or invalid learner policy: {self.path}")
        values: dict[str, dict[str, float]] = {}
        for state, actions in data["values"].items():
            if not isinstance(actions, dict):
                raise ValueError(f"Invalid learner policy state: {state}")
            values[state] = {
                action: float(value)
                for action, value in actions.items()
                if action in ("L", "R")
            }
        return QLearner(
            values,
            float(data.get("epsilon", 0.25)),
            int(data.get("iterations", 0)),
            int(data.get("consecutive_perfect_rounds", 0)),
            (
                int(data["completion_iteration"])
                if data.get("completion_iteration") is not None
                else None
            ),
        )

    def save(self, learner: QLearner) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "epsilon": learner.epsilon,
            "iterations": learner.iterations,
            "consecutive_perfect_rounds": learner.consecutive_perfect_rounds,
            "completion_iteration": learner.completion_iteration,
            "values": learner.values,
        }
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_name = temp_file.name
                json.dump(data, temp_file, indent=2, sort_keys=True)
                temp_file.write("\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_name, self.path)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)
