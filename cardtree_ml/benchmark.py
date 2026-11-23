from __future__ import annotations

import random
from dataclasses import dataclass

from .game import TrainingRound
from .learner import QLearner
from .punishment import PUNISHMENT_RULES, PunishmentRule

REPEATS = 5
MAX_ROUNDS_PER_RUN = 5000
EPISODES_PER_UPDATE = 20
BASE_SEED = 738_491


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    formula: str
    completion_iterations: tuple[int, ...]
    repeats: int

    @property
    def completed_runs(self) -> int:
        return len(self.completion_iterations)


class PunishmentBenchmark:
    """Compares penalty rules without changing the app's saved learner."""

    def __init__(
        self,
        rules: tuple[PunishmentRule, ...] = PUNISHMENT_RULES,
        repeats: int = REPEATS,
        max_rounds_per_run: int = MAX_ROUNDS_PER_RUN,
    ) -> None:
        self.rules = rules
        self.repeats = repeats
        self.max_rounds_per_run = max_rounds_per_run
        self.candidate_index = 0
        self.run_index = 0
        self.rounds_in_run = 0
        self.learner = QLearner()
        self.iterations_by_rule: list[list[int]] = [[] for _ in rules]

    @property
    def done(self) -> bool:
        return self.candidate_index >= len(self.rules)

    @property
    def results(self) -> tuple[BenchmarkResult, ...]:
        if not self.done:
            raise RuntimeError("benchmark results are not ready yet")
        return tuple(
            BenchmarkResult(rule.name, rule.formula, tuple(iterations), self.repeats)
            for rule, iterations in zip(self.rules, self.iterations_by_rule)
        )

    @property
    def progress(self) -> str:
        if self.done:
            return "Benchmark complete"
        rule = self.rules[self.candidate_index]
        return (
            f"Benchmarking {rule.name} "
            f"({self.candidate_index + 1}/{len(self.rules)}), run "
            f"{self.run_index + 1}/{self.repeats}; "
            f"round {self.rounds_in_run + 1}/{self.max_rounds_per_run}."
        )

    def advance(self, episodes: int = EPISODES_PER_UPDATE) -> bool:
        """Run a small batch of complete rounds; return whether all are done."""
        for _ in range(episodes):
            if self.done:
                break
            self._run_episode()
        return self.done

    def _run_episode(self) -> None:
        rule = self.rules[self.candidate_index]
        seed = BASE_SEED + self.run_index * 100_003 + self.rounds_in_run * 997
        round_ = TrainingRound(
            self.learner,
            rule.function,
            rng=random.Random(seed),
        )
        while not round_.finished:
            round_.attempt_next()
        self.rounds_in_run += 1

        if self.learner.completion_iteration is not None:
            self.iterations_by_rule[self.candidate_index].append(
                self.learner.completion_iteration
            )
            self._next_run()
        elif self.rounds_in_run >= self.max_rounds_per_run:
            self._next_run()

    def _next_run(self) -> None:
        self.run_index += 1
        self.rounds_in_run = 0
        self.learner = QLearner()
        if self.run_index >= self.repeats:
            self.candidate_index += 1
            self.run_index = 0
