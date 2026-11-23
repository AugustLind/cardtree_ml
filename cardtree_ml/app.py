from __future__ import annotations

import random
import statistics
import tkinter as tk
from tkinter import messagebox, ttk

from .benchmark import BenchmarkResult, PunishmentBenchmark
from .game import AttemptResult, TrainingRound
from .learner import PERFECT_STREAK_TARGET, LearnerStore
from .punishment import punishment
from .tree import BinarySearchTree, Node, Placement, rank_name

VISIBLE_STEP_DELAY_MS = 2000
BACKGROUND_ROUND_LIMIT = 5000


class CardTreeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Card Tree Learner")
        self.root.geometry("1000x760")
        self.store = LearnerStore()
        try:
            self.learner = self.store.load()
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot load learned policy", str(error))
            raise

        self.rng = random.Random()
        self.game = TrainingRound(self.learner, punishment, self.rng)
        self.running = True
        self.background_running = False
        self.benchmark_running = False
        self.benchmark: PunishmentBenchmark | None = None
        self.background_rounds = 0
        self.after_id: str | None = None
        self.last_result: AttemptResult | None = None

        self._build_view()
        self._schedule_step(1200)

    def _build_view(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Card Tree Learner",
            font=("TkDefaultFont", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Label(
            header, text=f"Attempt 0 of {len(self.game.cards_to_place)}"
        )
        self.progress.grid(row=0, column=1, padx=(20, 0))
        self.iteration_label = ttk.Label(header, text="Placement iterations: 0")
        self.iteration_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

        tree_frame = ttk.Frame(self.root, padding=(18, 0, 18, 12))
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            tree_frame,
            background="#fbfaf7",
            highlightthickness=1,
            highlightbackground="#d8d5ce",
            width=900,
            height=530,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.canvas.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(self.root, padding=(18, 0, 18, 14))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status = ttk.Label(
            footer,
            text=(
                f"{self.game.seed_count} cards are in the starter tree. "
                "The robot is choosing a placement…"
            ),
            wraplength=730,
        )
        self.status.grid(row=0, column=0, sticky="w")
        self.round_button = ttk.Button(
            footer, text="Start next round", command=self._start_next_round
        )
        self.round_button.grid(row=0, column=1, padx=(14, 0))
        self.round_button.state(["disabled"])
        self.train_button = ttk.Button(
            footer,
            text="Train until 3 perfect rounds",
            command=self._start_background_training,
        )
        self.train_button.grid(row=0, column=2, padx=(8, 0))
        self.train_button.state(["disabled"])
        self.benchmark_button = ttk.Button(
            footer,
            text="Benchmark punishments",
            command=self._start_benchmark,
        )
        self.benchmark_button.grid(row=0, column=3, padx=(8, 0))
        self.benchmark_button.state(["disabled"])
        self._draw_tree()

    def _schedule_step(self, delay: int = 850) -> None:
        if self.running and not self.game.finished:
            self.after_id = self.root.after(delay, self._step)

    def _step(self) -> None:
        self.after_id = None
        try:
            result = self.game.attempt_next()
            self.store.save(self.learner)
        except NotImplementedError as error:
            self.running = False
            self.status.configure(text=str(error))
            return
        except (OSError, ValueError) as error:
            self.running = False
            self.status.configure(text=f"Training paused: {error}")
            return

        self.last_result = result
        self.progress.configure(
            text=f"Attempt {result.attempt_number} of {len(self.game.cards_to_place)}"
        )
        self._update_iteration_label()
        if result.correct:
            self.status.configure(
                text=(
                    f"{rank_name(result.rank)} placed correctly at "
                    f"{self._format_path(result.attempted)}. The robot earned +0.2."
                )
            )
        else:
            self.status.configure(
                text=(
                    f"{rank_name(result.rank)} was placed at "
                    f"{self._format_path(result.attempted)}; the answer was "
                    f"{self._format_path(result.answer)}. Penalty: {result.penalty:g}. "
                    "The card was discarded."
                )
            )
        self._draw_tree(result)

        if self.game.finished:
            self.running = False
            self.round_button.state(["!disabled"])
            if self.learner.completion_iteration is None:
                self.train_button.state(["!disabled"])
            self.benchmark_button.state(["!disabled"])
            if self.learner.completion_iteration is not None:
                summary = (
                    f"Training complete: {PERFECT_STREAK_TARGET} perfect rounds in "
                    f"a row after {self.learner.completion_iteration} placement "
                    "iterations."
                )
            elif self.game.perfect:
                summary = (
                    "Perfect round! Streak: "
                    f"{self.learner.consecutive_perfect_rounds}/"
                    f"{PERFECT_STREAK_TARGET}."
                )
            else:
                summary = "This round had a mistake; the perfect-round streak reset."
            self.status.configure(
                text=(
                    f"Round complete: {self.game.correct_count} of "
                    f"{len(self.game.cards_to_place)} placements correct; total "
                    f"penalty {self.game.total_penalty:g}. {summary} "
                    "The learned policy has been saved."
                )
            )
        else:
            self._schedule_step(VISIBLE_STEP_DELAY_MS)

    def _start_next_round(self) -> None:
        self._show_next_visual_round(
            "New shuffled round. The robot is choosing a placement."
        )

    def _show_next_visual_round(self, message: str) -> None:
        self.game = TrainingRound(self.learner, punishment, self.rng)
        self.last_result = None
        self.running = True
        self.progress.configure(text=f"Attempt 0 of {len(self.game.cards_to_place)}")
        self._update_iteration_label()
        self.status.configure(text=f"{message} "
            f"{self.game.seed_count} cards are in the starter tree."
        )
        self.round_button.state(["disabled"])
        self.train_button.state(["disabled"])
        self.benchmark_button.state(["disabled"])
        self._draw_tree()
        self._schedule_step(1200)

    def _start_background_training(self) -> None:
        self.running = False
        self.background_running = True
        self.background_rounds = 0
        self.round_button.state(["disabled"])
        self.train_button.state(["disabled"])
        self.benchmark_button.state(["disabled"])
        self.status.configure(
            text=(
                f"Training without animation until {PERFECT_STREAK_TARGET} "
                "perfect rounds in a row are reached…"
            )
        )
        self.root.after(1, self._background_training_step)

    def _background_training_step(self) -> None:
        if not self.background_running:
            return
        try:
            training_round = TrainingRound(self.learner, punishment, self.rng)
            while not training_round.finished:
                training_round.attempt_next()
            self.background_rounds += 1
            self.store.save(self.learner)
        except NotImplementedError as error:
            self.background_running = False
            self.status.configure(text=str(error))
            self.round_button.state(["!disabled"])
            self.benchmark_button.state(["!disabled"])
            return
        except (OSError, ValueError) as error:
            self.background_running = False
            self.status.configure(text=f"Background training paused: {error}")
            self.round_button.state(["!disabled"])
            return

        self._update_iteration_label()
        if self.learner.completion_iteration is not None:
            completion_iteration = self.learner.completion_iteration
            self.background_running = False
            self._show_next_visual_round(
                f"Background training reached {PERFECT_STREAK_TARGET} perfect "
                f"rounds in a row after {self.background_rounds} rounds and "
                f"{completion_iteration} placement iterations. Showing the next round."
            )
        elif self.background_rounds >= BACKGROUND_ROUND_LIMIT:
            self.background_running = False
            self.status.configure(
                text=(
                    f"No {PERFECT_STREAK_TARGET}-round perfect streak after "
                    f"{self.background_rounds} background "
                    "tries. The current policy is saved; start a visual round or "
                    "try background training again."
                )
            )
            self.round_button.state(["!disabled"])
            self.train_button.state(["!disabled"])
            self.benchmark_button.state(["!disabled"])
        else:
            self.status.configure(
                text=(
                    f"Background training: {self.background_rounds} rounds, "
                    f"{self.learner.iterations} placement iterations."
                )
            )
            self.root.after(1, self._background_training_step)

    def _start_benchmark(self) -> None:
        self.running = False
        self.benchmark_running = True
        self.benchmark = PunishmentBenchmark()
        self.round_button.state(["disabled"])
        self.train_button.state(["disabled"])
        self.benchmark_button.state(["disabled"])
        self.status.configure(text="Starting punishment benchmark…")
        self.root.after(1, self._benchmark_step)

    def _benchmark_step(self) -> None:
        if not self.benchmark_running or self.benchmark is None:
            return
        try:
            finished = self.benchmark.advance()
        except (OSError, ValueError, RuntimeError) as error:
            self.benchmark_running = False
            self.status.configure(text=f"Benchmark stopped: {error}")
            self.round_button.state(["!disabled"])
            self.train_button.state(["!disabled"])
            self.benchmark_button.state(["!disabled"])
            return

        self.status.configure(text=self.benchmark.progress)
        if finished:
            results = self.benchmark.results
            self.benchmark_running = False
            self.round_button.state(["!disabled"])
            self.train_button.state(["!disabled"])
            self.benchmark_button.state(["!disabled"])
            self.status.configure(text="Punishment benchmark complete. Results are shown.")
            self._show_benchmark_results(results)
        else:
            self.root.after(1, self._benchmark_step)

    def _show_benchmark_results(
        self, results: tuple[BenchmarkResult, ...]
    ) -> None:
        window = tk.Toplevel(self.root)
        window.title("Punishment benchmark results")
        window.geometry("1180x460")
        frame = ttk.Frame(window, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=(
                "Each rule trained 5 fresh learners using the same random-seed "
                f"schedule. A run ends at {PERFECT_STREAK_TARGET} perfect rounds "
                "in a row or after 5,000 "
                "rounds. For incorrect placements, d is the shared path length "
                "before the first wrong branch and D is the tree-edge distance "
                "between attempted and correct slots. Lower placement counts are better."
            ),
            wraplength=1120,
        ).pack(anchor="w", pady=(0, 12))
        table = ttk.Treeview(
            frame,
            columns=("rule", "formula", "perfect", "average", "median"),
            show="headings",
            height=11,
        )
        table.heading("rule", text="Punishment rule")
        table.heading("formula", text="Cost formula")
        table.heading("perfect", text="3-round streaks")
        table.heading("average", text="Mean placements to 3-round streak")
        table.heading("median", text="Median placements")
        table.column("rule", width=245, anchor="w")
        table.column("formula", width=200, anchor="center")
        table.column("perfect", width=100, anchor="center")
        table.column("average", width=190, anchor="center")
        table.column("median", width=150, anchor="center")
        table.pack(fill="both", expand=True)

        def sort_key(result):
            mean = (
                statistics.mean(result.completion_iterations)
                if result.completion_iterations
                else float("inf")
            )
            return (-result.completed_runs, mean)

        for result in sorted(results, key=sort_key):
            if result.completion_iterations:
                mean_value = f"{statistics.mean(result.completion_iterations):.1f}"
                median_value = f"{statistics.median(result.completion_iterations):.1f}"
            else:
                mean_value = median_value = "No streak reached"
            table.insert(
                "",
                "end",
                values=(
                    result.name,
                    result.formula,
                    f"{result.completed_runs}/{result.repeats}",
                    mean_value,
                    median_value,
                ),
            )

    def _draw_tree(self, result: AttemptResult | None = None) -> None:
        self.canvas.delete("all")
        tree = self.game.tree
        height = max(540, self._tree_depth(tree.root) * 125 + 150)
        self.canvas.configure(scrollregion=(0, 0, 900, height))
        self._draw_node(tree.root, 40, 860, 0, result if result and result.correct else None)
        if result is not None and not result.correct:
            x, y = self._slot_position(tree, result.attempted)
            self._draw_card(x, y, result.rank, wrong=True)
            self.canvas.yview_moveto(max(0, (y - 250) / max(height, 1)))

    def _draw_node(
        self,
        node: Node | None,
        left: float,
        right: float,
        depth: int,
        result: AttemptResult | None,
        path: Placement = (),
    ) -> None:
        if node is None:
            return
        x = (left + right) / 2
        y = 70 + depth * 125
        if node.left is not None:
            child_x = (left + x) / 2
            self.canvas.create_line(x, y + 42, child_x, y + 125 - 42, fill="#8e9698", width=2)
        if node.right is not None:
            child_x = (x + right) / 2
            self.canvas.create_line(x, y + 42, child_x, y + 125 - 42, fill="#8e9698", width=2)

        inserted = result is not None and path == result.attempted
        self._draw_card(x, y, node.rank, correct=inserted)
        self._draw_node(node.left, left, x, depth + 1, result, path + ("L",))
        self._draw_node(node.right, x, right, depth + 1, result, path + ("R",))

    def _draw_card(
        self,
        x: float,
        y: float,
        rank: int,
        *,
        correct: bool = False,
        wrong: bool = False,
    ) -> None:
        half_width, half_height = 27, 40
        outline = "#b94136" if wrong else ("#33794c" if correct else "#6e7477")
        fill = "#f8e2df" if wrong else ("#f3fbf4" if correct else "#fffefa")
        ink = "#b12f25" if wrong else "#202426"
        self.canvas.create_rectangle(
            x - half_width,
            y - half_height,
            x + half_width,
            y + half_height,
            fill=fill,
            outline=outline,
            width=3 if correct or wrong else 2,
            dash=(4, 2) if wrong else (),
        )
        name = rank_name(rank)
        self.canvas.create_text(
            x - 18,
            y - 29,
            text=f"{name}\n♠",
            fill=ink,
            anchor="nw",
            justify="left",
            font=("TkDefaultFont", 9, "bold"),
        )
        self.canvas.create_text(
            x,
            y + 2,
            text="♠",
            fill=ink,
            font=("TkDefaultFont", 24),
        )
        self.canvas.create_text(
            x + 17,
            y + 28,
            text=f"{name}\n♠",
            fill=ink,
            anchor="se",
            justify="right",
            font=("TkDefaultFont", 9, "bold"),
        )

    @staticmethod
    def _tree_depth(node: Node | None) -> int:
        if node is None:
            return 0
        return 1 + max(
            CardTreeApp._tree_depth(node.left), CardTreeApp._tree_depth(node.right)
        )

    @staticmethod
    def _slot_position(tree: BinarySearchTree, placement: Placement) -> tuple[float, float]:
        left, right = 40.0, 860.0
        depth = 0
        node = tree.root
        for direction in placement:
            x = (left + right) / 2
            depth += 1
            if direction == "L":
                right = x
                node = node.left if node is not None else None
            else:
                left = x
                node = node.right if node is not None else None
        return (left + right) / 2, 70 + depth * 125

    @staticmethod
    def _format_path(path: Placement) -> str:
        return "root" if not path else "root → " + " → ".join(path)

    def _update_iteration_label(self) -> None:
        completion = self.learner.completion_iteration
        if completion is None:
            text = (
                f"Placement iterations: {self.learner.iterations} · perfect "
                f"streak: {self.learner.consecutive_perfect_rounds}/"
                f"{PERFECT_STREAK_TARGET}"
            )
        else:
            text = (
                f"Training complete after {completion} placement iterations · "
                f"streak: {PERFECT_STREAK_TARGET}/{PERFECT_STREAK_TARGET}"
            )
        self.iteration_label.configure(text=text)


def run() -> None:
    root = tk.Tk()
    try:
        CardTreeApp(root)
    except (OSError, ValueError):
        root.destroy()
        return
    root.mainloop()
