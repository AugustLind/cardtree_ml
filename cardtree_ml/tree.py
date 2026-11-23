from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Placement: TypeAlias = tuple[str, ...]

RANKS = tuple(range(1, 14))
RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}


def rank_name(rank: int) -> str:
    return RANK_NAMES.get(rank, str(rank))


@dataclass
class Node:
    rank: int
    left: Node | None = None
    right: Node | None = None


class BinarySearchTree:
    def __init__(self) -> None:
        self.root: Node | None = None

    def correct_placement(self, rank: int) -> Placement:
        """Return the unique empty child slot for rank in this tree."""
        path: list[str] = []
        node = self.root
        while node is not None:
            if rank == node.rank:
                raise ValueError(f"rank {rank} is already in the tree")
            direction = "L" if rank < node.rank else "R"
            path.append(direction)
            node = node.left if direction == "L" else node.right
        return tuple(path)

    def insert(self, rank: int, placement: Placement) -> None:
        """Insert at the correct empty slot, preserving the BST invariant."""
        if placement != self.correct_placement(rank):
            raise ValueError("card does not belong at this placement")
        if self.root is None:
            self.root = Node(rank)
            return

        node = self.root
        for direction in placement[:-1]:
            node = node.left if direction == "L" else node.right
            if node is None:
                raise ValueError("placement path does not lead through the tree")
        direction = placement[-1]
        if direction not in ("L", "R"):
            raise ValueError(f"invalid placement direction: {direction!r}")
        if getattr(node, "left" if direction == "L" else "right") is not None:
            raise ValueError("placement slot is already occupied")
        setattr(node, "left" if direction == "L" else "right", Node(rank))
