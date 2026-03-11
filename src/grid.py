from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


Position = tuple[int, int]


@dataclass(slots=True)
class GridMap:
    rows: int
    cols: int
    doc_ids: list[str]
    random_seed: int = 42
    rng: np.random.Generator = field(init=False, repr=False)
    grid: np.ndarray = field(init=False, repr=False)
    positions: dict[str, Position] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        capacity = self.rows * self.cols
        if len(self.doc_ids) > capacity:
            raise ValueError(
                f"Grid capacity is {capacity}, but received {len(self.doc_ids)} documents."
            )
        self.rng = np.random.default_rng(self.random_seed)
        self.grid: np.ndarray = np.full((self.rows, self.cols), None, dtype=object)
        self.positions: dict[str, Position] = {}

    def initialize_random(self) -> None:
        coordinates = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.rng.shuffle(coordinates)
        self.grid[:] = None
        self.positions.clear()
        for doc_id, position in zip(self.doc_ids, coordinates):
            row, col = position
            self.grid[row, col] = doc_id
            self.positions[doc_id] = position

    def get_position(self, doc_id: str) -> Position:
        return self.positions[doc_id]

    def get_doc_at(self, position: Position) -> str | None:
        row, col = position
        return self.grid[row, col]

    def set_position(self, doc_id: str, position: Position) -> None:
        row, col = position
        existing = self.grid[row, col]
        if existing is not None and existing != doc_id:
            raise ValueError(f"Cell {position} is already occupied by {existing}.")
        old_position = self.positions.get(doc_id)
        if old_position is not None:
            old_row, old_col = old_position
            self.grid[old_row, old_col] = None
        self.grid[row, col] = doc_id
        self.positions[doc_id] = position

    def swap(self, doc_a: str, doc_b: str) -> None:
        pos_a = self.positions[doc_a]
        pos_b = self.positions[doc_b]
        self.grid[pos_a] = doc_b
        self.grid[pos_b] = doc_a
        self.positions[doc_a], self.positions[doc_b] = pos_b, pos_a

    def get_neighbors(self, position: Position, radius: int = 1, include_center: bool = False) -> list[str]:
        row, col = position
        neighbors: list[str] = []
        for r in range(max(0, row - radius), min(self.rows, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.cols, col + radius + 1)):
                if not include_center and (r, c) == position:
                    continue
                doc_id = self.grid[r, c]
                if doc_id is not None:
                    neighbors.append(doc_id)
        return neighbors

    def grid_distance(self, pos_a: Position, pos_b: Position, metric: str = "euclidean") -> float:
        delta_r = pos_a[0] - pos_b[0]
        delta_c = pos_a[1] - pos_b[1]
        if metric == "manhattan":
            return float(abs(delta_r) + abs(delta_c))
        if metric == "chebyshev":
            return float(max(abs(delta_r), abs(delta_c)))
        return float(np.hypot(delta_r, delta_c))

    def random_doc_id(self) -> str:
        index = self.rng.integers(0, len(self.doc_ids))
        return self.doc_ids[int(index)]

    def iter_positions(self) -> Iterable[tuple[str, Position]]:
        return self.positions.items()

    @property
    def max_distance(self) -> float:
        return self.grid_distance((0, 0), (self.rows - 1, self.cols - 1))
