from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid import GridMap


@dataclass(slots=True)
class ACOMResult:
    positions: dict[str, tuple[int, int]]
    history: list[float]
    accepted_swaps: int
    total_attempts: int
    initial_cost: float
    final_cost: float


class ACOMMapper:
    def __init__(
        self,
        grid: GridMap,
        semantic_distances: np.ndarray,
        num_ants: int,
        max_iter: int,
        radius: int,
        semantic_k: int = 8,
        repulsion_weight: float = 0.35,
        random_seed: int = 42,
    ) -> None:
        self.grid = grid
        self.semantic_distances = semantic_distances.astype(np.float64)
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.radius = radius
        self.semantic_k = semantic_k
        self.repulsion_weight = repulsion_weight
        self.rng = np.random.default_rng(random_seed)
        self.doc_ids = list(grid.doc_ids)
        self.doc_index = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.semantic_neighbor_count = min(max(3, semantic_k), max(1, len(self.doc_ids) - 1))

        max_semantic = float(np.max(self.semantic_distances))
        self.semantic_norm = self.semantic_distances / max_semantic if max_semantic > 0 else self.semantic_distances
        self.semantic_neighbors = self._build_semantic_neighbors()
        self.reverse_semantic_neighbors = self._build_reverse_semantic_neighbors()
        self.neighbor_weights = self._build_neighbor_weights()

    def _build_semantic_neighbors(self) -> dict[str, list[str]]:
        neighbors: dict[str, list[str]] = {}
        for doc_id in self.doc_ids:
            index = self.doc_index[doc_id]
            ordered = [self.doc_ids[i] for i in np.argsort(self.semantic_norm[index]) if self.doc_ids[i] != doc_id]
            neighbors[doc_id] = ordered[: self.semantic_neighbor_count]
        return neighbors

    def _build_reverse_semantic_neighbors(self) -> dict[str, set[str]]:
        reverse_neighbors = {doc_id: set() for doc_id in self.doc_ids}
        for doc_id, neighbors in self.semantic_neighbors.items():
            for neighbor_id in neighbors:
                reverse_neighbors[neighbor_id].add(doc_id)
        return reverse_neighbors

    def _build_neighbor_weights(self) -> dict[str, dict[str, float]]:
        weights: dict[str, dict[str, float]] = {}
        for doc_id, neighbors in self.semantic_neighbors.items():
            doc_weights: dict[str, float] = {}
            for rank, neighbor_id in enumerate(neighbors, start=1):
                doc_weights[neighbor_id] = 1.0 / rank
            weights[doc_id] = doc_weights
        return weights

    def _normalized_grid_distance(self, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> float:
        return self.grid.grid_distance(pos_a, pos_b) / max(self.grid.max_distance, 1.0)

    def local_cost(self, doc_id: str) -> float:
        index = self.doc_index[doc_id]
        position = self.grid.get_position(doc_id)
        total = 0.0
        for neighbor_id in self.semantic_neighbors[doc_id]:
            neighbor_position = self.grid.get_position(neighbor_id)
            similarity_weight = self.neighbor_weights[doc_id][neighbor_id]
            total += similarity_weight * self._normalized_grid_distance(position, neighbor_position)

        for local_neighbor_id in self.grid.get_neighbors(position, radius=self.radius):
            local_index = self.doc_index[local_neighbor_id]
            total += self.repulsion_weight * self.semantic_norm[index, local_index]
        return total

    def swap_cost(self, doc_a: str, doc_b: str) -> tuple[float, float]:
        affected = self._affected_docs(doc_a, doc_b)
        before = self._cost_for_docs(affected)
        self.grid.swap(doc_a, doc_b)
        after = self._cost_for_docs(affected)
        self.grid.swap(doc_a, doc_b)
        return before, after

    def total_cost(self) -> float:
        return sum(self.local_cost(doc_id) for doc_id in self.doc_ids)

    def _cost_for_docs(self, doc_ids: set[str]) -> float:
        return sum(self.local_cost(doc_id) for doc_id in doc_ids)

    def _affected_docs(self, doc_a: str, doc_b: str) -> set[str]:
        pos_a = self.grid.get_position(doc_a)
        pos_b = self.grid.get_position(doc_b)
        affected = {doc_a, doc_b}
        affected.update(self.semantic_neighbors[doc_a])
        affected.update(self.semantic_neighbors[doc_b])
        affected.update(self.reverse_semantic_neighbors[doc_a])
        affected.update(self.reverse_semantic_neighbors[doc_b])
        affected.update(self.grid.get_neighbors(pos_a, radius=self.radius))
        affected.update(self.grid.get_neighbors(pos_b, radius=self.radius))
        return affected

    def _candidate_docs(self, doc_a: str) -> list[str]:
        index = self.doc_index[doc_a]
        semantic_order = [self.doc_ids[i] for i in np.argsort(self.semantic_norm[index]) if self.doc_ids[i] != doc_a]

        nearest = semantic_order[: max(6, self.semantic_neighbor_count * 2)]
        farthest = semantic_order[-max(4, self.radius * 4) :]
        current_neighbors = self.grid.get_neighbors(self.grid.get_position(doc_a), radius=max(1, self.radius))
        random_candidates = []
        while len(random_candidates) < min(8, max(0, len(self.doc_ids) - 1)):
            candidate = self.doc_ids[int(self.rng.integers(0, len(self.doc_ids)))]
            if candidate != doc_a and candidate not in random_candidates:
                random_candidates.append(candidate)

        ordered_candidates = []
        for candidate in nearest + farthest + current_neighbors + random_candidates:
            if candidate not in ordered_candidates:
                ordered_candidates.append(candidate)
        return ordered_candidates

    def _propose_swap(self) -> tuple[str, str, float, int]:
        doc_a = self.doc_ids[int(self.rng.integers(0, len(self.doc_ids)))]
        best_pair = (doc_a, doc_a)
        best_improvement = 0.0
        candidates = self._candidate_docs(doc_a)

        for candidate in candidates:
            before, after = self.swap_cost(doc_a, candidate)
            improvement = before - after
            if improvement > best_improvement:
                best_improvement = improvement
                best_pair = (doc_a, candidate)

        return best_pair[0], best_pair[1], best_improvement, len(candidates)

    def run(self) -> ACOMResult:
        initial_cost = self.total_cost()
        history: list[float] = [initial_cost]
        accepted_swaps = 0
        total_attempts = 0

        for _ in range(self.max_iter):
            improved_this_round = False
            for _ in range(self.num_ants):
                doc_a, doc_b, improvement, candidate_count = self._propose_swap()
                total_attempts += max(1, candidate_count)
                if doc_a == doc_b or improvement <= 0:
                    break

                self.grid.swap(doc_a, doc_b)
                accepted_swaps += 1
                improved_this_round = True

            history.append(self.total_cost())
            if not improved_this_round:
                break

        final_cost = history[-1]
        return ACOMResult(
            positions=dict(self.grid.positions),
            history=history,
            accepted_swaps=accepted_swaps,
            total_attempts=total_attempts,
            initial_cost=initial_cost,
            final_cost=final_cost,
        )
