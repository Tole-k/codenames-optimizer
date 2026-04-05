from itertools import combinations
from typing import Literal

import numpy as np

from src.embedder import Embedder
from src.game import Game


class Optimizer:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.embedder = Embedder(game)
        self.game.register_embedder(self.embedder)
        self.embedder.calculate_embeddings()
        weights = {"blank": 1, "opposite": 2, "black": 5}
        self.weights = {k: v / sum(weights.values()) for k, v in weights.items()}

    def _combo_scores(
        self,
        attract_sims: np.ndarray,
        combo: tuple[int, ...],
        risk_scores: np.ndarray,
    ) -> np.ndarray:
        combo_sims = attract_sims[:, combo]
        attract_scores = combo_sims.min(axis=1)
        return attract_scores - risk_scores

    @staticmethod
    def _selection_score(
        repel_centroid: np.ndarray,
        attract_vecs: np.ndarray,
        attract_centroid: np.ndarray,
    ) -> float:
        separation = np.linalg.norm(attract_centroid - repel_centroid)
        spread = np.mean(np.linalg.norm(attract_vecs - attract_centroid, axis=1))
        return float(separation - spread)

    def calculate_clue_scores(
        self,
        vocabulary: np.ndarray,
        attract: np.ndarray,
        selected_combo: tuple[int, ...],
        mode: Literal["avg-case", "worst-case"],
        repel_centroid: np.ndarray | None = None,
        attract_sims: np.ndarray | None = None,
        risk_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        if mode == "avg-case":
            if repel_centroid is None:
                raise ValueError("repel_centroid is required for avg-case")
            attract_centroid = np.mean(attract[list(selected_combo)], axis=0)
            direction = attract_centroid - repel_centroid
            direction_norm = np.linalg.norm(direction)
            if direction_norm == 0:
                return np.zeros(vocabulary.shape[0], dtype=np.float32)
            direction = direction / direction_norm
            return vocabulary @ direction

        if attract_sims is None or risk_scores is None:
            raise ValueError("attract_sims and risk_scores are required for worst-case")
        return self._combo_scores(attract_sims, selected_combo, risk_scores)

    def select_attract_words(
        self,
        attract: np.ndarray,
        vocabulary: np.ndarray,
        attract_sims: np.ndarray,
        risk_scores: np.ndarray,
        repel_centroid: np.ndarray,
        n: int,
        mode: Literal["avg-case", "worst-case"],
    ) -> tuple[int, ...]:
        best_score = -np.inf
        best_combo = None

        for combo in combinations(range(attract_sims.shape[1]), n):
            if mode == "avg-case":
                attract_vecs = attract[list(combo)]
                attract_centroid = np.mean(attract_vecs, axis=0)
                score = self._selection_score(
                    repel_centroid, attract_vecs, attract_centroid
                )
            else:
                clue_scores = self.calculate_clue_scores(
                    vocabulary=vocabulary,
                    attract=attract,
                    selected_combo=combo,
                    mode=mode,
                    attract_sims=attract_sims,
                    risk_scores=risk_scores,
                )
                score = float(clue_scores.max())
            if score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            raise RuntimeError("Unable to find a clue for the current board.")

        return best_combo

    def solve(
        self,
        team: Literal["blue", "red"],
        n: int,
        k: int,
        mode: Literal["avg-case", "worst-case"] = "worst-case",
    ):
        if n < 1:
            raise ValueError("n must be at least 1")
        if k < 1:
            raise ValueError("k must be at least 1")

        if team == "blue":
            attract = self.embedder.get_blue()
            opposite = self.embedder.get_red()
        else:
            attract = self.embedder.get_red()
            opposite = self.embedder.get_blue()

        if n > len(attract):
            raise ValueError(
                f"n={n} cannot be greater than available target words={len(attract)}"
            )

        blank = self.embedder.get_blank()
        black = self.embedder.get_black()
        vocabulary = self.embedder.get_vocab()

        attract_sims = vocabulary @ attract.T

        repel_vecs = np.concatenate([blank, opposite, black], axis=0)
        repel_weights = np.concatenate([
            np.full(len(blank), self.weights["blank"]),
            np.full(len(opposite), self.weights["opposite"]),
            np.full(len(black), self.weights["black"]),
        ])
        repel_centroid = np.average(repel_vecs, axis=0, weights=repel_weights)

        blank_sims = vocabulary @ blank.T
        opposite_sims = vocabulary @ opposite.T
        black_sims = vocabulary @ black.T
        repel_sims = np.concatenate(
            [
                blank_sims * self.weights["blank"],
                opposite_sims * self.weights["opposite"],
                black_sims * self.weights["black"],
            ],
            axis=1,
        )
        risk_scores = repel_sims.max(axis=1)

        selected_indices = self.select_attract_words(
            attract,
            vocabulary,
            attract_sims,
            risk_scores,
            repel_centroid,
            n,
            mode,
        )

        selected_words = [getattr(self.game, team)[i] for i in selected_indices]
        print(f"Optimizing prompt for {n} words: {selected_words}")

        clue_scores = self.calculate_clue_scores(
            vocabulary=vocabulary,
            attract=attract,
            selected_combo=selected_indices,
            mode=mode,
            repel_centroid=repel_centroid,
            attract_sims=attract_sims,
            risk_scores=risk_scores,
        )

        best_scores = np.argsort(-clue_scores)[:k]
        return [self.game.vocabulary[idx] for idx in best_scores]
