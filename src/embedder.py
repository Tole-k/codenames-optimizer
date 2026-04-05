import numpy as np
from sentence_transformers import SentenceTransformer

from src.game import Game


class Embedder:
    model_checkpoint = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self, game: Game) -> None:
        self.game = game
        self.model = SentenceTransformer(self.model_checkpoint)
        self.embeddings: dict[str, np.ndarray] | None = None

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.sqrt(np.sum(vectors * vectors, axis=1, keepdims=True))
        norms[norms == 0] = 1.0
        return vectors / norms

    def calculate_embeddings(self):
        if self.embeddings is None:
            self.embeddings = {}
            for clr in ["blue", "red", "blank", "black"]:
                clr_embeddings = self.model.encode(getattr(self.game, clr))
                self.embeddings[clr] = self._normalize_rows(np.asarray(clr_embeddings, dtype=np.float32))
            self.embeddings["vocabulary"] = self.calculate_vocab_embeddings_w_caching(
                self.game.language
            )
        return self.embeddings

    def calculate_vocab_embeddings_w_caching(self, language):
        try:
            with open(f"vocab_{language}.npy", "rb") as f:
                embeddings = np.load(f)
        except FileNotFoundError:
            embeddings = self.model.encode(self.game.vocabulary)
        embeddings = self._normalize_rows(np.asarray(embeddings, dtype=np.float32))
        with open(f"vocab_{language}.npy", "wb") as f:
            np.save(f, embeddings)
        return embeddings

    def exclude(self, code_name_indices: list[int], clr: str):
        if self.embeddings is None:
            raise TypeError("Call calculate embeddings first!")
        self.embeddings[clr] = np.delete(self.embeddings[clr], code_name_indices, axis=0)

    def get_blue(self):
        if self.embeddings is None:
            raise TypeError("Call calculate embeddings first!")
        return self.embeddings["blue"]

    def get_red(self):
        if self.embeddings is None:
            raise TypeError("Call calculate embeddings first!")
        return self.embeddings["red"]

    def get_blank(self):
        if self.embeddings is None:
            raise TypeError("Call calculate embeddings first!")
        return self.embeddings["blank"]

    def get_black(self):
        if self.embeddings is None:
            raise TypeError("Call calculate embeddings first!")
        return self.embeddings["black"]

    def get_vocab(self):
        if self.embeddings is None:
            raise TypeError("Call calculate embeddings first!")
        return self.embeddings["vocabulary"]
