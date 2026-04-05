from typing import TYPE_CHECKING

from wordfreq import top_n_list

if TYPE_CHECKING:
    from src.embedder import Embedder


class Game:
    model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(
        self, blue: list[str], red: list[str], blank: list[str], black: list[str], language:str
    ) -> None:
        self.blue = blue
        self.red = red
        self.blank = blank
        self.black = black
        self.language = language
        self.vocabulary = top_n_list(language, 20000)
        self.embedder = None

    def register_embedder(self, embedder: Embedder):
        self.embedder = embedder

    def exclude(self, codenames: set[str], clr: str):
        indices = []
        remaining = []
        for i, codename in enumerate(getattr(self, clr)):
            if codename in codenames:
                indices.append(i)
            else:
                remaining.append(codename)
        setattr(self, clr, remaining)
        if self.embedder is not None:
            self.embedder.exclude(indices, clr)
