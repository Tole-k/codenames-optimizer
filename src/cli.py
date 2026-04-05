import random
from typing import Literal

import fire

from src.game import Game
from src.optimizer import Optimizer


def random_play(language: Literal["pl", "en"] = "pl", n: int = 2, k: int = 10) -> None:
    with open(
        f"data/codenames/wordlist/{language.lower()}-{language.upper()}/default/wordlist.txt",
        encoding="utf-8",
    ) as f:
        vocab = [line.strip() for line in f if line.strip()]
    board = random.sample(vocab, 25)
    blue = board[:9]
    print(f"{blue=}")
    red = board[9:17]
    print(f"{red=}")
    blank = board[17:24]
    print(f"{blank=}")
    black = board[24:25]
    print(f"{black=}")
    game = Game(blue, red, blank, black, language)
    optimizer = Optimizer(game)
    moves = optimizer.solve("blue", n, k, "worst-case")
    print(f"Suggested clues: {moves}")


def play(
    blue: list[str],
    red: list[str],
    blank: list[str],
    black: list[str],
    language: Literal["pl", "en"] = "pl",
    n: int = 2,
    k: int = 10,
):
    game = Game(blue, red, blank, black, language)
    optimizer = Optimizer(game)
    moves = optimizer.solve("blue", n, k, "worst-case")
    print(f"Suggested clues: {moves}")


def cli():
    fire.Fire({"random_play": random_play, "play": play})


if __name__ == "__main__":
    cli()
