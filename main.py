import random

from src.game import Game
from src.optimizer import Optimizer


def main():
    language = "pl"
    with open(f"data/codenames/wordlist/{language.lower()}-{language.upper()}/default/wordlist.txt", encoding="utf-8") as f:
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
    moves = optimizer.solve("blue", 3, 10, "worst-case")
    print(moves)


if __name__ == "__main__":
    main()
