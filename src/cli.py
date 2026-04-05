import random
from pathlib import Path
from typing import Literal

import fire

from src.game import Game
from src.optimizer import Optimizer

Language = Literal["pl", "en"]
Team = Literal["blue", "red"]
WORDLIST_ROOT = Path(__file__).resolve().parents[1] / "data" / "codenames" / "wordlist"


def random_play(language: Language = "pl", n: int = 2, k: int = 10) -> None:
    starting_team, blue, red, blank, black = generate_game(language)
    optimizer, _ = setup_game(blue, red, blank, black, language)
    turn(k, optimizer, starting_team, n)


def play(
    blue: list[str],
    red: list[str],
    blank: list[str],
    black: list[str],
    language: Language = "pl",
    starting_team: Team | None = None,
    n: int = 2,
    k: int = 10,
) -> None:
    optimizer, _ = setup_game(blue, red, blank, black, language)
    turn(k, optimizer, resolve_starting_team(blue, red, starting_team), n)


def setup_game(
    blue: list[str],
    red: list[str],
    blank: list[str],
    black: list[str],
    language: Language,
) -> tuple[Optimizer, Game]:
    game = Game(blue, red, blank, black, language)
    optimizer = Optimizer(game)
    return optimizer, game


def random_game(language: Language = "pl", k: int = 10) -> None:
    starting_team, blue, red, blank, black = generate_game(language)
    optimizer, game = setup_game(blue, red, blank, black, language)
    run_game(k, optimizer, game, starting_team)


def game(
    blue: list[str],
    red: list[str],
    blank: list[str],
    black: list[str],
    language: Language = "pl",
    starting_team: Team | None = None,
    k: int = 10,
) -> None:
    optimizer, game = setup_game(blue, red, blank, black, language)
    run_game(k, optimizer, game, resolve_starting_team(blue, red, starting_team))


def run_game(k: int, optimizer: Optimizer, game: Game, starting_team: Team) -> None:
    team = starting_team
    while True:
        print(f"{team}'s turn")
        n = read_guess_count(team, game)
        turn(k, optimizer, team, n)
        blue_exclude = parse_words_input("guessed blue words: ")
        red_exclude = parse_words_input("guessed red words: ")
        blank_exclude = parse_words_input("guessed blank words: ")
        black_exclude = parse_words_input("guessed black words: ")
        if black_exclude:
            print(f"{team} lost")
            break
        if blue_exclude:
            game.exclude(set(blue_exclude), "blue")
        if red_exclude:
            game.exclude(set(red_exclude), "red")
        if blank_exclude:
            game.exclude(set(blank_exclude), "blank")
        if len(game.blue) == 0:
            print("blue won!")
            break
        if len(game.red) == 0:
            print("red won!")
            break
        team = "red" if team == "blue" else "blue"


def turn(k: int, optimizer: Optimizer, team: Team, n: int) -> None:
    moves = optimizer.solve(team, n, k, "worst-case")
    print(f"Suggested clues: {moves}")


def resolve_starting_team(
    blue: list[str],
    red: list[str],
    starting_team: Team | None = None,
) -> Team:
    if starting_team is not None:
        return starting_team
    if len(blue) == len(red) + 1:
        return "blue"
    if len(red) == len(blue) + 1:
        return "red"
    raise ValueError(
        "Unable to infer the starting team from the board. "
        "Pass starting_team='blue' or starting_team='red'."
    )


def read_guess_count(team: Team, game: Game) -> int:
    remaining_words = len(getattr(game, team))
    while True:
        raw_value = input("Number of words to match: ").strip()
        try:
            guess_count = int(raw_value)
        except ValueError:
            print("Enter a whole number.")
            continue

        if 1 <= guess_count <= remaining_words:
            return guess_count
        print(f"Enter a number between 1 and {remaining_words}.")


def parse_words_input(prompt: str) -> list[str]:
    return input(prompt).strip().split()


def generate_game(
    language: Language,
) -> tuple[Team, list[str], list[str], list[str], list[str]]:
    wordlist_path = (
        WORDLIST_ROOT
        / f"{language.lower()}-{language.upper()}"
        / "default"
        / "wordlist.txt"
    )
    with wordlist_path.open(encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    starting_team: Team = random.choice(["blue", "red"])
    board = random.sample(vocab, 25)
    blue_count = 9 if starting_team == "blue" else 8
    blue = board[:blue_count]
    print(f"{blue=}")
    red = board[blue_count:17]
    print(f"{red=}")
    blank = board[17:24]
    print(f"{blank=}")
    black = board[24:25]
    print(f"{black=}")
    return starting_team, blue, red, blank, black


def cli() -> None:
    fire.Fire({
        "random_play": random_play,
        "play": play,
        "random_game": random_game,
        "game": game,
    })


if __name__ == "__main__":
    cli()
