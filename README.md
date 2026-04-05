# Codenames Optimizer

Codenames Optimizer is a small project for generating clue suggestions for the board game Codenames.
It uses word embeddings to find a clue that is similar to the words on your team's cards and dissimilar to other cards.

## Wordlist Submodule

This project uses a Git submodule at data/codenames for multilingual Codenames word lists.
The submodule points to the upstream repository sagelga/codenames.

If the data/codenames directory is empty after cloning, initialize submodules with:

```bash
git submodule update --init --recursive
```
