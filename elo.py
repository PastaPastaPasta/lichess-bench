"""ELO rating calculations and confidence intervals."""

import math
from typing import Dict


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B.

    Uses standard ELO formula: E = 1 / (1 + 10^((Rb - Ra) / 400))
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(games_played: int) -> float:
    """Dynamic K-factor that decreases as more games are played.

    - First 30 games: K=40 (high volatility for fast convergence)
    - Games 30-100: K=20 (moderate adjustment)
    - After 100 games: K=10 (stable rating)
    """
    if games_played < 30:
        return 40.0
    elif games_played < 100:
        return 20.0
    else:
        return 10.0


def update_elo(current: float, opponent: float, won: bool, games_played: int) -> float:
    """Update ELO rating after a game.

    Args:
        current: Current player rating
        opponent: Opponent (puzzle) rating
        won: True if player won (solved puzzle correctly)
        games_played: Number of games played so far (for K-factor)

    Returns:
        New rating after update
    """
    expected = expected_score(current, opponent)
    actual = 1.0 if won else 0.0
    k = k_factor(games_played)
    return current + k * (actual - expected)


def confidence_interval(games_played: int, confidence: float = 0.95) -> float:
    """Calculate the +/- confidence interval for ELO rating.

    Based on Elo's empirical finding that individual game performance
    has a standard deviation of ~200 rating points.

    The standard error of the mean rating decreases as sqrt(n):
    SE = 200 / sqrt(n)

    For 95% CI: margin = 1.96 * SE

    Args:
        games_played: Number of games played
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        Plus/minus margin for the confidence interval
    """
    if games_played == 0:
        return float('inf')

    performance_sd = 200.0  # Standard deviation of single game performance
    standard_error = performance_sd / math.sqrt(games_played)

    # Z-scores for confidence levels
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    z = z_scores.get(confidence, 1.96)

    return z * standard_error


def puzzles_needed_for_confidence(target_margin: float, confidence: float = 0.95) -> int:
    """Calculate number of puzzles needed to achieve target confidence margin.

    Examples:
        puzzles_needed_for_confidence(25)  # ~246 puzzles for +-25 at 95%
        puzzles_needed_for_confidence(100) # ~16 puzzles for +-100 at 95%

    Args:
        target_margin: Desired +/- margin
        confidence: Confidence level (default 0.95)

    Returns:
        Number of puzzles needed
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    performance_sd = 200.0

    # margin = z * sd / sqrt(n)
    # n = (z * sd / margin)^2
    return math.ceil((z * performance_sd / target_margin) ** 2)


def update_tag_elos(
    tag_ratings: Dict[str, float],
    tag_games: Dict[str, int],
    tag_wins: Dict[str, int],
    themes: list,
    puzzle_rating: int,
    won: bool,
    initial_rating: float = 1500.0
) -> None:
    """Update ELO for each tag associated with a puzzle.

    Modifies the dictionaries in place.

    Args:
        tag_ratings: Dict mapping tag -> current rating
        tag_games: Dict mapping tag -> games played
        tag_wins: Dict mapping tag -> wins
        themes: List of tags (themes) for this puzzle
        puzzle_rating: The puzzle's difficulty rating
        won: Whether the puzzle was solved correctly
        initial_rating: Starting rating for new tags
    """
    for tag in themes:
        # Initialize tag if not seen before
        if tag not in tag_ratings:
            tag_ratings[tag] = initial_rating
            tag_games[tag] = 0
            tag_wins[tag] = 0

        # Update tag ELO
        old_rating = tag_ratings[tag]
        games = tag_games[tag]

        tag_ratings[tag] = update_elo(old_rating, puzzle_rating, won, games)
        tag_games[tag] += 1
        if won:
            tag_wins[tag] += 1
