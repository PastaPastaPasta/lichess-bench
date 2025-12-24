"""ELO rating calculations and confidence intervals."""

import math
from typing import Dict

from scipy.stats import t as t_dist


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
    """Calculate the +/- confidence interval for ELO rating using t-distribution.

    Uses the t-distribution instead of z-scores for statistical correctness
    at small sample sizes. The t-distribution converges to the normal
    distribution as n approaches infinity, so this works for all sample sizes.

    Based on Elo's empirical finding that individual game performance
    has a standard deviation of ~200 rating points.

    The standard error of the mean rating decreases as sqrt(n):
    SE = 200 / sqrt(n)

    For t-distribution CI: margin = t(df, alpha/2) * SE
    where df = n - 1 (degrees of freedom)

    Args:
        games_played: Number of games played
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        Plus/minus margin for the confidence interval
    """
    if games_played <= 1:
        # With 0 or 1 game, degrees of freedom is 0 or undefined
        return float('inf')

    performance_sd = 200.0  # Standard deviation of single game performance
    standard_error = performance_sd / math.sqrt(games_played)

    # Degrees of freedom = n - 1
    df = games_played - 1

    # Two-tailed t-value: for 95% CI, we need t at 0.975 (1 - 0.05/2)
    alpha = 1 - confidence
    t_value = t_dist.ppf(1 - alpha / 2, df)

    return t_value * standard_error


def is_low_confidence(games_played: int, ci: float = None,
                      min_games: int = 30, max_ci: float = 100.0) -> bool:
    """Check if a model's rating has low statistical confidence.

    Args:
        games_played: Number of games played
        ci: Pre-computed confidence interval (computed if not provided)
        min_games: Minimum games threshold (default 30)
        max_ci: Maximum acceptable CI (default 100)

    Returns:
        True if confidence is low (should show marker)
    """
    if games_played < min_games:
        return True

    if ci is None:
        ci = confidence_interval(games_played)

    return ci > max_ci


def puzzles_needed_for_confidence(target_margin: float, confidence: float = 0.95) -> int:
    """Calculate number of puzzles needed to achieve target confidence margin.

    Uses an iterative approach since the t-value depends on sample size.

    Examples:
        puzzles_needed_for_confidence(25)  # ~250 puzzles for +-25 at 95%
        puzzles_needed_for_confidence(100) # ~20 puzzles for +-100 at 95%

    Args:
        target_margin: Desired +/- margin
        confidence: Confidence level (default 0.95)

    Returns:
        Number of puzzles needed
    """
    performance_sd = 200.0

    # Start with z-score estimate as initial guess
    z_approx = 1.96 if confidence == 0.95 else t_dist.ppf(1 - (1 - confidence) / 2, 1000)
    n_estimate = math.ceil((z_approx * performance_sd / target_margin) ** 2)

    # Iteratively find exact n using t-distribution
    for n in range(max(2, n_estimate - 10), n_estimate + 50):
        ci = confidence_interval(n, confidence)
        if ci <= target_margin:
            return n

    # Fallback to z-score estimate if iteration doesn't converge
    return n_estimate


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
