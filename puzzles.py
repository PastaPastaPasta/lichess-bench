"""Puzzle loading and selection from Lichess CSV data."""

import csv
import random
import bisect
from typing import List, Dict, Optional, Set


def load_puzzles(
    csv_path: str,
    min_rating: int = 800,
    max_rating: int = 2800
) -> List[Dict]:
    """Load puzzles from CSV file, filtering by rating range.

    Args:
        csv_path: Path to the Lichess puzzle CSV file
        min_rating: Minimum puzzle rating to include
        max_rating: Maximum puzzle rating to include

    Returns:
        List of puzzle dicts sorted by rating, each containing:
        - puzzle_id: Unique identifier
        - fen: Starting position in FEN notation
        - moves: List of UCI moves (first is opponent setup)
        - rating: Puzzle difficulty rating
        - themes: List of theme tags
    """
    puzzles = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rating = int(row['Rating'])
            except (ValueError, KeyError):
                continue

            if min_rating <= rating <= max_rating:
                puzzles.append({
                    'puzzle_id': row['PuzzleId'],
                    'fen': row['FEN'],
                    'moves': row['Moves'].split(),
                    'rating': rating,
                    'themes': row['Themes'].split() if row.get('Themes') else []
                })

    # Sort by rating for efficient range queries
    puzzles.sort(key=lambda x: x['rating'])
    return puzzles


def get_puzzle_near_rating(
    puzzles: List[Dict],
    target_rating: float,
    window: int = 100,
    exclude_ids: Optional[Set[str]] = None
) -> Optional[Dict]:
    """Select a random puzzle within rating window of target.

    Uses binary search for efficient lookup in the sorted puzzle list.

    Args:
        puzzles: List of puzzles sorted by rating
        target_rating: Target rating to search around
        window: Plus/minus window for rating range
        exclude_ids: Set of puzzle IDs to exclude (already used)

    Returns:
        A puzzle dict, or None if no suitable puzzle found
    """
    if not puzzles:
        return None

    min_rating = target_rating - window
    max_rating = target_rating + window

    # Binary search for rating range bounds
    ratings = [p['rating'] for p in puzzles]
    left = bisect.bisect_left(ratings, min_rating)
    right = bisect.bisect_right(ratings, max_rating)

    # Collect candidates (excluding already-used puzzles)
    candidates = []
    for i in range(left, right):
        puzzle = puzzles[i]
        if exclude_ids is None or puzzle['puzzle_id'] not in exclude_ids:
            candidates.append(puzzle)

    if not candidates:
        return None

    return random.choice(candidates)
