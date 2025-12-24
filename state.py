"""State persistence for resumable benchmark runs."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List


@dataclass
class BenchmarkState:
    """Persistent state for a benchmark run."""
    model: str
    started_at: str

    # Random seed for reproducibility
    seed: int = 42

    # Overall ELO tracking
    elo_rating: float = 1500.0
    games_played: int = 0
    wins: int = 0

    # First-shot ELO tracking (success = complete forcing line on first response)
    first_shot_elo: float = 1500.0
    first_shot_games: int = 0
    first_shot_wins: int = 0

    # Valid response ELO tracking (success = valid XML + all legal chess moves)
    valid_response_elo: float = 1500.0
    valid_response_games: int = 0
    valid_response_wins: int = 0

    # Per-tag tracking
    tag_ratings: Dict[str, float] = field(default_factory=dict)
    tag_games: Dict[str, int] = field(default_factory=dict)
    tag_wins: Dict[str, int] = field(default_factory=dict)

    # Used puzzle IDs (to avoid repeats)
    used_puzzle_ids: List[str] = field(default_factory=list)

    # Result history for analysis
    results: List[Dict] = field(default_factory=list)


def state_path(model: str, data_dir: str = "data") -> Path:
    """Get state file path for a model.

    Sanitizes model name for use as filename.
    """
    safe_name = model.replace("/", "_").replace(":", "_")
    return Path(data_dir) / f"{safe_name}_state.json"


def load_state(model: str, data_dir: str = "data") -> Optional[BenchmarkState]:
    """Load existing state from disk.

    Args:
        model: Model identifier
        data_dir: Directory containing state files

    Returns:
        BenchmarkState if found, None otherwise
    """
    path = state_path(model, data_dir)
    if not path.exists():
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    # Backwards compatibility: add default values for new fields
    if 'first_shot_elo' not in data:
        data['first_shot_elo'] = 1500.0
    if 'first_shot_games' not in data:
        data['first_shot_games'] = 0
    if 'first_shot_wins' not in data:
        data['first_shot_wins'] = 0
    if 'valid_response_elo' not in data:
        data['valid_response_elo'] = 1500.0
    if 'valid_response_games' not in data:
        data['valid_response_games'] = 0
    if 'valid_response_wins' not in data:
        data['valid_response_wins'] = 0

    return BenchmarkState(**data)


def save_state(state: BenchmarkState, data_dir: str = "data") -> None:
    """Save state to disk.

    Creates the data directory if it doesn't exist.

    Args:
        state: BenchmarkState to save
        data_dir: Directory for state files
    """
    path = state_path(state.model, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(asdict(state), f, indent=2)


def create_new_state(model: str, initial_elo: float = 1500.0, seed: int = 42) -> BenchmarkState:
    """Create fresh state for a new benchmark run.

    Args:
        model: Model identifier
        initial_elo: Starting ELO rating
        seed: Random seed for reproducibility

    Returns:
        New BenchmarkState
    """
    return BenchmarkState(
        model=model,
        started_at=datetime.now().isoformat(),
        seed=seed,
        elo_rating=initial_elo
    )
