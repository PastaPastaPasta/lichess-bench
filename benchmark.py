#!/usr/bin/env python3
"""
Minimal Chess Puzzle Benchmark for LLM Evaluation

Evaluates language models on Lichess chess puzzles, calculating ELO ratings
with confidence intervals. Models must solve the entire puzzle line correctly.

Usage:
    python benchmark.py --model anthropic/claude-3.5-sonnet
    python benchmark.py --model openai/gpt-4 --puzzles 500
    python benchmark.py --model anthropic/claude-3.5-sonnet --resume
"""

import argparse
import os
import random
import re
import sys
from typing import Dict, Optional

import chess
from openai import OpenAI

from puzzles import load_puzzles, get_puzzle_near_rating
from elo import update_elo, confidence_interval, puzzles_needed_for_confidence, update_tag_elos
from state import load_state, save_state, create_new_state, BenchmarkState
from traces import PuzzleTrace, TurnTrace, append_trace, create_turn_trace
from dataclasses import asdict
from datetime import datetime


# Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PUZZLE_CSV = "puzzles/lichess_db_puzzle.csv"
RATING_WINDOW = 100  # Select puzzles within +/-100 of model's current rating
INITIAL_ELO = 1500.0
SAVE_INTERVAL = 10  # Save state every N puzzles

SYSTEM_PROMPT = """You are solving a chess puzzle. You will receive positions and must find the best move.

Rules:
- Respond with ONLY the move in UCI notation (e.g., e2e4, g1f3, e7e8q for promotion)
- No explanation, just the move
- The puzzle may require multiple moves - you will be told the opponent's response after each correct move
"""

# Regex pattern for UCI moves: source square + destination square + optional promotion piece
UCI_MOVE_PATTERN = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', re.IGNORECASE)


def extract_uci_move(response: str) -> str:
    """Extract a UCI move from model response.

    Handles both clean responses (just the move) and responses with
    chain-of-thought reasoning. Prefers the last UCI move found
    (models often state the final answer at the end).

    Args:
        response: Raw model response text

    Returns:
        Extracted UCI move in lowercase, or empty string if not found
    """
    if not response:
        return ""

    # Find all UCI moves in the response
    matches = UCI_MOVE_PATTERN.findall(response)

    if matches:
        # Return the last match (final answer is usually at the end)
        return matches[-1].lower()

    # Fallback: try first word if it looks like a move
    first_word = response.strip().split()[0].lower() if response.strip() else ""
    if UCI_MOVE_PATTERN.match(first_word):
        return first_word

    return ""


def evaluate_puzzle(
    client: OpenAI,
    model: str,
    puzzle: Dict,
    verbose: bool = False
) -> Dict:
    """Evaluate a single puzzle using multi-turn conversation.

    The model must get ALL moves correct to pass the puzzle.

    Args:
        client: OpenAI client configured for OpenRouter
        model: Model identifier
        puzzle: Puzzle dict with fen, moves, rating, themes
        verbose: Print conversation details

    Returns:
        Result dict with:
        - puzzle_id, rating, themes
        - success: True if ALL moves correct
        - moves_correct: Number of correct moves before failure
        - moves_total: Total moves required
        - error: Error message if API call failed
    """
    moves = puzzle['moves']
    fen = puzzle['fen']

    # Trace collection
    timestamp_start = datetime.now().isoformat()
    turn_traces: list[TurnTrace] = []

    # Apply setup move (opponent's first move) to get the puzzle position
    board = chess.Board(fen)
    try:
        board.push_uci(moves[0])
    except (ValueError, chess.InvalidMoveError) as e:
        timestamp_end = datetime.now().isoformat()
        player_moves_count = len(moves) // 2
        return {
            'puzzle_id': puzzle['puzzle_id'],
            'rating': puzzle['rating'],
            'themes': puzzle['themes'],
            'success': False,
            'moves_correct': 0,
            'moves_total': player_moves_count,
            'error': f"Invalid setup move: {e}",
            'trace': PuzzleTrace(
                puzzle_id=puzzle['puzzle_id'],
                puzzle_rating=puzzle['rating'],
                puzzle_themes=puzzle['themes'],
                initial_fen=fen,
                puzzle_fen=fen,  # Setup move failed, use initial fen
                correct_moves=moves,
                player_moves=[moves[i] for i in range(1, len(moves), 2)],
                model=model,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                success=False,
                moves_correct=0,
                moves_total=player_moves_count,
                error=f"Invalid setup move: {e}",
                turns=[]
            )
        }

    # Player moves are at odd indices (1, 3, 5, ...)
    # Opponent responses are at even indices after setup (2, 4, 6, ...)
    player_moves = [moves[i] for i in range(1, len(moves), 2)]
    opponent_responses = [moves[i] for i in range(2, len(moves), 2)]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # First user message with position
    side = "white" if board.turn else "black"
    puzzle_fen = board.fen()  # FEN after setup move
    first_msg = f"Position ({side} to move): {puzzle_fen}"
    messages.append({"role": "user", "content": first_msg})

    if verbose:
        print(f"  Puzzle {puzzle['puzzle_id']} (rating {puzzle['rating']})")
        print(f"  {first_msg}")

    moves_correct = 0

    for i, expected_move in enumerate(player_moves):
        turn_number = i + 1
        # Get model's response
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=65536,  # Allow for long chain-of-thought reasoning
                temperature=0
            )
            # Check if we got a valid response
            if not response.choices:
                if verbose:
                    print(f"  [Warning: No choices in response]")
                model_move = ""
            else:
                content = response.choices[0].message.content or ""
                model_move = extract_uci_move(content)

                if verbose:
                    # Show truncated response for readability
                    display = content[:200] + "..." if len(content) > 200 else content
                    print(f"  Raw response: {repr(display)}")
                    if not model_move:
                        fr = response.choices[0].finish_reason
                        print(f"  [No UCI move found, finish_reason: {fr}]")

            # Create trace for this turn
            turn_trace = create_turn_trace(
                turn_number=turn_number,
                messages=messages,
                response=response,
                extracted_move=model_move,
                expected_move=expected_move,
                model=model
            )
            turn_traces.append(turn_trace)

        except Exception as e:
            timestamp_end = datetime.now().isoformat()
            return {
                'puzzle_id': puzzle['puzzle_id'],
                'rating': puzzle['rating'],
                'themes': puzzle['themes'],
                'success': False,
                'moves_correct': moves_correct,
                'moves_total': len(player_moves),
                'error': str(e),
                'trace': PuzzleTrace(
                    puzzle_id=puzzle['puzzle_id'],
                    puzzle_rating=puzzle['rating'],
                    puzzle_themes=puzzle['themes'],
                    initial_fen=fen,
                    puzzle_fen=puzzle_fen,
                    correct_moves=moves,
                    player_moves=player_moves,
                    model=model,
                    timestamp_start=timestamp_start,
                    timestamp_end=timestamp_end,
                    success=False,
                    moves_correct=moves_correct,
                    moves_total=len(player_moves),
                    error=str(e),
                    turns=[asdict(t) for t in turn_traces]
                )
            }

        if verbose:
            print(f"  Model: {model_move} (expected: {expected_move})")

        messages.append({"role": "assistant", "content": model_move})

        # Check if move is correct
        if model_move != expected_move.lower():
            timestamp_end = datetime.now().isoformat()
            total_prompt = sum(t.prompt_tokens or 0 for t in turn_traces)
            total_completion = sum(t.completion_tokens or 0 for t in turn_traces)
            total_reasoning = sum(t.reasoning_tokens or 0 for t in turn_traces)
            return {
                'puzzle_id': puzzle['puzzle_id'],
                'rating': puzzle['rating'],
                'themes': puzzle['themes'],
                'success': False,
                'moves_correct': moves_correct,
                'moves_total': len(player_moves),
                'error': None,
                'trace': PuzzleTrace(
                    puzzle_id=puzzle['puzzle_id'],
                    puzzle_rating=puzzle['rating'],
                    puzzle_themes=puzzle['themes'],
                    initial_fen=fen,
                    puzzle_fen=puzzle_fen,
                    correct_moves=moves,
                    player_moves=player_moves,
                    model=model,
                    timestamp_start=timestamp_start,
                    timestamp_end=timestamp_end,
                    success=False,
                    moves_correct=moves_correct,
                    moves_total=len(player_moves),
                    error=None,
                    turns=[asdict(t) for t in turn_traces],
                    total_prompt_tokens=total_prompt,
                    total_completion_tokens=total_completion,
                    total_reasoning_tokens=total_reasoning
                )
            }

        moves_correct += 1

        # Apply the move to the board
        try:
            board.push_uci(expected_move)
        except (ValueError, chess.InvalidMoveError):
            pass  # Move validation already done by matching expected

        # If there's an opponent response, send it
        if i < len(opponent_responses):
            opp_move = opponent_responses[i]
            try:
                board.push_uci(opp_move)
            except (ValueError, chess.InvalidMoveError):
                pass

            next_msg = f"Opponent played: {opp_move}. Your move."
            messages.append({"role": "user", "content": next_msg})

            if verbose:
                print(f"  {next_msg}")

    # All moves correct!
    timestamp_end = datetime.now().isoformat()
    total_prompt = sum(t.prompt_tokens or 0 for t in turn_traces)
    total_completion = sum(t.completion_tokens or 0 for t in turn_traces)
    total_reasoning = sum(t.reasoning_tokens or 0 for t in turn_traces)
    return {
        'puzzle_id': puzzle['puzzle_id'],
        'rating': puzzle['rating'],
        'themes': puzzle['themes'],
        'success': True,
        'moves_correct': moves_correct,
        'moves_total': len(player_moves),
        'error': None,
        'trace': PuzzleTrace(
            puzzle_id=puzzle['puzzle_id'],
            puzzle_rating=puzzle['rating'],
            puzzle_themes=puzzle['themes'],
            initial_fen=fen,
            puzzle_fen=puzzle_fen,
            correct_moves=moves,
            player_moves=player_moves,
            model=model,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            success=True,
            moves_correct=moves_correct,
            moves_total=len(player_moves),
            error=None,
            turns=[asdict(t) for t in turn_traces],
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_reasoning_tokens=total_reasoning
        )
    }


def print_summary(state: BenchmarkState) -> None:
    """Print final benchmark summary."""
    ci = confidence_interval(state.games_played)
    win_rate = state.wins / state.games_played if state.games_played > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {state.model}")
    print(f"Puzzles: {state.games_played}")
    print(f"Win Rate: {win_rate:.1%} ({state.wins}/{state.games_played})")
    print(f"\nOverall ELO: {state.elo_rating:.0f} (+-{ci:.0f} at 95% CI)")

    # Per-tag results (only show tags with enough samples)
    tag_data = []
    for tag in state.tag_games:
        games = state.tag_games[tag]
        if games >= 10:
            rating = state.tag_ratings[tag]
            wins = state.tag_wins[tag]
            tag_ci = confidence_interval(games)
            tag_data.append((tag, rating, tag_ci, wins, games))

    if tag_data:
        print("\nPer-Tag ELO (tags with 10+ puzzles):")
        print("-" * 60)

        # Sort by rating descending
        tag_data.sort(key=lambda x: x[1], reverse=True)

        for tag, rating, tag_ci, wins, games in tag_data[:20]:  # Top 20
            wr = wins / games
            print(f"  {tag:25} {rating:6.0f} (+-{tag_ci:3.0f}) | {wr:4.0%} ({games} games)")


def main():
    parser = argparse.ArgumentParser(
        description="Chess Puzzle Benchmark for LLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --model anthropic/claude-3.5-sonnet
  python benchmark.py --model openai/gpt-4 --puzzles 500
  python benchmark.py --model meta-llama/llama-3-70b --resume
        """
    )
    parser.add_argument(
        "--model", required=True,
        help="Model to evaluate (e.g., anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--puzzles", type=int, default=250,
        help="Number of puzzles to evaluate (default: 250)"
    )
    parser.add_argument(
        "--target-ci", type=int, default=25,
        help="Target confidence interval +/- (default: 25)"
    )
    parser.add_argument(
        "--puzzle-csv", default=DEFAULT_PUZZLE_CSV,
        help="Path to Lichess puzzle CSV file"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory for state files (default: data)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output (show conversation details)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize OpenAI client for OpenRouter
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    # Load or create state
    if args.resume:
        state = load_state(args.model, args.data_dir)
        if state is None:
            print(f"No saved state found for {args.model}, starting fresh")
            state = create_new_state(args.model, INITIAL_ELO, args.seed)
            random.seed(args.seed)
        else:
            print(f"Resuming from {state.games_played} puzzles, ELO: {state.elo_rating:.0f}")
            # Restore random state: re-seed and advance by games already played
            random.seed(state.seed)
            for _ in range(state.games_played):
                random.random()  # Advance RNG to maintain puzzle sequence
    else:
        state = create_new_state(args.model, INITIAL_ELO, args.seed)
        random.seed(args.seed)

    # Load puzzles
    print(f"Loading puzzles from {args.puzzle_csv}...")
    puzzles = load_puzzles(args.puzzle_csv)
    print(f"Loaded {len(puzzles):,} puzzles")

    # Create set of used puzzle IDs for efficient lookup
    used_ids = set(state.used_puzzle_ids)

    # Calculate puzzles needed for target CI
    puzzles_for_ci = puzzles_needed_for_confidence(args.target_ci)
    target_puzzles = max(args.puzzles, puzzles_for_ci)
    remaining = target_puzzles - state.games_played

    print(f"\nModel: {args.model}")
    print(f"Seed: {state.seed}")
    print(f"Target: {target_puzzles} puzzles (for +-{args.target_ci} CI at 95%)")
    print(f"Starting ELO: {state.elo_rating:.0f}")
    if state.games_played > 0:
        print(f"Already completed: {state.games_played} puzzles")
    print("-" * 60)

    try:
        while state.games_played < target_puzzles:
            # Select puzzle near current rating
            puzzle = get_puzzle_near_rating(
                puzzles,
                state.elo_rating,
                RATING_WINDOW,
                used_ids
            )

            if puzzle is None:
                # Expand search window
                puzzle = get_puzzle_near_rating(
                    puzzles,
                    state.elo_rating,
                    RATING_WINDOW * 3,
                    used_ids
                )

            if puzzle is None:
                print("Error: No more unused puzzles available")
                break

            # Evaluate puzzle
            result = evaluate_puzzle(client, args.model, puzzle, args.verbose)

            # Save trace immediately after each puzzle
            if 'trace' in result and result['trace'] is not None:
                append_trace(result['trace'], args.data_dir)
                if args.verbose:
                    trace = result['trace']
                    total_tokens = trace.total_prompt_tokens + trace.total_completion_tokens
                    print(f"  [Trace saved: {len(trace.turns)} turns, {total_tokens} tokens]")

            # Update state
            won = result['success']
            old_elo = state.elo_rating
            state.elo_rating = update_elo(old_elo, puzzle['rating'], won, state.games_played)
            state.games_played += 1
            if won:
                state.wins += 1

            # Update per-tag ELOs
            update_tag_elos(
                state.tag_ratings,
                state.tag_games,
                state.tag_wins,
                puzzle['themes'],
                puzzle['rating'],
                won
            )

            # Track used puzzle
            used_ids.add(puzzle['puzzle_id'])
            state.used_puzzle_ids.append(puzzle['puzzle_id'])
            state.results.append({
                'puzzle_id': puzzle['puzzle_id'],
                'rating': puzzle['rating'],
                'themes': puzzle['themes'],
                'success': won,
                'moves_correct': result['moves_correct'],
                'moves_total': result['moves_total'],
                'elo_before': old_elo,
                'elo_after': state.elo_rating
            })

            # Progress output
            ci = confidence_interval(state.games_played)
            symbol = "+" if won else "-"
            moves_info = f"{result['moves_correct']}/{result['moves_total']}"
            print(
                f"[{state.games_played}/{target_puzzles}] {symbol} "
                f"Puzzle {puzzle['puzzle_id']} (r{puzzle['rating']}) [{moves_info}] | "
                f"ELO: {state.elo_rating:.0f} (+-{ci:.0f})"
            )

            # Save periodically
            if state.games_played % SAVE_INTERVAL == 0:
                save_state(state, args.data_dir)
                if args.verbose:
                    print("  [Saved state]")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving state...")

    finally:
        # Always save state
        save_state(state, args.data_dir)

    # Print final summary
    print_summary(state)


if __name__ == "__main__":
    main()
