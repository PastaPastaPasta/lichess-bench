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
from typing import Dict, Optional, List
from dataclasses import dataclass, field

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

SYSTEM_PROMPT = """You are solving a chess tactic puzzle.

Return EXACTLY one XML document and nothing else. The XML must have a single root tag <response> and MUST contain these four child tags exactly once each, in this order:
1) <state_summary>
2) <potentials>
3) <solution_forced_line>
4) <solution_single>

Move notation rules:
- Use UCI only: ^[a-h][1-8][a-h][1-8][qrbn]?$ (e.g., e2e4, g1f3, e7e8q).
- Castling must be in UCI king-move form: e1g1, e1c1, e8g8, e8c8.
- Do not use SAN, PGN, algebraic notation, or commentary outside the XML.

<solution_single> MUST contain exactly one UCI move (your committed move for the current position).

<solution_forced_line> MUST be either empty OR a space-separated sequence of UCI moves that begins with the same move as <solution_single>. Provide it only if you believe the line is forcing against best defense. Stop the line at a clear terminal (checkmate or decisive win) or after at most 12 plies.

Evaluation protocol (for you to assume):
- Your committed move (<solution_single>) will be played.
- If your <solution_forced_line> is provided, it may be checked for correctness by the evaluator; if verified, the puzzle may be marked solved immediately.
- Otherwise, the evaluator will provide the opponent's reply and you will continue from the new position.

Now solve the position given by the user."""

# Regex pattern for UCI moves: source square + destination square + optional promotion piece
UCI_MOVE_PATTERN = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', re.IGNORECASE)
# Strict UCI pattern for validation (must match entire string)
UCI_MOVE_STRICT = re.compile(r'^[a-h][1-8][a-h][1-8][qrbn]?$', re.IGNORECASE)


@dataclass
class XMLResponse:
    """Parsed XML response from the model."""
    state_summary: Optional[str] = None
    potentials: Optional[str] = None
    solution_forced_line: Optional[str] = None
    solution_single: Optional[str] = None
    forced_line_moves: List[str] = field(default_factory=list)
    single_move: str = ""
    parse_success: bool = False
    parse_error: Optional[str] = None


def extract_xml_tag(response: str, tag_name: str) -> Optional[str]:
    """Extract content from a single XML tag.

    Args:
        response: Full model response
        tag_name: Name of the XML tag to extract

    Returns:
        Tag content (stripped) or None if not found
    """
    pattern = rf'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_xml_response(response: str) -> XMLResponse:
    """Parse the model's XML-structured response.

    Expects a <response> root tag containing:
    - <state_summary>
    - <potentials>
    - <solution_forced_line>
    - <solution_single>

    Args:
        response: Raw model response text

    Returns:
        XMLResponse with parse_success=True if valid, otherwise parse_error set
    """
    result = XMLResponse()

    if not response:
        result.parse_error = "Empty response"
        return result

    # Check for <response> root tag
    response_content = extract_xml_tag(response, 'response')
    if response_content is None:
        result.parse_error = "Missing <response> root tag"
        return result

    # Extract child tags from within <response>
    result.state_summary = extract_xml_tag(response_content, 'state_summary')
    result.potentials = extract_xml_tag(response_content, 'potentials')
    result.solution_forced_line = extract_xml_tag(response_content, 'solution_forced_line')
    result.solution_single = extract_xml_tag(response_content, 'solution_single')

    # Validate required <solution_single>
    if result.solution_single is None:
        result.parse_error = "Missing <solution_single> tag"
        return result

    # Parse and validate single move
    single = result.solution_single.strip().lower()
    if not UCI_MOVE_STRICT.match(single):
        result.parse_error = f"Invalid UCI move in <solution_single>: {single}"
        return result
    result.single_move = single

    # Parse forced line (if non-empty)
    if result.solution_forced_line and result.solution_forced_line.strip():
        moves = result.solution_forced_line.strip().split()
        for m in moves:
            if not UCI_MOVE_STRICT.match(m):
                result.parse_error = f"Invalid UCI move in forced line: {m}"
                return result
        result.forced_line_moves = [m.lower() for m in moves]

        # Validate forced line starts with single move
        if result.forced_line_moves and result.forced_line_moves[0] != result.single_move:
            result.parse_error = "Forced line must start with <solution_single> move"
            return result

    result.parse_success = True
    return result


def check_forced_line_match(forced_line: List[str], remaining_player_moves: List[str]) -> bool:
    """Check if the model's forced line matches the remaining puzzle solution.

    All-or-nothing: the player moves in forced line must match EXACTLY all remaining player moves.
    The forced_line may contain alternating player and opponent moves (full line),
    so we extract only the player moves (indices 0, 2, 4, ...) for comparison.

    Args:
        forced_line: List of UCI moves from the model's <solution_forced_line>
                     May be full line (player + opponent moves) or just player moves
        remaining_player_moves: List of expected player moves remaining in puzzle

    Returns:
        True if player moves in forced_line match exactly, False otherwise
    """
    if not forced_line or not remaining_player_moves:
        return False

    # Extract player moves from forced line (indices 0, 2, 4, ...)
    # The model may provide full line with opponent responses interleaved
    player_moves_in_line = [forced_line[i] for i in range(0, len(forced_line), 2)]

    # Must have exactly the right number of player moves
    if len(player_moves_in_line) != len(remaining_player_moves):
        return False

    # All player moves must match
    return all(f.lower() == e.lower() for f, e in zip(player_moves_in_line, remaining_player_moves))


@dataclass
class ValidationResult:
    """Result of validating a model's response for legal chess moves."""
    valid: bool = False
    error: Optional[str] = None


def validate_response_moves(
    board: chess.Board,
    xml_result: 'XMLResponse',
    opponent_responses: List[str],
    current_move_index: int
) -> ValidationResult:
    """Validate that all moves in the response are legal chess moves.

    Checks:
    1. The single move is legal on the current board
    2. If a forced line is provided, all moves in the sequence are legal
       The forced line may contain alternating player and opponent moves,
       so we validate each move on the current board state.

    Args:
        board: Current board state
        xml_result: Parsed XML response
        opponent_responses: List of opponent responses from puzzle (unused, kept for API compat)
        current_move_index: Current index in player_moves (unused, kept for API compat)

    Returns:
        ValidationResult with valid=True if all moves are legal
    """
    if not xml_result.parse_success:
        return ValidationResult(valid=False, error=xml_result.parse_error)

    # Check single move is legal
    try:
        move = chess.Move.from_uci(xml_result.single_move)
        if move not in board.legal_moves:
            return ValidationResult(
                valid=False,
                error=f"Single move {xml_result.single_move} is not legal in this position"
            )
    except (ValueError, chess.InvalidMoveError) as e:
        return ValidationResult(
            valid=False,
            error=f"Invalid single move format: {xml_result.single_move} - {e}"
        )

    # If no forced line, we're done
    if not xml_result.forced_line_moves:
        return ValidationResult(valid=True)

    # Validate forced line by playing through moves on a copy of the board
    # The forced line may contain alternating player and opponent moves,
    # so we just validate each move is legal on the current board state
    test_board = board.copy()

    for i, move_str in enumerate(xml_result.forced_line_moves):
        try:
            move = chess.Move.from_uci(move_str)
            if move not in test_board.legal_moves:
                return ValidationResult(
                    valid=False,
                    error=f"Forced line move {i+1} ({move_str}) is not legal"
                )
            test_board.push(move)

        except (ValueError, chess.InvalidMoveError) as e:
            return ValidationResult(
                valid=False,
                error=f"Invalid move format in forced line: {move_str} - {e}"
            )

    return ValidationResult(valid=True)


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
            'first_shot_success': False,
            'valid_response': False,  # No response yet, treat as invalid
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
                first_shot_attempted=False,
                first_shot_success=False,
                first_shot_moves=None,
                all_responses_valid=False,
                first_invalid_turn=None,
                validity_error=f"Invalid setup move: {e}",
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
    first_shot_attempted = False
    first_shot_success = False
    first_shot_moves: Optional[List[str]] = None

    # Valid response tracking
    all_responses_valid = True
    first_invalid_turn: Optional[int] = None
    first_validity_error: Optional[str] = None

    for i, expected_move in enumerate(player_moves):
        turn_number = i + 1
        remaining_player_moves = player_moves[i:]  # All moves from this point forward

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
                xml_result = XMLResponse(parse_error="No response choices")
            else:
                content = response.choices[0].message.content or ""

                # Parse XML response (strict - no fallback)
                xml_result = parse_xml_response(content)

                if not xml_result.parse_success:
                    model_move = ""
                    if verbose:
                        print(f"  [XML Parse Error: {xml_result.parse_error}]")
                        display = content[:200] + "..." if len(content) > 200 else content
                        print(f"  Raw response: {repr(display)}")
                else:
                    model_move = xml_result.single_move

                    if verbose:
                        print(f"  State: {xml_result.state_summary[:80]}..." if xml_result.state_summary and len(xml_result.state_summary) > 80 else f"  State: {xml_result.state_summary}")
                        if xml_result.forced_line_moves:
                            print(f"  Forced line: {' '.join(xml_result.forced_line_moves)}")

            # Validate moves are legal chess
            validation = validate_response_moves(board, xml_result, opponent_responses, i)
            if not validation.valid and all_responses_valid:
                all_responses_valid = False
                first_invalid_turn = turn_number
                first_validity_error = validation.error
                if verbose:
                    print(f"  [Validation Error: {validation.error}]")

            # Check for forced line match on FIRST turn only
            if turn_number == 1 and xml_result.forced_line_moves:
                first_shot_attempted = True
                first_shot_moves = xml_result.forced_line_moves

                if check_forced_line_match(xml_result.forced_line_moves, remaining_player_moves):
                    first_shot_success = True
                    if verbose:
                        print(f"  [FIRST-SHOT SUCCESS: Complete forcing line matched!]")

                    # Early exit - puzzle solved completely
                    timestamp_end = datetime.now().isoformat()

                    # Create trace for this turn with XML data
                    turn_trace = create_turn_trace(
                        turn_number=turn_number,
                        messages=messages,
                        response=response,
                        extracted_move=model_move,
                        expected_move=expected_move,
                        model=model
                    )
                    # Add XML fields to trace
                    turn_trace.xml_state_summary = xml_result.state_summary
                    turn_trace.xml_potentials = xml_result.potentials
                    turn_trace.xml_forced_line = xml_result.solution_forced_line
                    turn_trace.xml_single_move = xml_result.solution_single
                    turn_trace.xml_parse_success = xml_result.parse_success
                    turn_trace.move_correct = True
                    turn_trace.response_valid = validation.valid
                    turn_trace.validity_error = validation.error
                    turn_traces.append(turn_trace)

                    total_prompt = sum(t.prompt_tokens or 0 for t in turn_traces)
                    total_completion = sum(t.completion_tokens or 0 for t in turn_traces)
                    total_reasoning = sum(t.reasoning_tokens or 0 for t in turn_traces)

                    return {
                        'puzzle_id': puzzle['puzzle_id'],
                        'rating': puzzle['rating'],
                        'themes': puzzle['themes'],
                        'success': True,
                        'first_shot_success': True,
                        'valid_response': validation.valid,
                        'moves_correct': len(player_moves),
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
                            moves_correct=len(player_moves),
                            moves_total=len(player_moves),
                            error=None,
                            first_shot_attempted=True,
                            first_shot_success=True,
                            first_shot_moves=first_shot_moves,
                            all_responses_valid=validation.valid,
                            first_invalid_turn=None if validation.valid else 1,
                            validity_error=validation.error,
                            turns=[asdict(t) for t in turn_traces],
                            total_prompt_tokens=total_prompt,
                            total_completion_tokens=total_completion,
                            total_reasoning_tokens=total_reasoning
                        )
                    }

            # Create trace for this turn with XML data
            turn_trace = create_turn_trace(
                turn_number=turn_number,
                messages=messages,
                response=response,
                extracted_move=model_move,
                expected_move=expected_move,
                model=model
            )
            # Add XML fields to trace
            turn_trace.xml_state_summary = xml_result.state_summary
            turn_trace.xml_potentials = xml_result.potentials
            turn_trace.xml_forced_line = xml_result.solution_forced_line
            turn_trace.xml_single_move = xml_result.solution_single
            turn_trace.xml_parse_success = xml_result.parse_success
            turn_trace.xml_parse_error = xml_result.parse_error
            turn_trace.response_valid = validation.valid
            turn_trace.validity_error = validation.error
            turn_traces.append(turn_trace)

        except Exception as e:
            timestamp_end = datetime.now().isoformat()
            return {
                'puzzle_id': puzzle['puzzle_id'],
                'rating': puzzle['rating'],
                'themes': puzzle['themes'],
                'success': False,
                'first_shot_success': False,
                'valid_response': all_responses_valid,
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
                    first_shot_attempted=first_shot_attempted,
                    first_shot_success=False,
                    first_shot_moves=first_shot_moves,
                    all_responses_valid=all_responses_valid,
                    first_invalid_turn=first_invalid_turn,
                    validity_error=first_validity_error,
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
                'first_shot_success': False,
                'valid_response': all_responses_valid,
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
                    first_shot_attempted=first_shot_attempted,
                    first_shot_success=False,
                    first_shot_moves=first_shot_moves,
                    all_responses_valid=all_responses_valid,
                    first_invalid_turn=first_invalid_turn,
                    validity_error=first_validity_error,
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

    # All moves correct (via multi-turn, not first-shot)!
    timestamp_end = datetime.now().isoformat()
    total_prompt = sum(t.prompt_tokens or 0 for t in turn_traces)
    total_completion = sum(t.completion_tokens or 0 for t in turn_traces)
    total_reasoning = sum(t.reasoning_tokens or 0 for t in turn_traces)
    return {
        'puzzle_id': puzzle['puzzle_id'],
        'rating': puzzle['rating'],
        'themes': puzzle['themes'],
        'success': True,
        'first_shot_success': False,  # Solved via multi-turn, not first-shot
        'valid_response': all_responses_valid,
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
            first_shot_attempted=first_shot_attempted,
            first_shot_success=False,
            first_shot_moves=first_shot_moves,
            all_responses_valid=all_responses_valid,
            first_invalid_turn=first_invalid_turn,
            validity_error=first_validity_error,
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

    # First-shot metrics
    fs_ci = confidence_interval(state.first_shot_games) if state.first_shot_games > 0 else float('inf')
    fs_win_rate = state.first_shot_wins / state.first_shot_games if state.first_shot_games > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {state.model}")
    print(f"Puzzles: {state.games_played}")
    print(f"Win Rate: {win_rate:.1%} ({state.wins}/{state.games_played})")
    print(f"\nOverall ELO: {state.elo_rating:.0f} (+-{ci:.0f} at 95% CI)")

    print(f"\n--- First-Shot Performance ---")
    print(f"First-Shot Win Rate: {fs_win_rate:.1%} ({state.first_shot_wins}/{state.first_shot_games})")
    if fs_ci != float('inf'):
        print(f"First-Shot ELO: {state.first_shot_elo:.0f} (+-{fs_ci:.0f} at 95% CI)")
    else:
        print(f"First-Shot ELO: {state.first_shot_elo:.0f} (insufficient data for CI)")

    # Valid response metrics
    vr_ci = confidence_interval(state.valid_response_games) if state.valid_response_games > 0 else float('inf')
    vr_win_rate = state.valid_response_wins / state.valid_response_games if state.valid_response_games > 0 else 0

    print(f"\n--- Valid Response Performance ---")
    print(f"Valid Response Rate: {vr_win_rate:.1%} ({state.valid_response_wins}/{state.valid_response_games})")
    if vr_ci != float('inf'):
        print(f"Valid Response ELO: {state.valid_response_elo:.0f} (+-{vr_ci:.0f} at 95% CI)")
    else:
        print(f"Valid Response ELO: {state.valid_response_elo:.0f} (insufficient data for CI)")

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
            first_shot_won = result.get('first_shot_success', False)
            valid_response = result.get('valid_response', False)

            old_elo = state.elo_rating
            state.elo_rating = update_elo(old_elo, puzzle['rating'], won, state.games_played)
            state.games_played += 1
            if won:
                state.wins += 1

            # Update first-shot ELO (always track, regardless of attempt)
            old_first_shot_elo = state.first_shot_elo
            state.first_shot_elo = update_elo(
                old_first_shot_elo,
                puzzle['rating'],
                first_shot_won,
                state.first_shot_games
            )
            state.first_shot_games += 1
            if first_shot_won:
                state.first_shot_wins += 1

            # Update valid response ELO (tracks if model follows format + legal moves)
            old_valid_response_elo = state.valid_response_elo
            state.valid_response_elo = update_elo(
                old_valid_response_elo,
                puzzle['rating'],
                valid_response,
                state.valid_response_games
            )
            state.valid_response_games += 1
            if valid_response:
                state.valid_response_wins += 1

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
                'first_shot_success': first_shot_won,
                'valid_response': valid_response,
                'moves_correct': result['moves_correct'],
                'moves_total': result['moves_total'],
                'elo_before': old_elo,
                'elo_after': state.elo_rating,
                'first_shot_elo_before': old_first_shot_elo,
                'first_shot_elo_after': state.first_shot_elo,
                'valid_response_elo_before': old_valid_response_elo,
                'valid_response_elo_after': state.valid_response_elo
            })

            # Progress output
            ci = confidence_interval(state.games_played)
            symbol = "+" if won else "-"
            fs_symbol = "F" if first_shot_won else ""
            vr_symbol = "V" if valid_response else "X"
            moves_info = f"{result['moves_correct']}/{result['moves_total']}"
            print(
                f"[{state.games_played}/{target_puzzles}] {symbol}{fs_symbol}{vr_symbol} "
                f"Puzzle {puzzle['puzzle_id']} (r{puzzle['rating']}) [{moves_info}] | "
                f"ELO: {state.elo_rating:.0f} (+-{ci:.0f}) | 1st: {state.first_shot_elo:.0f} | Valid: {state.valid_response_elo:.0f}"
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
