"""Trace persistence for saving full model responses and thinking tokens."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class TurnTrace:
    """Trace data for a single API call (one turn in a multi-turn puzzle)."""
    turn_number: int                           # 1-indexed turn within puzzle
    timestamp: str                             # ISO format timestamp

    # Prompt information
    messages_sent: List[Dict[str, str]]        # Full message history sent to model

    # Response content
    content: str                               # response.choices[0].message.content
    reasoning: Optional[str] = None            # OpenAI o1/o3: message.reasoning
    reasoning_details: Optional[List[Dict]] = None  # OpenRouter structured reasoning

    # Extracted data
    extracted_move: str = ""                   # UCI move extracted via regex
    expected_move: str = ""                    # Correct move for this turn
    move_correct: bool = False

    # Response metadata
    finish_reason: Optional[str] = None
    model_id: Optional[str] = None             # Model that actually responded

    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None     # Reasoning/thinking tokens if available
    total_tokens: Optional[int] = None


@dataclass
class PuzzleTrace:
    """Complete trace for an entire puzzle (may have multiple turns)."""
    # Puzzle identification
    puzzle_id: str
    puzzle_rating: int
    puzzle_themes: List[str]

    # Puzzle context
    initial_fen: str                           # Original FEN before setup move
    puzzle_fen: str                            # FEN after opponent's setup move
    correct_moves: List[str]                   # Full solution move list
    player_moves: List[str]                    # Just the player's expected moves

    # Model information
    model: str
    timestamp_start: str
    timestamp_end: str

    # Result summary
    success: bool
    moves_correct: int
    moves_total: int
    error: Optional[str] = None

    # Turn-by-turn traces
    turns: List[Dict] = field(default_factory=list)  # Serialized TurnTrace objects

    # Aggregate token usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0


def trace_file_path(model: str, data_dir: str = "data") -> Path:
    """Get trace file path for a model.

    Creates filenames like: openai_gpt-4_traces.jsonl
    """
    safe_name = model.replace("/", "_").replace(":", "_")
    return Path(data_dir) / f"{safe_name}_traces.jsonl"


def append_trace(trace: PuzzleTrace, data_dir: str = "data") -> None:
    """Append a single puzzle trace to the JSONL file.

    Creates the file and directory if they don't exist.
    """
    path = trace_file_path(trace.model, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'a', encoding='utf-8') as f:
        json.dump(asdict(trace), f, ensure_ascii=False)
        f.write('\n')


def extract_reasoning_from_response(response, model: str) -> Dict[str, Any]:
    """Extract reasoning/thinking tokens from API response based on model type.

    Handles different formats:
    - OpenAI o1/o3: response.choices[0].message.reasoning
    - OpenRouter reasoning_details: response.choices[0].message.reasoning_details
    - Claude thinking blocks: embedded in content or separate blocks

    Returns dict with:
    - reasoning: str or None
    - reasoning_details: list or None
    - reasoning_tokens: int or None
    """
    result = {
        'reasoning': None,
        'reasoning_details': None,
        'reasoning_tokens': None
    }

    if not response.choices:
        return result

    message = response.choices[0].message

    # OpenAI o1/o3 models: check for reasoning attribute
    if hasattr(message, 'reasoning') and message.reasoning:
        result['reasoning'] = message.reasoning

    # OpenRouter unified format: reasoning_details array
    if hasattr(message, 'reasoning_details') and message.reasoning_details:
        try:
            result['reasoning_details'] = [
                {'type': getattr(rd, 'type', None), 'content': getattr(rd, 'content', None)}
                for rd in message.reasoning_details
            ]
        except (TypeError, AttributeError):
            # Handle if reasoning_details is already a list of dicts
            if isinstance(message.reasoning_details, list):
                result['reasoning_details'] = message.reasoning_details

    # Extract reasoning token count from usage
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        # OpenAI format: completion_tokens_details.reasoning_tokens
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            details = usage.completion_tokens_details
            if hasattr(details, 'reasoning_tokens'):
                result['reasoning_tokens'] = details.reasoning_tokens
        # OpenRouter format: direct reasoning_tokens
        if hasattr(usage, 'reasoning_tokens') and usage.reasoning_tokens:
            result['reasoning_tokens'] = usage.reasoning_tokens

    return result


def extract_usage_from_response(response) -> Dict[str, Optional[int]]:
    """Extract token usage from API response.

    Returns dict with prompt_tokens, completion_tokens, total_tokens.
    """
    result = {
        'prompt_tokens': None,
        'completion_tokens': None,
        'total_tokens': None
    }

    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        result['prompt_tokens'] = getattr(usage, 'prompt_tokens', None)
        result['completion_tokens'] = getattr(usage, 'completion_tokens', None)
        result['total_tokens'] = getattr(usage, 'total_tokens', None)

    return result


def create_turn_trace(
    turn_number: int,
    messages: List[Dict[str, str]],
    response,
    extracted_move: str,
    expected_move: str,
    model: str
) -> TurnTrace:
    """Create a TurnTrace from an API response.

    Consolidates all response data extraction in one place.
    """
    content = ""
    finish_reason = None
    model_id = None

    if response.choices:
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason

    if hasattr(response, 'model'):
        model_id = response.model

    reasoning_data = extract_reasoning_from_response(response, model)
    usage_data = extract_usage_from_response(response)

    return TurnTrace(
        turn_number=turn_number,
        timestamp=datetime.now().isoformat(),
        messages_sent=[msg.copy() for msg in messages],  # Copy to preserve state
        content=content,
        reasoning=reasoning_data['reasoning'],
        reasoning_details=reasoning_data['reasoning_details'],
        extracted_move=extracted_move,
        expected_move=expected_move,
        move_correct=(extracted_move == expected_move.lower()),
        finish_reason=finish_reason,
        model_id=model_id,
        prompt_tokens=usage_data['prompt_tokens'],
        completion_tokens=usage_data['completion_tokens'],
        reasoning_tokens=reasoning_data['reasoning_tokens'],
        total_tokens=usage_data['total_tokens']
    )


def load_traces(model: str, data_dir: str = "data") -> List[PuzzleTrace]:
    """Load all traces for a model from JSONL file.

    Useful for analysis and debugging.
    """
    path = trace_file_path(model, data_dir)
    if not path.exists():
        return []

    traces = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                traces.append(PuzzleTrace(**data))
    return traces
