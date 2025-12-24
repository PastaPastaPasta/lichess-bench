#!/usr/bin/env python3
"""
Generate an SVG leaderboard table from benchmark state files.

Reads all *_state.json files from the data/ directory and generates
a styled SVG table showing model rankings by ELO rating.

Usage:
    python generate_table.py
    python generate_table.py --data-dir data --output leaderboard.svg
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from elo import confidence_interval, is_low_confidence


def load_all_states(data_dir: str = "data") -> List[Dict]:
    """Load all state files and extract model summary data."""
    data_path = Path(data_dir)
    models = []

    for state_file in data_path.glob("*_state.json"):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            model_name = state.get("model", "Unknown")
            elo = state.get("elo_rating", 1500)
            games = state.get("games_played", 0)
            wins = state.get("wins", 0)
            ci = confidence_interval(games)
            win_rate = (wins / games * 100) if games > 0 else 0
            low_conf = is_low_confidence(games, ci)

            models.append({
                "name": model_name,
                "elo": elo,
                "games": games,
                "wins": wins,
                "ci": ci,
                "win_rate": win_rate,
                "low_confidence": low_conf
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {state_file}: {e}")
            continue

    # Sort by ELO descending
    models.sort(key=lambda x: x["elo"], reverse=True)
    return models


def format_model_name(model: str) -> str:
    """Clean up model names for display."""
    # Remove provider prefix
    if "/" in model:
        model = model.split("/", 1)[1]

    # Capitalize and clean up
    return model.replace("-", " ").replace("_", " ").title()


def generate_svg(models: List[Dict], output_path: str = "leaderboard.svg") -> None:
    """Generate styled SVG table."""
    if not models:
        print("No models found to generate table.")
        return

    # Check if any models have low confidence
    has_low_confidence = any(m.get("low_confidence", False) for m in models)
    footnote_height = 30 if has_low_confidence else 0

    # SVG dimensions
    row_height = 36
    header_height = 44
    padding = 20
    width = 700
    height = header_height + (len(models) * row_height) + padding + footnote_height

    # Column positions
    col_rank = 25
    col_model = 70
    col_elo = 380
    col_ci = 460
    col_games = 560
    col_winrate = 640

    svg_parts = []

    # SVG header
    svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      .bg {{ fill: #ffffff; }}
      .header-bg {{ fill: #1a1a2e; }}
      .row-even {{ fill: #f8f9fa; }}
      .row-odd {{ fill: #ffffff; }}
      .header-text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; font-weight: 600; fill: #ffffff; }}
      .rank {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 14px; font-weight: 700; fill: #6c757d; }}
      .rank-top {{ fill: #ffc107; }}
      .model {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 14px; font-weight: 500; fill: #212529; }}
      .elo {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 14px; font-weight: 700; fill: #0d6efd; }}
      .ci {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 12px; fill: #6c757d; }}
      .games {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; fill: #495057; }}
      .winrate {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 13px; fill: #198754; }}
      .border {{ stroke: #dee2e6; stroke-width: 1; fill: none; }}
      .footnote {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; font-size: 11px; fill: #6c757d; font-style: italic; }}
    </style>
  </defs>

  <!-- Background -->
  <rect class="bg" width="{width}" height="{height}" rx="8"/>
  <rect class="border" x="0.5" y="0.5" width="{width-1}" height="{height-1}" rx="8"/>

  <!-- Header -->
  <rect class="header-bg" width="{width}" height="{header_height}" rx="8"/>
  <rect class="header-bg" y="8" width="{width}" height="{header_height - 8}"/>

  <text x="{col_rank}" y="{header_height/2 + 5}" class="header-text">#</text>
  <text x="{col_model}" y="{header_height/2 + 5}" class="header-text">Model</text>
  <text x="{col_elo}" y="{header_height/2 + 5}" class="header-text">ELO</text>
  <text x="{col_ci}" y="{header_height/2 + 5}" class="header-text">95% CI</text>
  <text x="{col_games}" y="{header_height/2 + 5}" class="header-text">Games</text>
  <text x="{col_winrate}" y="{header_height/2 + 5}" class="header-text">Win%</text>
''')

    # Data rows
    for i, model in enumerate(models):
        y_offset = header_height + (i * row_height)
        text_y = y_offset + (row_height / 2) + 5

        row_class = "row-even" if i % 2 == 0 else "row-odd"
        rank_class = "rank rank-top" if i < 3 else "rank"

        # Format values
        display_name = format_model_name(model["name"])
        if model.get("low_confidence", False):
            display_name += " †"
        elo_str = f"{model['elo']:.0f}"
        ci_str = f"±{model['ci']:.0f}"
        games_str = str(model["games"])
        winrate_str = f"{model['win_rate']:.1f}%"

        svg_parts.append(f'''
  <!-- Row {i + 1} -->
  <rect class="{row_class}" y="{y_offset}" width="{width}" height="{row_height}"/>
  <text x="{col_rank}" y="{text_y}" class="{rank_class}">{i + 1}</text>
  <text x="{col_model}" y="{text_y}" class="model">{display_name}</text>
  <text x="{col_elo}" y="{text_y}" class="elo">{elo_str}</text>
  <text x="{col_ci}" y="{text_y}" class="ci">{ci_str}</text>
  <text x="{col_games}" y="{text_y}" class="games">{games_str}</text>
  <text x="{col_winrate}" y="{text_y}" class="winrate">{winrate_str}</text>''')

    # Add footnote if any models have low confidence
    if has_low_confidence:
        footnote_y = header_height + (len(models) * row_height) + 20
        svg_parts.append(f'''
  <!-- Footnote -->
  <text x="{col_model}" y="{footnote_y}" class="footnote">† Low confidence: fewer than 30 games or 95% CI exceeds ±100 ELO</text>''')

    svg_parts.append("\n</svg>")

    # Write to file
    svg_content = "".join(svg_parts)
    with open(output_path, 'w') as f:
        f.write(svg_content)

    print(f"Generated {output_path} with {len(models)} models")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SVG leaderboard from benchmark state files"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing state files (default: data)"
    )
    parser.add_argument(
        "--output",
        default="leaderboard.svg",
        help="Output SVG file path (default: leaderboard.svg)"
    )
    args = parser.parse_args()

    models = load_all_states(args.data_dir)
    generate_svg(models, args.output)


if __name__ == "__main__":
    main()
