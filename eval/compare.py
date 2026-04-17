"""
Agent comparison script: runs a round-robin match matrix and prints aggregate stats.

Usage:
    python eval/compare.py --episodes 20 --seed 0 --save_dir logs/compare
    python eval/compare.py --matchups heuristic:bluff honest:heuristic --episodes 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from itertools import combinations
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.scripted import HonestAgent, BluffAgent, HeuristicAgent
from agents.llm_agent import LLMAgent
from agents.base import BaseAgent
from runner import run_episode
from eval.metrics import load_episode, aggregate_stats, print_report

AGENT_MAP = {
    "honest":    HonestAgent,
    "bluff":     BluffAgent,
    "heuristic": HeuristicAgent,
    "llm":       LLMAgent,
}

DEFAULT_MATCHUPS: List[Tuple[str, str]] = [
    ("heuristic", "bluff"),
    ("heuristic", "honest"),
    ("honest",    "bluff"),
]


def build_agent(
    name: str,
    player_id: int,
    model: Optional[str] = None,
    memory: str = "full_history",
) -> BaseAgent:
    if name == "llm":
        return LLMAgent(player_id=player_id, name=name,
                        model=model or LLMAgent.DEFAULT_MODEL,
                        memory_mode=memory)
    return AGENT_MAP[name](player_id=player_id, name=name)


def parse_matchups(raw: List[str]) -> List[Tuple[str, str]]:
    result = []
    for s in raw:
        parts = s.split(":")
        if len(parts) != 2 or parts[0] not in AGENT_MAP or parts[1] not in AGENT_MAP:
            raise ValueError(f"Invalid matchup {s!r}. Use agent0:agent1 with known agent names.")
        result.append((parts[0], parts[1]))
    return result


def run_matchup(
    a0_name: str,
    a1_name: str,
    n_episodes: int,
    base_seed: int,
    save_dir: Optional[Path],
    verbose: bool,
    model: Optional[str] = None,
    memory: str = "full_history",
) -> List[dict]:
    logs = []
    wins = {0: 0, 1: 0, -1: 0}
    for ep in range(n_episodes):
        seed = base_seed + ep
        save_path = (
            save_dir / f"{a0_name}_vs_{a1_name}_ep{ep:04d}.json" if save_dir else None
        )
        log = run_episode(
            agent0=build_agent(a0_name, 0, model, memory),
            agent1=build_agent(a1_name, 1, model, memory),
            seed=seed,
            verbose=verbose,
            save_path=save_path,
        )
        logs.append(log.to_dict())
        wins[log.outcome["winner"]] += 1

    print(
        f"  {a0_name:>12} vs {a1_name:<12} │ "
        f"P0 wins: {wins[0]:3d}  P1 wins: {wins[1]:3d}  draws: {wins[-1]:3d}"
    )
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matchups", nargs="*", default=None,
                        help="Pairs like heuristic:bluff. Defaults to all scripted pairs.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default=None,
                        help="Anthropic model ID for llm agents")
    parser.add_argument("--memory", default="full_history",
                        choices=["current_only", "full_history"],
                        help="LLM agent memory mode (default: full_history)")
    args = parser.parse_args()

    matchups = parse_matchups(args.matchups) if args.matchups else DEFAULT_MATCHUPS
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    all_logs: List[dict] = []
    print(f"\nRunning {args.episodes} episodes per matchup\n")
    print(f"{'Matchup':>30}   Results")
    print("─" * 65)

    for a0, a1 in matchups:
        logs = run_matchup(a0, a1, args.episodes, args.seed, save_dir, args.verbose,
                           args.model, args.memory)
        all_logs.extend(logs)

    print("\n── Aggregate stats across all episodes ──────────────────────────")
    agg = aggregate_stats(all_logs)
    print_report(agg)


if __name__ == "__main__":
    main()
