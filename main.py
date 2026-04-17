"""
Entry point: run N episodes between two scripted agents and print a summary.

    python main.py --agent0 heuristic --agent1 bluff --episodes 20 --seed 0 --verbose
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from collections import Counter

# make project root importable
sys.path.insert(0, str(Path(__file__).parent))

from agents.scripted import HonestAgent, BluffAgent, HeuristicAgent
from agents.llm_agent import LLMAgent
from runner import run_episode

AGENT_MAP = {
    "honest": HonestAgent,
    "bluff": BluffAgent,
    "heuristic": HeuristicAgent,
    "llm": LLMAgent,
}


def build_agent(name: str, player_id: int, model: str | None = None, memory: str = "full_history"):
    if name == "llm":
        return LLMAgent(player_id=player_id, name=name,
                        model=model or LLMAgent.DEFAULT_MODEL,
                        memory_mode=memory)
    cls = AGENT_MAP[name]
    return cls(player_id=player_id, name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent0", default="heuristic", choices=AGENT_MAP)
    parser.add_argument("--agent1", default="bluff", choices=AGENT_MAP)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--model", default=None,
                        help="Anthropic model ID for llm agents (default: LLMAgent.DEFAULT_MODEL)")
    parser.add_argument("--memory", default="full_history",
                        choices=["current_only", "full_history"],
                        help="LLM agent memory mode (default: full_history)")
    args = parser.parse_args()

    wins = Counter()
    total_turns = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        save_path = (
            Path(args.save_dir) / f"ep_{ep:04d}.json" if args.save_dir else None
        )
        log = run_episode(
            agent0=build_agent(args.agent0, 0, args.model, args.memory),
            agent1=build_agent(args.agent1, 1, args.model, args.memory),
            seed=seed,
            verbose=args.verbose,
            save_path=save_path,
        )
        outcome = log.outcome
        winner = outcome["winner"]
        turns = outcome["total_turns"]
        wins[winner] += 1
        total_turns.append(turns)
        print(
            f"Episode {ep:3d} | seed={seed} | "
            f"winner=P{winner} ({args.agent0 if winner==0 else args.agent1}) | "
            f"turns={turns}"
        )

    print("\n── Summary ──────────────────────────────")
    print(f"  P0 ({args.agent0}) wins : {wins[0]} / {args.episodes}")
    print(f"  P1 ({args.agent1}) wins : {wins[1]} / {args.episodes}")
    print(f"  Draws               : {wins[-1]} / {args.episodes}")
    print(f"  Avg turns           : {sum(total_turns)/len(total_turns):.1f}")


if __name__ == "__main__":
    main()
