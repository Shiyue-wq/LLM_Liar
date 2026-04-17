"""
Evaluation metrics computed from episode log files.

Per-player stats returned by episode_stats():
    wins              int    — 1 if this player won, else 0
    bluff_rate        float  — fraction of play turns that were bluffs
    honest_rate       float  — 1 - bluff_rate
    challenge_rate    float  — fraction of turns where player challenged when they could
    challenge_acc     float  — fraction of challenges that were correct (opponent was bluffing)
    bluff_caught_rate float  — fraction of own bluffs that were caught
    total_play_turns  int
    total_challenges  int
    correct_challenges int
    total_turns       int

aggregate_stats(episodes) averages numeric fields across a list of episode stat dicts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_episode(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def episode_stats(log: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return per-player stats for a single episode log dict."""
    players = {a["player_id"]: a["name"] for a in log["agents"]}
    winner = (log["outcome"] or {}).get("winner", -1)

    # Accumulators
    play_turns:       Dict[int, int] = {p: 0 for p in players}
    bluff_turns:      Dict[int, int] = {p: 0 for p in players}
    challengeable:    Dict[int, int] = {p: 0 for p in players}  # turns player could have challenged
    challenged:       Dict[int, int] = {p: 0 for p in players}
    correct_ch:       Dict[int, int] = {p: 0 for p in players}
    bluffs_caught:    Dict[int, int] = {p: 0 for p in players}

    for turn_rec in log["turns"]:
        player  = turn_rec["player"]
        event   = turn_rec["event"]
        atype   = turn_rec["action"]["type"]

        if atype == "play":
            play_turns[player] += 1
            if not event.get("honest", True):
                bluff_turns[player] += 1

            # The *other* player could have challenged after this play.
            opponent = 1 - player
            challengeable[opponent] += 1

        elif atype == "challenge":
            challenged[player] += 1
            result = event.get("challenge_result", "")
            if result == "caught_bluffing":
                correct_ch[player] += 1
                # The bluffer is the player who played before the challenge
                bluffer = event.get("pile_goes_to")
                if bluffer is not None:
                    bluffs_caught[bluffer] += 1

    stats: Dict[int, Dict[str, Any]] = {}
    for pid, name in players.items():
        pt  = play_turns[pid]
        ch  = challenged[pid]
        cch = correct_ch[pid]
        can = challengeable[pid]
        bc  = bluffs_caught[pid]
        bt  = bluff_turns[pid]

        stats[pid] = {
            "name":               name,
            "player_id":          pid,
            "win_rate":           int(winner == pid),
            "total_play_turns":   pt,
            "bluff_turns":        bt,
            "bluff_rate":         bt / pt if pt else 0.0,
            "honest_rate":        1.0 - (bt / pt if pt else 0.0),
            "total_challenges":   ch,
            "challengeable_turns": can,
            "challenge_rate":     ch / can if can else 0.0,
            "correct_challenges": cch,
            "challenge_acc":      cch / ch if ch else 0.0,
            "bluffs_caught":      bc,
            "bluff_caught_rate":  bc / bt if bt else 0.0,
            "total_turns":        log["outcome"]["total_turns"],
        }
    return stats


def aggregate_stats(
    episodes: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Average per-player numeric stats across a list of episode log dicts.
    Returns {agent_name: {metric: mean_value}}.
    """
    from collections import defaultdict

    accumulator: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for log in episodes:
        per_player = episode_stats(log)
        for pid, s in per_player.items():
            name = s["name"]
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    accumulator[name][k].append(v)

    result: Dict[str, Dict[str, float]] = {}
    for name, metrics in accumulator.items():
        result[name] = {k: sum(vs) / len(vs) for k, vs in metrics.items()}
    return result


def print_report(agg: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print an aggregate stats dict."""
    fields = [
        ("win_rate",         "Win rate"),
        ("bluff_rate",       "Bluff rate"),
        ("challenge_rate",   "Challenge rate"),
        ("challenge_acc",    "Challenge accuracy"),
        ("bluff_caught_rate","Bluff caught rate"),
        ("total_turns",      "Avg turns/game"),
    ]
    name_w = max(len(n) for n in agg) + 2
    header = f"{'Agent':<{name_w}}" + "".join(f"{label:>22}" for _, label in fields)
    print(header)
    print("─" * len(header))
    for agent_name, stats in sorted(agg.items()):
        row = f"{agent_name:<{name_w}}"
        for key, _ in fields:
            val = stats.get(key, float("nan"))
            row += f"{val:>22.3f}"
        print(row)
