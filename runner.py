"""
Episode runner: wires engine + agents + logger into one game loop.

Usage:
    from runner import run_episode
    from agents.scripted import HeuristicAgent, BluffAgent

    log = run_episode(
        agent0=HeuristicAgent(0, "heuristic"),
        agent1=BluffAgent(1, "bluffer"),
        seed=42,
        max_turns=500,
    )
"""
from __future__ import annotations
import random
from typing import Any, Dict, Optional
from pathlib import Path

from engine.card import full_deck, deal
from engine.state import GameState
from engine.game import apply_action
from agents.base import BaseAgent
from logging_.logger import EpisodeLogger

DRAW_REASON = "max_turns_reached"


def run_episode(
    agent0: BaseAgent,
    agent1: BaseAgent,
    seed: int = 0,
    max_turns: int = 500,
    save_path: Optional[Path] = None,
    verbose: bool = False,
) -> EpisodeLogger:
    rng = random.Random(seed)
    deck = full_deck()
    rng.shuffle(deck)
    hands = deal(deck, 2)

    state = GameState.new(hands)
    agents = {0: agent0, 1: agent1}

    for agent in agents.values():
        agent.reset()

    logger = EpisodeLogger(seed=seed)
    logger.agents = [
        {"player_id": a.player_id, "name": a.name, "type": type(a).__name__}
        for a in agents.values()
    ]

    while not state.is_terminal and state.turn < max_turns:
        player = state.current_player
        agent = agents[player]
        obs = state.public_observation(player)

        action = agent.choose_action(obs)
        belief = agent.report_belief(obs)

        new_state, event = apply_action(state, action)

        # Give both players a role-appropriate view of what just happened.
        for pid, ag in agents.items():
            ag.observe_event(_public_event_for(event, pid))

        logger.log_turn(
            turn=state.turn,
            player=player,
            action=action,
            observation_before=obs,
            event=event,
            hand_sizes_after=new_state.hand_sizes(),
            beliefs={str(player): belief} if belief is not None else None,
        )

        if verbose:
            _print_turn(event, new_state)

        state = new_state

    if state.winner is not None:
        logger.log_outcome(winner=state.winner, total_turns=state.turn)
    else:
        logger.log_outcome(winner=-1, total_turns=state.turn, reason=DRAW_REASON)

    if save_path:
        logger.save(save_path)

    return logger


def _public_event_for(event: Dict[str, Any], for_player: int) -> Dict[str, Any]:
    """
    Filter an engine event to what `for_player` can legally observe.

    Play actions:  the actor sees everything; the observer sees only the claim
                   (claimed_rank, n_cards) — never actual_cards or honest.
    Challenge actions: both players see actual_cards (the cards are revealed).
    """
    atype = event["action"]["type"]
    if atype == "challenge":
        return event  # challenge reveal is public

    # play action
    actor = event["player"]
    if for_player == actor:
        return event  # actor knows what they played

    # opponent sees only the public claim
    return {
        "turn": event["turn"],
        "player": actor,
        "action": {"type": "play"},
        "claimed_rank": event["claimed_rank"],
        "n_cards": event["n_cards"],
        # actual_cards and honest are intentionally omitted
    }


def _print_turn(event: dict, state: GameState) -> None:
    t = event["turn"]
    p = event["player"]
    atype = event["action"]["type"]
    if atype == "play":
        honest_flag = "(honest)" if event.get("honest") else "(BLUFF)"
        print(
            f"[T{t:03d}] P{p} plays {event['n_cards']}x {event['claimed_rank']} "
            f"{honest_flag}  actual={event['actual_cards']}  "
            f"hands={state.hand_sizes()}"
        )
    elif atype == "challenge":
        result = event.get("challenge_result", "?")
        pile_to = event.get("pile_goes_to", "?")
        print(
            f"[T{t:03d}] P{p} CHALLENGES → {result}  "
            f"pile→P{pile_to}  hands={state.hand_sizes()}"
        )
    if "winner" in event:
        print(f"  *** P{event['winner']} wins! ***")
