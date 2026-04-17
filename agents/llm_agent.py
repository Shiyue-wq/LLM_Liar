"""
LLM-backed agent using the Anthropic API.

memory_mode="current_only"  — prompt contains only the current observation
memory_mode="full_history"  — prompt also includes every public event so far

The agent asks the LLM to return a JSON block:
{
  "reasoning": "...",
  "action": {"type": "play", "cards": ["A♠", "K♥"], "claimed_rank": "A"}
           | {"type": "challenge"},
  "belief": {"opponent_bluffing_prob": 0.0–1.0, "explanation": "..."}
}

belief is optional; the agent reports it via report_belief() for the logger.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import anthropic

from engine.card import Card, card_from_str, RANKS
from .base import BaseAgent

_RULES = """\
## Game rules — Liar / Cheat (2-player)

A full 52-card deck is shuffled and dealt evenly. Players alternate turns.
Each turn the active player must play 1–4 cards face-down on the pile and \
CLAIM they are all of the current required rank. The required rank cycles \
A → 2 → 3 → … → K → A with every play.

After each play the opponent may either:
  • ACCEPT  — say nothing and become the next player (they will then play the \
next rank).
  • CHALLENGE — call the bluff. Cards are revealed:
      – If the claim was FALSE: the playing player picks up the entire pile.
      – If the claim was TRUE:  the challenging player picks up the entire pile.
After a challenge the player who picked up the pile plays next.

Goal: be the first to empty your hand.

You may play honestly (actual cards match claimed rank) or bluff (any cards, \
but claim the required rank). Strategic bluffing and targeted challenging are \
both valid tactics.
"""

_SYSTEM = (
    _RULES
    + """
## Response format

You must respond with a single JSON block (no other text):
```json
{
  "reasoning": "<your private reasoning, 1–3 sentences>",
  "action": <action object>,
  "belief": {"opponent_bluffing_prob": <0.0–1.0>, "explanation": "<1 sentence>"}
}
```

Action objects:
  Play:      {"type": "play", "cards": ["<card>", ...], "claimed_rank": "<rank>"}
  Challenge: {"type": "challenge"}

Card strings use the format RANK+SUIT, e.g. "A♠", "10♦", "K♣".
You may only play cards that appear in your current hand.
The claimed_rank MUST equal the current required rank.
"""
)


class LLMAgent(BaseAgent):
    """
    Args:
        player_id:   0 or 1
        name:        display name
        model:       Anthropic model string (default: claude-haiku-4-5-20251001)
        memory_mode: "current_only" | "full_history"
        api_key:     overrides ANTHROPIC_API_KEY env var
        temperature: sampling temperature (default 1.0)
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(
        self,
        player_id: int,
        name: str = "llm",
        model: str = DEFAULT_MODEL,
        memory_mode: str = "current_only",
        api_key: Optional[str] = None,
        temperature: float = 1.0,
    ):
        super().__init__(player_id, name)
        if memory_mode not in ("current_only", "full_history"):
            raise ValueError("memory_mode must be 'current_only' or 'full_history'")
        self.model = model
        self.memory_mode = memory_mode
        self.temperature = temperature
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError(
                "LLMAgent requires ANTHROPIC_API_KEY to be set (or pass api_key=)."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._history: List[Dict[str, Any]] = []
        self._last_belief: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._history = []
        self._last_belief = None

    def observe_event(self, public_event: Dict[str, Any]) -> None:
        if self.memory_mode == "full_history":
            self._history.append(public_event)

    def choose_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(obs)
        raw = self._call_api(prompt)
        action, belief = self._parse_response(raw, obs)
        self._last_belief = belief
        return action

    def report_belief(self, obs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return self._last_belief

    # ── Prompt construction ────────────────────────────────────────────────

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        parts: List[str] = []

        if self.memory_mode == "full_history" and self._history:
            parts.append("## Turn history (your perspective)\n")
            for ev in self._history:
                parts.append(_format_event(ev))
            parts.append("")

        parts.append("## Current situation\n")
        parts.append(f"You are Player {obs['current_player']}.")
        parts.append(f"Your hand ({obs['my_hand_size']} cards): {_hand_str(obs['my_hand'])}")
        parts.append(f"Opponent hand size: {obs['opponent_hand_size']}")
        parts.append(f"Pile size: {obs['pile_size']}")
        parts.append(f"Current required rank: {obs['current_rank']}")
        parts.append(f"Turn: {obs['turn']}")

        last = obs["last_claim"]
        if last:
            parts.append(
                f"Last claim: Player {last['player_id']} claimed to play "
                f"{last['n_cards']}× {last['claimed_rank']}. "
                "You may challenge this or play your cards."
            )
        else:
            parts.append("No pending claim. Play your cards.")

        parts.append("\nRespond with JSON only.")
        return "\n".join(parts)

    # ── API call ───────────────────────────────────────────────────────────

    def _call_api(self, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM,
                    "cache_control": {"type": "ephemeral"},  # cache static rules
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(
        self, raw: str, obs: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Extract (action, belief) from the LLM text. Falls back gracefully."""
        data = _extract_json(raw)
        belief = data.get("belief") if isinstance(data, dict) else None

        try:
            action_raw = data["action"]
            if action_raw["type"] == "challenge":
                return {"type": "challenge"}, belief

            if action_raw["type"] == "play":
                card_strs: List[str] = action_raw.get("cards", [])
                cards = _resolve_cards(card_strs, obs["my_hand"])
                claimed = action_raw.get("claimed_rank", obs["current_rank"])
                if claimed != obs["current_rank"]:
                    claimed = obs["current_rank"]  # silently fix wrong rank
                if not cards:
                    cards = _fallback_cards(obs)
                return {"type": "play", "cards": cards, "claimed_rank": claimed}, belief
        except Exception:
            pass

        # Fallback: play one card honestly if possible, else bluff
        return {"type": "play", "cards": _fallback_cards(obs), "claimed_rank": obs["current_rank"]}, belief


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _hand_str(hand: List[Card]) -> str:
    return "[" + ", ".join(str(c) for c in hand) + "]"


def _format_event(ev: Dict[str, Any]) -> str:
    atype = ev.get("action", {}).get("type") or ev.get("action")
    if atype == "play":
        actual = ev.get("actual_cards")
        honest = ev.get("honest")
        if actual is not None:
            # This is your own play event — full info
            tag = "(honest)" if honest else "(BLUFF)"
            return (
                f"T{ev['turn']}: You played {ev['n_cards']}× {ev['claimed_rank']} "
                f"{tag} [actual: {actual}]"
            )
        # Opponent play — only public info visible
        return (
            f"T{ev['turn']}: Opponent claimed {ev['n_cards']}× {ev['claimed_rank']}"
        )
    if atype == "challenge":
        result = ev.get("challenge_result", "?")
        pile_to = ev.get("pile_goes_to", "?")
        revealed = ev.get("actual_cards", [])
        return (
            f"T{ev['turn']}: Player {ev['player']} challenged → {result} "
            f"(actual={revealed}, pile→P{pile_to})"
        )
    return f"T{ev.get('turn', '?')}: {ev}"


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from text (handles ```json ... ``` fences)."""
    # Try fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _resolve_cards(card_strs: List[str], hand: List[Card]) -> List[Card]:
    """Map LLM card string output back to actual Card objects in hand."""
    resolved: List[Card] = []
    remaining = list(hand)
    for s in card_strs:
        try:
            target = card_from_str(s)
        except ValueError:
            continue
        if target in remaining:
            resolved.append(target)
            remaining.remove(target)
    return resolved


def _fallback_cards(obs: Dict[str, Any]) -> List[Card]:
    hand: List[Card] = obs["my_hand"]
    rank = obs["current_rank"]
    matching = [c for c in hand if c.rank == rank]
    return matching[:1] if matching else hand[:1]
