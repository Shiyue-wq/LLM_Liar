# Liar Game — LLM Agent Evaluation Pipeline

A two-player *Liar / Cheat* card game engine wired to scripted baselines and an Anthropic LLM agent.  The pipeline is designed for controlled evaluation: every game is fully reproducible, all hidden information is enforced at the engine level, and every turn is logged in a structured format suitable for analysis or fine-tuning.

---

## Project layout

```
liar_game/
├── engine/
│   ├── card.py        # card primitives, deck construction, Joker constants
│   ├── state.py       # GameState, Claim, public_observation()
│   └── game.py        # apply_action() — immutable state transitions
├── agents/
│   ├── base.py        # BaseAgent ABC — choose_action / report_belief / observe_event
│   ├── scripted.py    # HonestAgent, BluffAgent, HeuristicAgent
│   └── llm_agent.py   # LLMAgent backed by the Anthropic Messages API
├── logging_/
│   └── logger.py      # EpisodeLogger — structured JSON turn logs
├── eval/
│   ├── metrics.py     # episode_stats(), aggregate_stats(), print_report()
│   └── compare.py     # round-robin matchup runner
├── runner.py          # run_episode() — wires engine + agents + logger
└── main.py            # CLI entry point
```

---

## Game rules

Full 108-card deck: two standard 52-card decks plus 4 Jokers.

Players alternate turns.  Each turn the active player places 1–4 cards face-down on the pile and **claims** they are all the current required rank.  The required rank cycles A → 2 → 3 → … → K → A with every play.

The opponent may then:
- **Accept** — say nothing and become the next player (they then play the next rank).
- **Challenge** — call the bluff.  Cards are revealed:
  - Claim was **false** → the playing player picks up the entire pile.
  - Claim was **true** → the challenging player picks up the entire pile.

After a challenge, the player who picked up the pile plays next.

**Jokers are wildcards**: a Joker always counts as the claimed rank during bluff validation.  A play consisting of real matching cards and/or Jokers is always honest.

**Goal**: be the first to empty your hand.

---

## Design constraints

These invariants are preserved throughout:

| Constraint | Where enforced |
|---|---|
| `apply_action()` returns a new state and never mutates | `engine/game.py` |
| `GameState.public_observation(player_id)` is the only legal agent interface | `engine/state.py`, `runner.py` |
| Hidden information (opponent hand, `honest` flag) is filtered before agents see it | `runner.py → _public_event_for()` |
| `BaseAgent.choose_action(obs) → action` is the sole agent entry point | `agents/base.py` |
| Every turn is logged for full replay and analysis | `logging_/logger.py` |

---

## Agents

### Scripted baselines

| Agent | Behaviour | Challenges | Randomness |
|---|---|---|---|
| `HonestAgent` | Play real rank match → Joker → forced bluff | Never | None |
| `BluffAgent` | Play off-rank → Joker/real fallback | 15 % probability | Challenge only |
| `HeuristicAgent` | Real match → Joker → minimal bluff | Deterministic threshold | None |

**HeuristicAgent** is the primary scripted baseline.  It computes a structured belief after every opponent claim:

```python
{
    "opponent_bluff_prob": 0.75,   # estimated probability the last claim was a bluff
    "confidence":          0.60,   # how reliable this estimate is
    "reason_tag":          "high_claim_ratio"  # dominant signal
}
```

Signals used: `high_claim_ratio`, `high_claim_count`, `low_opp_hand`, `large_pile`.  Challenge fires when `opponent_bluff_prob ≥ 0.55`.

### LLM agent

`LLMAgent` calls the Anthropic Messages API on every turn.  The static game-rules system prompt is cached with `cache_control: ephemeral` to minimise token cost.

```python
from agents.llm_agent import LLMAgent

agent = LLMAgent(
    player_id=0,
    name="claude",
    model="claude-haiku-4-5-20251001",   # default, will adjust to Gemini later
    memory_mode="full_history",           # "current_only" | "full_history"
)
```



`memory_mode="full_history"` includes the full public turn history in every prompt.  `"current_only"` sends only the current observation.

The API key is read from `ANTHROPIC_API_KEY`.

---

## Quickstart

### Install dependencies

```bash
pip install anthropic
```

### Run a single matchup

```bash
python main.py --agent0 heuristic --agent1 bluff --episodes 10
```

```
Episode   0 | seed=0 | winner=P0 (heuristic) | turns=130
Episode   1 | seed=1 | winner=P0 (heuristic) | turns=126
...
── Summary ──────────────────────────────
  P0 (heuristic) wins : 10 / 10
  P1 (bluff) wins : 0 / 10
  Avg turns           : 147.0
```

### Watch turns unfold

```bash
python main.py --agent0 heuristic --agent1 bluff --episodes 1 --verbose
```

```
[T000] P0 plays 1x A (honest)  actual=['A♣']  hands=[53, 54]
[T001] P1 plays 1x 2 (BLUFF)   actual=['K♦']  hands=[53, 53]
[T013] P1 CHALLENGES → honest   pile→P1        hands=[47, 61]
```

### Save episode logs

```bash
python main.py --agent0 heuristic --agent1 bluff --episodes 20 --save_dir logs/run1
```

Each episode is saved as `logs/run1/ep_NNNN.json`.

### Run an LLM agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --agent0 llm --agent1 heuristic --episodes 5
```

Ablate memory mode or swap models:

```bash
python main.py --agent0 llm --agent1 heuristic --memory current_only
python main.py --agent0 llm --agent1 heuristic --model claude-haiku-4-5-20251001
```

---

## Evaluation

### Round-robin comparison

```bash
python eval/compare.py --episodes 20 --save_dir logs/compare
```

```
Running 20 episodes per matchup

                       Matchup   Results
─────────────────────────────────────────────────────────────────
     heuristic vs bluff        │ P0 wins:  20  P1 wins:   0  draws:   0
     heuristic vs honest       │ P0 wins:  12  P1 wins:   8  draws:   0
        honest vs bluff        │ P0 wins:  19  P1 wins:   1  draws:   0

── Aggregate stats across all episodes ──────────────────────────
Agent           Win rate   Bluff rate  Challenge rate  Challenge accuracy  Bluff caught rate  Avg turns/game
───────────────────────────────────────────────────────────────────────────────────────────────────────────
bluff              0.000        1.000           0.143               0.152              0.000         182.600
heuristic          0.800        0.164           0.081               0.441              0.071         166.600
honest             0.700        0.195           0.000               0.000              0.247         156.200
```

### Custom matchup including LLM

```bash
python eval/compare.py --matchups llm:heuristic llm:bluff --episodes 10
```

### Compute metrics from saved logs

```python
from pathlib import Path
from eval.metrics import load_episode, aggregate_stats, print_report

logs = [load_episode(p) for p in Path("logs/compare").glob("*.json")]
print_report(aggregate_stats(logs))
```

Available per-player metrics: `win_rate`, `bluff_rate`, `honest_rate`, `challenge_rate`, `challenge_acc`, `bluff_caught_rate`, `total_turns`.

---

## Episode log format

```jsonc
{
  "episode_id": "5595d7c9",
  "seed": 0,
  "agents": [
    {"player_id": 0, "name": "heuristic", "type": "HeuristicAgent"},
    {"player_id": 1, "name": "bluff",     "type": "BluffAgent"}
  ],
  "turns": [
    {
      "turn": 0,
      "player": 0,
      "action": {"type": "play", "cards": ["A♣"], "claimed_rank": "A"},
      "observation_before": { ... },   // what P0 saw before acting
      "event": {                       // full engine event (includes actual_cards)
        "turn": 0, "player": 0,
        "claimed_rank": "A", "n_cards": 1,
        "actual_cards": ["A♣"], "honest": true,
        ...
      },
      "hand_sizes_after": [53, 54],
      "beliefs": {"0": {"opponent_bluff_prob": 0.55, ...}}  // optional
    }
  ],
  "outcome": {"winner": 0, "total_turns": 130, "reason": "empty_hand"}
}
```

`observation_before` contains only what the acting player could legally see (no opponent hand, no `honest` flag on opponent plays).  `event` contains the full ground truth and is used for metrics and analysis — never exposed to agents during play.

---

## Adding a new agent

1. Subclass `BaseAgent` in `agents/`.
2. Implement `choose_action(self, obs) -> dict`.
3. Optionally implement `report_belief(self, obs=None) -> dict | None` and `observe_event(self, event) -> None`.
4. Register the agent name in `AGENT_MAP` in `main.py` and `eval/compare.py`.

```python
from agents.base import BaseAgent

class MyAgent(BaseAgent):
    def choose_action(self, obs):
        # obs keys: my_hand, my_hand_size, opponent_hand_size,
        #           pile_size, current_rank, current_player, turn, last_claim
        rank = obs["current_rank"]
        matching = [c for c in obs["my_hand"] if c.rank == rank]
        cards = matching[:1] if matching else obs["my_hand"][:1]
        return {"type": "play", "cards": cards, "claimed_rank": rank}
```
