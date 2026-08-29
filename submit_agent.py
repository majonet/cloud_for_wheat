"""
Wraps the trained UnitPolicy + MarketPolicy into a single function with the
{"farmer": [...], "hands": [...], "market": [...]} signature kaggle_environments expects.
Load once at import time; kaggle_environments calls the returned `wheat_rl_agent`
function once per turn.
"""
import os
import torch

import config as C
from agent import DQNAgent
from state_encoder import encode_global, encode_unit
from action_space import (
    unit_action_mask, market_action_mask,
    unit_action_to_command, market_action_to_command,
)

_DEVICE = "cpu"
_UNIT_AGENT = DQNAgent(C.UNIT_STATE_DIM, C.N_UNIT_ACTIONS, device=_DEVICE)
_MARKET_AGENT = DQNAgent(C.MARKET_STATE_DIM, C.N_MARKET_ACTIONS, device=_DEVICE)
_UNIT_AGENT.load(os.path.join(C.CHECKPOINT_DIR, "unit_agent_final.pt"))
_MARKET_AGENT.load(os.path.join(C.CHECKPOINT_DIR, "market_agent_final.pt"))
_prev_price_holder = {"price": None}


def wheat_rl_agent(obs):
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    board_size = len(farm["tiles"])
    private = obs.get("private", {}) or {}
    seeds = private.get("seeds", {}).get("WHEAT", 0)
    shed_wheat = private.get("shed", {}).get("WHEAT", 0)
    inventories = private.get("inventories", [])

    global_vec, price = encode_global(obs, player, _prev_price_holder["price"])
    _prev_price_holder["price"] = price

    units = [("farmer", tuple(farm["farmer"]))]
    for hand_xy in farm.get("hands", []):
        units.append(("hand", tuple(hand_xy)))

    unit_cmds = []
    for i, (_, xy) in enumerate(units):
        inv = inventories[i] if i < len(inventories) else {}
        wheat_in_hand = inv.get("WHEAT", 0) if isinstance(inv, dict) else 0
        state = encode_unit(obs, player, xy, wheat_in_hand, global_vec)
        mask = unit_action_mask(obs, player, xy, seeds > 0, board_size)
        action_idx = _UNIT_AGENT.act(state, mask, greedy=True)
        unit_cmds.append(unit_action_to_command(action_idx))

    market_mask = market_action_mask(obs, player)
    market_idx = _MARKET_AGENT.act(global_vec, market_mask, greedy=True)
    market_cmd = market_action_to_command(market_idx, shed_wheat)
    market_list = [market_cmd] if market_cmd is not None else []

    return {"farmer": unit_cmds[0], "hands": unit_cmds[1:], "market": market_list}
