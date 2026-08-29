"""
Translates between discrete action indices (what the networks output) and the
actual game command strings, and builds valid-action masks so illegal actions
never get selected (per the README's action-legality rules).
"""
import numpy as np

from config import (
    UNIT_ACTIONS, MARKET_ACTIONS, N_UNIT_ACTIONS, N_MARKET_ACTIONS,
    WHEAT_SEED_COST, LAND_COSTS,
)

_STEP_DELTA = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}


def unit_action_mask(obs, player, unit_xy, has_wheat_seed, board_size):
    """Boolean mask of length N_UNIT_ACTIONS: True = legal to attempt this turn."""
    farm = obs["farms"][player]
    fx, fy = unit_xy
    tile = farm["tiles"][fy][fx]

    mask = np.ones(N_UNIT_ACTIONS, dtype=bool)

    is_wheat = isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT"
    is_empty = tile is None
    is_locked = tile == "LOCKED"
    is_weed = isinstance(tile, dict) and tile.get("kind") == "WEED"

    for i, name in enumerate(UNIT_ACTIONS):
        if name in _STEP_DELTA:
            dx, dy = _STEP_DELTA[name]
            nx, ny = fx + dx, fy + dy
            # Off-edge moves are legal no-ops per the rules; we allow them (agent just wastes a turn).
            mask[i] = True
        elif name == "PLANT":
            mask[i] = is_empty and has_wheat_seed and not is_locked
        elif name == "WATER":
            mask[i] = is_wheat and not tile.get("watered_today", False)
        elif name == "HARVEST":
            mask[i] = is_wheat and tile.get("yield_units", 0) > 0
        elif name == "DIG":
            mask[i] = is_weed or is_wheat  # remove a plant to free the tile, or clear a weed
        elif name in ("PICKUP", "DROP"):
            mask[i] = True  # legality depends on shed-adjacency; env no-ops it harmlessly if not
        elif name == "PASS":
            mask[i] = True
    return mask


def market_action_mask(obs, player):
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    money = farm.get("money", 0.0)
    shed_wheat = private.get("shed", {}).get("WHEAT", 0)
    n_quadrants = len(farm.get("unlocked_quadrants", []))
    hires_today = farm.get("hires_today", 0)

    # Fibonacci-based hire cost: 1,1,2,3,5,8,13,... indexed by hires_today.
    def fib(n):
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    next_hire_cost = fib(hires_today)  # multiplied by farmHandCostMult inside the env; treat as relative
    next_land_cost = LAND_COSTS[n_quadrants - 1] if 0 < n_quadrants <= len(LAND_COSTS) else None

    mask = np.ones(N_MARKET_ACTIONS, dtype=bool)
    for i, (name, param) in enumerate(MARKET_ACTIONS):
        if name == "NOOP":
            mask[i] = True
        elif name == "BUY_SEED":
            mask[i] = money >= WHEAT_SEED_COST * param
        elif name == "SELL":
            mask[i] = shed_wheat > 0
        elif name == "HIRE":
            mask[i] = money >= next_hire_cost
        elif name == "BUY_LAND":
            mask[i] = next_land_cost is not None and money >= next_land_cost
    return mask


def unit_action_to_command(idx):
    name = UNIT_ACTIONS[idx]
    if name == "PLANT":
        return ["PLANT", "WHEAT"]
    return [name]


def market_action_to_command(idx, shed_wheat):
    name, param = MARKET_ACTIONS[idx]
    if name == "NOOP":
        return None
    if name == "BUY_SEED":
        return ["BUY_SEED", "WHEAT", int(param)]
    if name == "SELL":
        qty = max(1, int(round(shed_wheat * param))) if shed_wheat > 0 else 0
        return ["SELL", "WHEAT", qty] if qty > 0 else None
    if name == "HIRE":
        return ["HIRE"]
    if name == "BUY_LAND":
        return ["BUY_LAND"]
    return None
