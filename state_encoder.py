"""
Turns the raw kaggriculture observation dict into fixed-size numpy feature vectors.

Two encoders:
  - encode_global(obs, player): economy/time/land features shared by every unit this turn.
  - encode_unit(obs, player, unit_xy, unit_inventory, global_vec): local 5x5 window
    around a specific unit (farmer or a hand) + that unit's own inventory + distance features.
"""
import numpy as np

from config import (
    LOCAL_WINDOW, LOCAL_CELL_FEATURES, GLOBAL_FEATURES,
    WHEAT_MAX_YIELD_NO_FERTILIZER, BOARD_SIZE_DEFAULT,
)

MONEY_NORM = 10_000.0
PRICE_NORM = 100.0
MARKET_INV_NORM = 500.0
SEED_NORM = 50.0
SHED_NORM = 200.0
TILE_COUNT_NORM = 100.0
HIRES_NORM = 5.0
HANDS_NORM = 5.0


def _tile_counts(farm):
    """Count free / wheat / harvestable-wheat / weed tiles across the whole visible board."""
    free = wheat = harvestable = weeds = 0
    for row in farm["tiles"]:
        for tile in row:
            if tile is None:
                free += 1
            elif tile == "LOCKED":
                continue
            elif isinstance(tile, dict):
                if tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
                    wheat += 1
                    if tile.get("yield_units", 0) > 0:
                        harvestable += 1
                elif tile.get("kind") == "WEED":
                    weeds += 1
    return free, wheat, harvestable, weeds


def encode_global(obs, player, prev_price=None):
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}

    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    step = day * 24 + hour
    remaining = max(0, 720 - step)

    money = farm.get("money", 0.0)
    price = market.get("prices", {}).get("WHEAT", 0)
    market_inv = market.get("inventory", {}).get("WHEAT", 0)
    price_change = 0.0 if prev_price is None else (price - prev_price)

    seeds = private.get("seeds", {}).get("WHEAT", 0)
    shed = private.get("shed", {}).get("WHEAT", 0)

    n_quadrants = len(farm.get("unlocked_quadrants", []))
    free, wheat, harvestable, weeds = _tile_counts(farm)
    hires_today = farm.get("hires_today", 0)
    n_hands = len(farm.get("hands", []))

    vec = np.array([
        day / 30.0,
        hour / 24.0,
        remaining / 720.0,
        min(money / MONEY_NORM, 5.0),
        min(price / PRICE_NORM, 5.0),
        min(market_inv / MARKET_INV_NORM, 5.0),
        np.clip(price_change / 50.0, -5.0, 5.0),
        min(seeds / SEED_NORM, 5.0),
        min(shed / SHED_NORM, 5.0),
        n_quadrants / 4.0,
        min(free / TILE_COUNT_NORM, 5.0),
        min(wheat / TILE_COUNT_NORM, 5.0),
        min(harvestable / TILE_COUNT_NORM, 5.0),
        min(weeds / TILE_COUNT_NORM, 5.0),
        hires_today / HIRES_NORM,
        n_hands / HANDS_NORM,
    ], dtype=np.float32)
    assert vec.shape[0] == GLOBAL_FEATURES
    return vec, price


def _encode_cell(tile):
    """8 features per cell: [locked, empty, is_wheat, age_norm, yield_norm, watered, unwatered_norm, is_weed]."""
    f = np.zeros(LOCAL_CELL_FEATURES, dtype=np.float32)
    if tile == "LOCKED":
        f[0] = 1.0
    elif tile is None:
        f[1] = 1.0
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
        f[2] = 1.0
        f[3] = min(1.0, 0.0)  # age is not directly given; yield_units is the informative signal instead
        f[4] = min(tile.get("yield_units", 0) / WHEAT_MAX_YIELD_NO_FERTILIZER, 1.5)
        f[5] = 1.0 if tile.get("watered_today") else 0.0
        f[6] = min(tile.get("consecutive_unwatered", 0) / 2.0, 1.5)
    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
        f[7] = 1.0
    # Any other structure (COOP/PASTURE) or other crop: leave as all-zero "other" cell.
    return f


def _nearest_offset(fx, fy, coords, board_size):
    if not coords:
        return (0.0, 0.0)
    tx, ty = min(coords, key=lambda c: abs(c[0] - fx) + abs(c[1] - fy))
    return ((tx - fx) / board_size, (ty - fy) / board_size)


def encode_unit(obs, player, unit_xy, unit_wheat_count, global_vec):
    farm = obs["farms"][player]
    tiles = farm["tiles"]
    board_size = len(tiles)
    fx, fy = unit_xy
    half = LOCAL_WINDOW // 2

    local = np.zeros(LOCAL_WINDOW * LOCAL_WINDOW * LOCAL_CELL_FEATURES, dtype=np.float32)
    idx = 0
    unwatered_coords, harvestable_coords = [], []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x, y = fx + dx, fy + dy
            if 0 <= x < board_size and 0 <= y < board_size:
                tile = tiles[y][x]
                local[idx:idx + LOCAL_CELL_FEATURES] = _encode_cell(tile)
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
                    if tile.get("yield_units", 0) > 0:
                        harvestable_coords.append((x, y))
                    elif not tile.get("watered_today"):
                        unwatered_coords.append((x, y))
            else:
                local[idx] = 1.0  # treat off-board as "locked/blocked"
            idx += LOCAL_CELL_FEATURES

    dx_u, dy_u = _nearest_offset(fx, fy, unwatered_coords, board_size)
    dx_h, dy_h = _nearest_offset(fx, fy, harvestable_coords, board_size)
    inv_feat = np.array([min(unit_wheat_count / 100.0, 2.0)], dtype=np.float32)
    dist_feat = np.array([dx_u, dy_u, dx_h, dy_h], dtype=np.float32)

    return np.concatenate([global_vec, local, inv_feat, dist_feat]).astype(np.float32)
