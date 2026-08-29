"""
Wraps kaggle_environments' kaggriculture env for the WHEAT-only agent.

Design (method B — shared per-unit policy):
  - Every turn, UnitPolicy is queried once for the farmer and once per hand (same network,
    different local state), MarketPolicy is queried once for the whole turn.
  - All of those decisions share ONE scalar reward: r_t = Δmoney_t*scale - penalties.
    Because Σ Δmoney_t telescopes to (final_money - start_money), this keeps every unit's
    learning signal exactly aligned with the true objective (final bank money) without a
    sparse terminal-only reward.
  - The opponent (player 1) is a fixed PASS agent, since our objective is our own money,
    not zero-sum competition against player 1.
"""
import numpy as np
from kaggle_environments import make

import config as C
from state_encoder import encode_global, encode_unit
from action_space import (
    unit_action_mask, market_action_mask,
    unit_action_to_command, market_action_to_command,
)


def _opponent_pass_agent(obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}


class WheatFarmEnv:
    def __init__(self, episode_steps=C.EPISODE_STEPS, seed=None):
        self.episode_steps = episode_steps
        self.seed = seed
        self.env = None
        self.player = 0
        self.prev_price = None
        self.prev_money = None
        self.prev_weeds = 0

    def reset(self):
        self.env = make("kaggriculture", configuration={"episodeSteps": self.episode_steps, "seed": self.seed})
        state = self.env.reset()
        obs = state[self.player]["observation"]
        self.prev_price = obs.get("market", {}).get("prices", {}).get("WHEAT", C.WHEAT_BASE_PRICE)
        self.prev_money = obs["farms"][self.player]["money"]
        self.prev_weeds = self._count_weeds(obs)
        return obs

    def _count_weeds(self, obs):
        farm = obs["farms"][self.player]
        n = 0
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    n += 1
        return n

    def _units(self, obs):
        farm = obs["farms"][self.player]
        units = [("farmer", tuple(farm["farmer"]))]
        for i, hand_xy in enumerate(farm.get("hands", [])):
            units.append((f"hand_{i}", tuple(hand_xy)))
        return units

    def collect_decisions(self, obs, unit_agent, market_agent, greedy=False):
        """Query the policies for this turn. Returns everything needed to both act
        and later store transitions (states/masks/actions), keyed by unit id."""
        farm = obs["farms"][self.player]
        board_size = len(farm["tiles"])
        private = obs.get("private", {}) or {}
        seeds = private.get("seeds", {}).get("WHEAT", 0)
        inventories = private.get("inventories", [])

        global_vec, price = encode_global(obs, self.player, self.prev_price)

        unit_decisions = {}
        for unit_i, (unit_id, xy) in enumerate(self._units(obs)):
            inv = inventories[unit_i] if unit_i < len(inventories) else {}
            wheat_in_hand = inv.get("WHEAT", 0) if isinstance(inv, dict) else 0
            state = encode_unit(obs, self.player, xy, wheat_in_hand, global_vec)
            mask = unit_action_mask(obs, self.player, xy, seeds > 0, board_size)
            action_idx = unit_agent.act(state, mask, greedy=greedy)
            unit_decisions[unit_id] = {"state": state, "mask": mask, "action": action_idx}

        market_state = global_vec.copy()
        market_mask = market_action_mask(obs, self.player)
        market_action_idx = market_agent.act(market_state, market_mask, greedy=greedy)

        return unit_decisions, {"state": market_state, "mask": market_mask, "action": market_action_idx}, price

    def build_command(self, obs, unit_decisions, market_decision):
        farm = obs["farms"][self.player]
        private = obs.get("private", {}) or {}
        shed_wheat = private.get("shed", {}).get("WHEAT", 0)

        units = self._units(obs)
        farmer_cmd = unit_action_to_command(unit_decisions[units[0][0]]["action"])
        hand_cmds = [unit_action_to_command(unit_decisions[uid]["action"]) for uid, _ in units[1:]]

        market_cmd = market_action_to_command(market_decision["action"], shed_wheat)
        market_list = [market_cmd] if market_cmd is not None else []

        return {"farmer": farmer_cmd, "hands": hand_cmds, "market": market_list}

    def step(self, command):
        opponent_cmd = _opponent_pass_agent(None)
        state = self.env.step([command, opponent_cmd])
        obs = state[self.player]["observation"]
        done = self.env.done
        return obs, done

    def reward(self, obs):
        farm = obs["farms"][self.player]
        money = farm["money"]
        weeds = self._count_weeds(obs)

        delta_money = money - self.prev_money
        new_weeds = max(0, weeds - self.prev_weeds)

        r = delta_money * C.REWARD_MONEY_SCALE + new_weeds * C.WEED_PENALTY

        self.prev_money = money
        self.prev_weeds = weeds
        return float(r), float(delta_money)
