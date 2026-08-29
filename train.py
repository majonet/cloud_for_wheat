import os
import csv
import numpy as np
import torch

import config as C
from agent import DQNAgent
from env_wrapper import WheatFarmEnv


def next_state_for_unit(env, obs_next, unit_id, unit_idx, global_vec_next):
    """Re-encode a specific unit's state after the env step (used as s' for its transition)."""
    from state_encoder import encode_unit
    farm = obs_next["farms"][env.player]
    units = env._units(obs_next)
    if unit_idx >= len(units):
        # unit disappeared this turn (e.g. a hand expired at day rollover) -> reuse zeros
        return np.zeros(C.UNIT_STATE_DIM, dtype=np.float32), np.zeros(C.N_UNIT_ACTIONS, dtype=bool)
    _, xy = units[unit_idx]
    private = obs_next.get("private", {}) or {}
    seeds = private.get("seeds", {}).get("WHEAT", 0)
    inventories = private.get("inventories", [])
    inv = inventories[unit_idx] if unit_idx < len(inventories) else {}
    wheat_in_hand = inv.get("WHEAT", 0) if isinstance(inv, dict) else 0
    state = encode_unit(obs_next, env.player, xy, wheat_in_hand, global_vec_next)
    from action_space import unit_action_mask
    board_size = len(farm["tiles"])
    mask = unit_action_mask(obs_next, env.player, xy, seeds > 0, board_size)
    return state, mask


def run_episode(env, unit_agent, market_agent, greedy=False):
    obs = env.reset()
    done = False
    ep_reward = 0.0
    ep_money_delta = 0.0
    metrics = dict(seed_bought=0, planted=0, harvested=0, sold_qty=0, hires=0, land_buys=0, invalid=0)

    while not done:
        unit_decisions, market_decision, price = env.collect_decisions(obs, unit_agent, market_agent, greedy)
        command = env.build_command(obs, unit_decisions, market_decision)

        # lightweight bookkeeping for eval metrics
        if command["farmer"][0] == "PLANT":
            metrics["planted"] += 1
        if command["farmer"][0] == "HARVEST":
            metrics["harvested"] += 1
        for m in command["market"]:
            if m[0] == "BUY_SEED":
                metrics["seed_bought"] += m[2]
            elif m[0] == "SELL":
                metrics["sold_qty"] += m[2]
            elif m[0] == "HIRE":
                metrics["hires"] += 1
            elif m[0] == "BUY_LAND":
                metrics["land_buys"] += 1

        obs_next, done = env.step(command)
        r, delta_money = env.reward(obs_next)
        ep_reward += r
        ep_money_delta += delta_money

        from state_encoder import encode_global
        global_vec_next, _ = encode_global(obs_next, env.player, price)

        units_before = env._units(obs)
        for unit_idx, (unit_id, _) in enumerate(units_before):
            dec = unit_decisions[unit_id]
            ns, nm = next_state_for_unit(env, obs_next, unit_id, unit_idx, global_vec_next)
            unit_agent.remember(dec["state"], dec["action"], r, ns, float(done), dec["mask"], nm)

        from action_space import market_action_mask
        next_market_mask = market_action_mask(obs_next, env.player)
        market_agent.remember(
            market_decision["state"], market_decision["action"], r,
            global_vec_next, float(done), market_decision["mask"], next_market_mask,
        )

        if not greedy:
            unit_agent.learn()
            market_agent.learn()

        obs = obs_next

    final_money = obs["farms"][env.player]["money"]
    return ep_reward, final_money, metrics


def main():
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unit_agent = DQNAgent(C.UNIT_STATE_DIM, C.N_UNIT_ACTIONS, device=device)
    market_agent = DQNAgent(C.MARKET_STATE_DIM, C.N_MARKET_ACTIONS, device=device)
    env = WheatFarmEnv()

    log_path = os.path.join(C.CHECKPOINT_DIR, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "final_money", "planted", "harvested",
                          "sold_qty", "seed_bought", "hires", "land_buys"])

    recent_money = []
    for ep in range(1, C.NUM_EPISODES + 1):
        ep_reward, final_money, m = run_episode(env, unit_agent, market_agent, greedy=False)
        recent_money.append(final_money)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, ep_reward, final_money, m["planted"], m["harvested"],
                              m["sold_qty"], m["seed_bought"], m["hires"], m["land_buys"]])

        if ep % C.LOG_EVERY == 0:
            avg_money = np.mean(recent_money[-C.LOG_EVERY:])
            print(f"[ep {ep}] reward={ep_reward:.2f} final_money={final_money:.1f} "
                  f"avg_money(last {C.LOG_EVERY})={avg_money:.1f} eps={unit_agent.epsilon():.3f}")

        if ep % C.CHECKPOINT_EVERY == 0:
            unit_agent.save(os.path.join(C.CHECKPOINT_DIR, f"unit_agent_ep{ep}.pt"))
            market_agent.save(os.path.join(C.CHECKPOINT_DIR, f"market_agent_ep{ep}.pt"))

    unit_agent.save(os.path.join(C.CHECKPOINT_DIR, "unit_agent_final.pt"))
    market_agent.save(os.path.join(C.CHECKPOINT_DIR, "market_agent_final.pt"))
    print("Training complete. Log at", log_path)


if __name__ == "__main__":
    main()
