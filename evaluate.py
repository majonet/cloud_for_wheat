import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config as C
from agent import DQNAgent
from env_wrapper import WheatFarmEnv
from action_space import unit_action_to_command, market_action_to_command


# ---------------------------------------------------------------------------
# Heuristic baseline: Buy -> Plant -> Water daily -> Harvest at yield_units==4
# -> Sell immediately. No HIRE, no BUY_LAND (matches the earlier melon_maxxer
# pattern, adapted to WHEAT's real numbers from the README).
# ---------------------------------------------------------------------------
def heuristic_agent(obs, player=0):
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    seeds = private.get("seeds", {}).get("WHEAT", 0)
    shed = private.get("shed", {}).get("WHEAT", 0)
    price = obs.get("market", {}).get("prices", {}).get("WHEAT", 0)

    market = []
    if shed > 0 and price > 0:
        market.append(["SELL", "WHEAT", shed])
    if seeds == 0 and farm["money"] >= C.WHEAT_SEED_COST:
        market.append(["BUY_SEED", "WHEAT", 1])

    farmer = ["PASS"]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
        if tile.get("yield_units", 0) >= C.WHEAT_MAX_YIELD_NO_FERTILIZER:
            farmer = ["HARVEST"]
        elif not tile.get("watered_today"):
            farmer = ["WATER"]
    elif tile is None and seeds > 0:
        farmer = ["PLANT", "WHEAT"]

    return {"farmer": farmer, "hands": [], "market": market}


def run_heuristic_episode(episode_steps=C.EPISODE_STEPS, seed=None):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": episode_steps, "seed": seed})
    state = env.reset()
    while not env.done:
        obs = state[0]["observation"]
        cmd0 = heuristic_agent(obs, player=0)
        cmd1 = {"farmer": ["PASS"], "hands": [], "market": []}
        state = env.step([cmd0, cmd1])
    final_money = state[0]["observation"]["farms"][0]["money"]
    return final_money


# ---------------------------------------------------------------------------
# Trained-agent evaluation
# ---------------------------------------------------------------------------
def run_trained_episode(env, unit_agent, market_agent, print_trajectory=False):
    obs = env.reset()
    done = False
    trajectory = []
    while not done:
        unit_decisions, market_decision, _ = env.collect_decisions(obs, unit_agent, market_agent, greedy=True)
        command = env.build_command(obs, unit_decisions, market_decision)
        if print_trajectory:
            day, hour = obs.get("day", 0), obs.get("hour", 0)
            for m in command["market"]:
                trajectory.append(f"Day {day} h{hour}: {m}")
            if command["farmer"][0] != "PASS":
                trajectory.append(f"Day {day} h{hour}: FARMER {command['farmer']}")
        obs, done = env.step(command)
    final_money = obs["farms"][env.player]["money"]
    return final_money, trajectory


def summarize(values, label):
    values = np.array(values, dtype=float)
    print(f"\n--- {label} ---")
    print(f"mean={values.mean():.1f}  median={np.median(values):.1f}  "
          f"best={values.max():.1f}  worst={values.min():.1f}  std={values.std():.1f}")


def plot_training_curves(log_path, out_dir):
    if not os.path.exists(log_path):
        print("No training log found at", log_path)
        return
    episodes, rewards, money, sold, hires, land = [], [], [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            money.append(float(row["final_money"]))
            sold.append(float(row["sold_qty"]))
            hires.append(float(row["hires"]))
            land.append(float(row["land_buys"]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plots = [
        ("Final Money", money), ("Reward", rewards), ("Wheat Sold", sold),
        ("Hires / episode", hires), ("Land buys / episode", land),
    ]
    for ax, (title, series) in zip(axes.flat, plots):
        ax.plot(episodes, series)
        ax.set_title(title)
        ax.set_xlabel("Episode")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=120)
    print("Saved", out_path)


def main(n_eval_episodes=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unit_agent = DQNAgent(C.UNIT_STATE_DIM, C.N_UNIT_ACTIONS, device=device)
    market_agent = DQNAgent(C.MARKET_STATE_DIM, C.N_MARKET_ACTIONS, device=device)
    unit_agent.load(os.path.join(C.CHECKPOINT_DIR, "unit_agent_final.pt"))
    market_agent.load(os.path.join(C.CHECKPOINT_DIR, "market_agent_final.pt"))

    env = WheatFarmEnv()

    rl_money = []
    for i in range(n_eval_episodes):
        print_traj = (i == 0)
        final_money, trajectory = run_trained_episode(env, unit_agent, market_agent, print_trajectory=print_traj)
        rl_money.append(final_money)
        if print_traj:
            print("\n=== Sample trajectory (episode 0) ===")
            for line in trajectory[:60]:
                print(line)

    heuristic_money = [run_heuristic_episode() for _ in range(n_eval_episodes)]

    summarize(rl_money, "Trained RL agent (final money)")
    summarize(heuristic_money, "Heuristic baseline (final money)")

    plot_training_curves(os.path.join(C.CHECKPOINT_DIR, "train_log.csv"), C.CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
