import argparse
import itertools
import os
import random
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import flappy_bird_gymnasium  # noqa: F401  (registers the FlappyBird-v0 env)
from dqn import DQN
from experience_replay import ReplayMemory

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


def discretize_state(state_tensor, bins=10, low=-2.0, high=2.0, n_buckets=None):
    """Turn a continuous state vector into a hashable key for visit counting.

    Flappy Bird's observation space is continuous, so exact tabular visit
    counts (as used by UCB-Q-Learning / Delayed Q-Learning style algorithms)
    aren't directly applicable. This bins each dimension so we can still keep
    a visit-count table.

    With the 12-value "simple" observation this is fine as-is. With the
    180-reading LiDAR observation, a raw per-dimension bin tuple would have
    an astronomical number of distinct keys (bins^180) — visit counts would
    almost never repeat, making the UCB bonus meaningless. Pass n_buckets to
    hash the binned tuple down into a fixed-size table instead; this trades
    exactness for tractability (different states can collide into the same
    bucket), which is the practical fix used here for the LiDAR setting.
    """
    state_np = state_tensor.detach().cpu().numpy()
    clipped = np.clip(state_np, low, high)
    bin_idx = np.floor((clipped - low) / (high - low) * bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    key = tuple(bin_idx.tolist())
    if n_buckets is not None:
        return hash(key) % n_buckets
    return key


class Agent:
    def __init__(self, param_set, parameters_file="parameters.yaml"):
        self.param_set = param_set
        with open(parameters_file, "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        # Core DQN hyperparameters (shared across exploration strategies)
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        # Exploration-strategy selection + strategy-specific knobs
        self.exploration = params.get("exploration", "epsilon_greedy")
        self.epsilon_fixed = params.get("epsilon_fixed", 0.1)
        self.temp_init = params.get("temp_init", 1.0)
        self.temp_min = params.get("temp_min", 0.05)
        self.temp_decay = params.get("temp_decay", 0.995)
        self.ucb_c = params.get("ucb_c", 2.0)
        self.count_beta = params.get("count_beta", 0.1)
        self.state_bins = params.get("state_bins", 10)
        self.count_hash_buckets = params.get("count_hash_buckets", None)  # bucket count for high-dim (LiDAR) states
        self.grad_alpha = params.get("grad_alpha", 0.1)  # step-size for preference updates (gradient method)

        # Environment / network sizing
        self.use_lidar = params.get("use_lidar", False)
        self.hidden_dim = params.get("hidden_dim", 256)
        self.log_every = params.get("log_every", 1)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")
        self.RESUME_MODEL_FILE = "flappybirdv0_backup.pt"

    def select_action(self, state, policy_dqn, num_actions):
        """Pick an action according to this agent's exploration strategy."""

        if self.exploration == "epsilon_greedy":
            if random.random() < self.epsilon:
                return torch.tensor(random.randrange(num_actions), dtype=torch.long, device=device)
            with torch.no_grad():
                return policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

        elif self.exploration == "fixed_epsilon":
            if random.random() < self.epsilon_fixed:
                return torch.tensor(random.randrange(num_actions), dtype=torch.long, device=device)
            with torch.no_grad():
                return policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

        elif self.exploration == "boltzmann":
            with torch.no_grad():
                q = policy_dqn(state.unsqueeze(dim=0)).squeeze()
                probs = torch.softmax(q / max(self.temp, 1e-3), dim=0)
                action = torch.multinomial(probs, 1).item()
            return torch.tensor(action, dtype=torch.long, device=device)

        elif self.exploration == "ucb":
            with torch.no_grad():
                q = policy_dqn(state.unsqueeze(dim=0)).squeeze().cpu().numpy()
            key = discretize_state(state, bins=self.state_bins, n_buckets=self.count_hash_buckets)
            counts = self.visit_counts[key]
            total = counts.sum() + 1.0
            bonus = self.ucb_c * np.sqrt(np.log(total + 1.0) / (counts + 1.0))
            action = int(np.argmax(q + bonus))
            counts[action] += 1
            return torch.tensor(action, dtype=torch.long, device=device)

        elif self.exploration == "gradient":
            # Gradient bandit / REINFORCE-style approach: maintain action
            # preferences H(s,a), convert to a policy via softmax, and sample.
            # The preference update (using the reward baseline) happens in
            # run() once the reward for this action is observed.
            key = discretize_state(state, bins=self.state_bins, n_buckets=self.count_hash_buckets)
            H = self.preferences[key]
            exp_H = np.exp(H - np.max(H))  # numerically stable softmax
            probs = exp_H / exp_H.sum()
            action = int(np.random.choice(len(probs), p=probs))
            # cache for the update step in run()
            self._last_grad_key = key
            self._last_grad_probs = probs
            self._last_grad_action = action
            return torch.tensor(action, dtype=torch.long, device=device)

        elif self.exploration == "count_bonus":
            # Near-greedy action selection; the exploration incentive is injected
            # into the *reward signal* instead (see run()), inspired by the
            # optimism-under-uncertainty idea behind Delayed/UCB Q-Learning.
            if random.random() < 0.05:
                action = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()
            return torch.tensor(action, dtype=torch.long, device=device)

        else:
            raise ValueError(f"Unknown exploration strategy: {self.exploration}")

    def run(self, is_training=True, render=False, max_episodes=None, seed=None):
        """Train or evaluate. If max_episodes is given, stops after that many
        episodes and returns the per-episode TRUE environment reward history
        (i.e. never includes any exploration bonus), so strategies are
        compared on the same footing.
        """
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=self.use_lidar)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions, hidden_dim=self.hidden_dim).to(device)

        reward_history = []

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            if os.path.exists(self.RESUME_MODEL_FILE):
                policy_dqn.load_state_dict(torch.load(self.RESUME_MODEL_FILE, map_location=device))
                print(f"[{self.param_set}] Loaded {self.RESUME_MODEL_FILE}")

            self.epsilon = self.epsilon_init
            self.temp = self.temp_init
            self.visit_counts = defaultdict(lambda: np.zeros(num_actions))
            self.preferences = defaultdict(lambda: np.zeros(num_actions))  # H(s,a) for gradient method
            self.reward_baseline = 0.0  # running mean reward R_bar_t
            self.baseline_count = 0

            target_dqn = DQN(num_states, num_actions, hidden_dim=self.hidden_dim).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
            best_episode = None
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()

        episode_iter = range(max_episodes) if max_episodes is not None else itertools.count()

        for episode in episode_iter:
            state, _ = env.reset(seed=seed)
            state = torch.tensor(state, dtype=torch.float, device=device)
            episode_reward = 0.0
            terminated = False

            while not terminated and episode_reward < self.reward_threshold:
                if is_training:
                    action = self.select_action(state, policy_dqn, num_actions)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, _ = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                episode_reward += reward  # true env reward, used for comparison

                if is_training:
                    if self.exploration == "gradient":
                        # R_bar_t: incremental running mean of rewards seen so far
                        self.baseline_count += 1
                        self.reward_baseline += (reward - self.reward_baseline) / self.baseline_count

                        H = self.preferences[self._last_grad_key]
                        probs = self._last_grad_probs
                        a_taken = self._last_grad_action
                        delta = self.grad_alpha * (reward - self.reward_baseline)
                        # H_{t+1}(A_t) = H_t(A_t) + alpha*(R_t - Rbar_t)*(1 - pi_t(A_t))
                        # H_{t+1}(a)   = H_t(a)   - alpha*(R_t - Rbar_t)*pi_t(a)   for all a != A_t
                        H[a_taken] += delta * (1 - probs[a_taken])
                        others = np.arange(num_actions) != a_taken
                        H[others] -= delta * probs[others]

                    stored_reward = reward
                    if self.exploration == "count_bonus":
                        key = discretize_state(state, bins=self.state_bins, n_buckets=self.count_hash_buckets)
                        counts = self.visit_counts[key]
                        bonus = self.count_beta / np.sqrt(counts[action.item()] + 1.0)
                        counts[action.item()] += 1
                        stored_reward = reward + bonus  # shaped reward for training only

                    stored_reward_t = torch.tensor(stored_reward, dtype=torch.float, device=device)
                    memory.append((state, action, next_state, stored_reward_t, terminated))
                    steps += 1

                state = next_state

            reward_history.append(episode_reward)

            if is_training:
                if self.exploration == "epsilon_greedy":
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                elif self.exploration == "boltzmann":
                    self.temp = max(self.temp * self.temp_decay, self.temp_min)

                if episode_reward > best_reward:
                    log_msg = f"best reward={episode_reward} for episode={episode + 1}"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                    best_episode = episode + 1

                if (episode + 1) % self.log_every == 0:
                    extra = ""
                    if self.exploration in ("epsilon_greedy", "fixed_epsilon"):
                        eps = self.epsilon if self.exploration == "epsilon_greedy" else self.epsilon_fixed
                        extra = f" epsilon={eps:.4f}"
                    elif self.exploration == "boltzmann":
                        extra = f" temp={self.temp:.4f}"
                    elif self.exploration == "ucb":
                        extra = f" known_states={len(self.visit_counts)}"
                    elif self.exploration == "gradient":
                        extra = f" reward_baseline={self.reward_baseline:.3f}"
                    best_str = f"{best_reward:.1f}" if best_reward != float("-inf") else "n/a"
                    print(f"[{self.param_set}] episode={episode + 1} reward={episode_reward:.1f}"
                          f" best_so_far={best_str} (ep {best_episode}){extra}")
            else:
                print(f"[{self.param_set}] episode={episode + 1} reward={episode_reward:.1f}")

            if is_training and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)
                if steps >= self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

        env.close()

        if is_training:
            # Expose the best result as attributes so it's easy to inspect
            # after run() returns, e.g. dql.best_reward, dql.best_episode.
            self.best_reward = best_reward
            self.best_episode = best_episode
            print(f"\n[{self.param_set}] TRAINING DONE — "
                  f"best_reward={best_reward:.1f} at episode={best_episode} "
                  f"(out of {len(reward_history)} episodes). "
                  f"Model saved to {self.MODEL_FILE}")

        return reward_history

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def compare_exploration_strategies(param_sets, num_episodes=200, seed=None, plot=True):
    """Train one Agent per exploration strategy from scratch, for the same
    number of episodes, and compare their reward curves / convergence speed.
    """
    results = {}
    for pset in param_sets:
        print(f"\n=== Training strategy: {pset} ===")
        agent = Agent(param_set=pset)
        history = agent.run(is_training=True, render=False, max_episodes=num_episodes, seed=seed)
        results[pset] = history

    _save_rewards_csv(results)
    if plot:
        plot_comparison(results, window=min(20, max(1, num_episodes // 10)))

    print("\n=== Summary (rolling-mean reward over the final 20 episodes) ===")
    for pset, history in results.items():
        arr = np.array(history)
        tail = arr[-20:] if len(arr) >= 20 else arr
        print(f"{pset:24s} final_avg_reward={tail.mean():.2f}  best_reward={arr.max():.2f}")

    return results


def _save_rewards_csv(results):
    csv_path = os.path.join(RUNS_DIR, "comparison_rewards.csv")
    max_len = max(len(h) for h in results.values())
    with open(csv_path, "w") as f:
        f.write("episode," + ",".join(results.keys()) + "\n")
        for i in range(max_len):
            row = [str(i + 1)]
            for pset in results:
                h = results[pset]
                row.append(f"{h[i]:.3f}" if i < len(h) else "")
            f.write(",".join(row) + "\n")
    print(f"Saved raw per-episode rewards to {csv_path}")


def plot_comparison(results, window=20):
    plt.figure(figsize=(10, 6))
    for pset, history in results.items():
        arr = np.array(history)
        if len(arr) >= window:
            smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
            x = np.arange(window, len(arr) + 1)
        else:
            smoothed = arr
            x = np.arange(1, len(arr) + 1)
        plt.plot(x, smoothed, label=pset)

    plt.xlabel("Episode")
    plt.ylabel(f"Reward ({window}-episode rolling mean)")
    plt.title("Exploration Strategy Comparison (DQN on FlappyBird-v0)")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = os.path.join(RUNS_DIR, "comparison_reward.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test a FlappyBird DQN, or compare exploration strategies.")
    parser.add_argument("hyperparameters", nargs="?",
                         help="Parameter set name in parameters.yaml (required unless --compare)")
    parser.add_argument("--train", help="Training mode", action="store_true")
    parser.add_argument("--compare", help="Train and compare multiple exploration strategies", action="store_true")
    parser.add_argument("--strategies", nargs="+",
                         default=["flappybird_gradient", "flappybird_ucb"],
                         help="Parameter set names to compare (used with --compare)")
    parser.add_argument("--episodes", type=int, default=None,
                         help="Number of episodes to train (used for --compare, default 200 there; "
                              "also works for a plain --train run, default unlimited there)")
    parser.add_argument("--log-every", type=int, default=None,
                         help="Print progress every N episodes during training (overrides parameters.yaml)")
    args = parser.parse_args()

    if args.compare:
        compare_exploration_strategies(args.strategies, num_episodes=args.episodes or 200)
    else:
        if not args.hyperparameters:
            parser.error("hyperparameters is required unless --compare is used")
        dql = Agent(param_set=args.hyperparameters)
        if args.log_every is not None:
            dql.log_every = args.log_every
        if args.train:
            dql.run(is_training=True, max_episodes=args.episodes)
        else:
            dql.run(is_training=False, render=True)