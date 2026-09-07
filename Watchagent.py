import os

# Headless rendering setup (needed on Kaggle / servers without a display).
# Safe to leave in even if you do have a display, since we're saving a video
# rather than opening a live window.
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-dir")
os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)

import argparse

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import yaml

import flappy_bird_gymnasium  # noqa: F401  (registers the FlappyBird-v0 env)
from dqn import DQN

RUNS_DIR = "runs"


def watch_agent(param_set, num_episodes=20, max_steps=3000, epsilon=0.0,
                 parameters_file="parameters.yaml", output_video=None):
    with open(parameters_file, "r") as f:
        params = yaml.safe_load(f)[param_set]

    use_lidar = params.get("use_lidar", False)
    hidden_dim = params.get("hidden_dim", 256)
    model_path = os.path.join(RUNS_DIR, f"{param_set}.pt")

    if output_video is None:
        output_video = os.path.join(RUNS_DIR, f"{param_set}_best_episode.mp4")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=use_lidar)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"state_dim={state_dim} action_dim={action_dim} use_lidar={use_lidar} hidden_dim={hidden_dim}")

    policy_dqn = DQN(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    policy_dqn.load_state_dict(torch.load(model_path, map_location=device))
    policy_dqn.eval()
    print(f"Loaded {model_path}")

    best_reward = float("-inf")
    best_reward_episode = -1
    best_steps = -1
    best_steps_episode = -1
    best_frames = []

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        episode_frames = []

        import random  # local import keeps epsilon optional/rare-path

        while not done and step < max_steps:
            episode_frames.append(env.render())

            if epsilon > 0.0 and random.random() < epsilon:
                action = env.action_space.sample()  # optional exploration during eval
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_dqn(state_tensor)
                action = q_values.argmax(dim=1).item()  # exploit trained policy

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        if total_reward > best_reward:
            best_reward = total_reward
            best_reward_episode = episode
            best_frames = episode_frames

        if step > best_steps:
            best_steps = step
            best_steps_episode = episode

        print(f"episode={episode} reward={total_reward:.1f} steps={step}")

    env.close()

    print("=" * 40)
    print(f"Ran {num_episodes} episodes on '{param_set}' (epsilon={epsilon})")
    print(f"Best reward: {best_reward:.1f}  (episode {best_reward_episode})")
    print(f"Most steps survived: {best_steps}  (episode {best_steps_episode})")
    print("=" * 40)

    if len(best_frames) > 0:
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.axis("off")
        img = ax.imshow(best_frames[0])

        def update(i):
            img.set_array(best_frames[i])
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=len(best_frames), interval=33, blit=True)
        os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

        try:
            ani.save(output_video, writer="ffmpeg", fps=30)
            print(f"Saved best-reward episode video to: {output_video}")
        except (FileNotFoundError, ValueError) as e:
            # ffmpeg isn't installed / available -> fall back to an animated GIF
            gif_path = os.path.splitext(output_video)[0] + ".gif"
            print(f"ffmpeg writer failed ({e}); falling back to GIF at {gif_path}")
            ani.save(gif_path, writer="pillow", fps=30)
            print(f"Saved best-reward episode GIF to: {gif_path}")

        plt.close(fig)
    else:
        print("No frames captured (episode length was 0).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained FlappyBird agent play and save the best episode as video.")
    parser.add_argument("param_set", nargs="?", default="flappybird_ucb_lidar",
                         help="Parameter set name in parameters.yaml (default: flappybird_ucb_lidar)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes to run")
    parser.add_argument("--max-steps", type=int, default=3000, help="Safety cap on steps per episode")
    parser.add_argument("--epsilon", type=float, default=0.0,
                         help="Optional random-action probability during eval (default 0.0 = fully greedy)")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default runs/<param_set>_best_episode.mp4)")
    args = parser.parse_args()

    watch_agent(args.param_set, num_episodes=args.episodes, max_steps=args.max_steps,
                epsilon=args.epsilon, output_video=args.output)