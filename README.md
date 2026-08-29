# Wheat-only Deep RL agent for Kaggriculture

Dueling Double DQN + Prioritized Experience Replay, PyTorch. WHEAT economy only
(no fertilizer, no other crops/animals). See design notes in the chat that produced this.

## Install
    pip install kaggle-environments torch numpy matplotlib

## Files
- config.py          hyperparameters + game constants (from the README)
- state_encoder.py    obs -> global vector (16) + per-unit local 5x5 window vector (221)
- action_space.py     action index <-> game command, legal-action masks
- network.py          Dueling Q-network
- replay_buffer.py    Prioritized (sum-tree) replay buffer
- agent.py            DQNAgent: masked epsilon-greedy acting + Double DQN learning
- env_wrapper.py       drives one game turn: queries UnitPolicy for farmer+hands and
                       MarketPolicy once, merges into {"farmer","hands","market"}, reward = delta-money - penalties
- train.py            training loop, logs metrics/episode to checkpoints/train_log.csv
- evaluate.py          loads trained checkpoints, runs greedy episodes, prints a
                       trajectory, reports mean/median/best/worst/std, compares vs
                       a Buy->Plant->Water->Harvest->Sell heuristic baseline, plots curves
- submit_agent.py      final `wheat_rl_agent(obs)` function, ready for
                       `env.run([wheat_rl_agent, "random"])` or Kaggle submission

## Run
    python train.py          # trains, saves checkpoints/unit_agent_final.pt + market_agent_final.pt
    python evaluate.py       # evaluates + plots + compares to baseline

## Notes / things to tune once you can actually train
- NUM_EPISODES=2000 x 720 steps is a LOT of env steps; start smaller (e.g. 100 episodes)
  to sanity-check reward trends before a long run.
- WHEAT max yield without fertilizer is 4 (not 6) -- config.py already reflects this.
- HIRE cost uses the fibonacci sequence from the README multiplied by an internal
  farmHandCostMult the wrapper doesn't know exactly; the mask is conservative
  (uses raw fib(n) as a lower-bound cost) -- verify against the real env and adjust
  action_space.market_action_mask if BUY orders get rejected in practice.
- PICKUP/DROP masks are left permissive (env no-ops them harmlessly if illegal);
  tighten if you want tighter exploration.
