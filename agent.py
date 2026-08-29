import numpy as np
import torch
import torch.nn.functional as F

from network import DuelingQNetwork
from replay_buffer import PrioritizedReplayBuffer
import config as C


class DQNAgent:
    """One Dueling-Double-DQN agent. Used twice: once for UnitPolicy, once for MarketPolicy,
    each with its own state_dim / n_actions / replay buffer / networks."""

    def __init__(self, state_dim, n_actions, device="cpu"):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        self.online = DuelingQNetwork(state_dim, n_actions).to(device)
        self.target = DuelingQNetwork(state_dim, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=C.LR)
        self.buffer = PrioritizedReplayBuffer(C.REPLAY_CAPACITY, alpha=C.PER_ALPHA, eps=C.PER_EPS)

        self.total_steps = 0
        self.grad_steps = 0

    # ---- exploration schedule ----
    def epsilon(self):
        frac = min(1.0, self.total_steps / C.EPS_DECAY_STEPS)
        return C.EPS_START + frac * (C.EPS_END - C.EPS_START)

    def beta(self):
        frac = min(1.0, self.total_steps / C.PER_BETA_STEPS)
        return C.PER_BETA_START + frac * (C.PER_BETA_END - C.PER_BETA_START)

    # ---- acting ----
    def act(self, state, mask, greedy=False):
        self.total_steps += 1
        valid_idxs = np.flatnonzero(mask)
        if valid_idxs.size == 0:
            return 0  # degenerate fallback; masks should always leave at least PASS/NOOP legal

        if (not greedy) and np.random.rand() < self.epsilon():
            return int(np.random.choice(valid_idxs))

        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).squeeze(0).cpu().numpy()
        q_masked = np.where(mask, q, -np.inf)
        return int(np.argmax(q_masked))

    def remember(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.push(state, action, reward, next_state, done, mask, next_mask)

    # ---- learning ----
    def learn(self):
        if len(self.buffer) < max(C.WARMUP_STEPS, C.BATCH_SIZE):
            return None

        states, actions, rewards, next_states, dones, masks, next_masks, idxs, is_weights = \
            self.buffer.sample(C.BATCH_SIZE, beta=self.beta())

        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, device=self.device)
        is_weights_t = torch.as_tensor(is_weights, device=self.device)

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select best next action with the ONLINE net, evaluate it with the TARGET net.
            next_q_online = self.online(next_states)
            next_q_online = next_q_online.masked_fill(~next_masks_t, -1e9)
            best_next_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target(next_states).gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (1.0 - dones) * C.GAMMA * next_q_target

        td_errors = (q_values - targets).detach().cpu().numpy()
        loss = (is_weights_t * F.smooth_l1_loss(q_values, targets, reduction="none")).mean()  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), C.GRAD_CLIP_NORM)
        self.optimizer.step()

        self.buffer.update_priorities(idxs, td_errors)

        self.grad_steps += 1
        if self.grad_steps % C.TARGET_UPDATE_FREQ == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())
