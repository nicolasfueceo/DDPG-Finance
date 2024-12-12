import torch
import torch.optim as optim
import numpy as np
import random
import copy
from src.rl.actor_critic import EIIEActor, EIIECritic
from src.rl.replay_buffer import ReplayBuffer
import torch.nn.functional as F

class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, buffer_capacity, batch_size, seed=42):
        self.actor = EIIEActor(state_dim, action_dim)
        self.critic = EIIECritic(state_dim, action_dim)
        self.target_actor = EIIEActor(state_dim, action_dim)
        self.target_critic = EIIECritic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        random.seed(seed)
        self._update_target_networks(tau=1.0)

    def _update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for t_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
        for t_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)

    def select_action(self, state, portfolio_weights, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            # actor now returns raw logits
            logits = self.actor(state, portfolio_weights)
        self.actor.train()

        # Add noise to logits, not to already-softmaxed probabilities
        noise = (torch.randn_like(logits) * noise_scale).clamp(-0.05, 0.05)
        logits = logits + noise

        # Now apply softmax only once here to get a proper probability distribution
        action_probs = F.softmax(logits, dim=-1)
        return action_probs.detach().cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        # state and next_state are dicts: {"price_tensor", "portfolio_weights"}
        self.replay_buffer.add(
            state["price_tensor"], state["portfolio_weights"],
            action, reward,
            next_state["price_tensor"], next_state["portfolio_weights"],
            done
        )

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        (price_tensors, pws, actions, rewards, next_price_tensors, next_pws, dones) = zip(*batch)

        price_tensors = np.array(price_tensors)
        pws = np.array(pws)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_price_tensors = np.array(next_price_tensors)
        next_pws = np.array(next_pws)
        dones = np.array(dones)

        states = torch.tensor(price_tensors, dtype=torch.float32)
        pws = torch.tensor(pws, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_price_tensors, dtype=torch.float32)
        next_pws = torch.tensor(next_pws, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states, next_pws)
            target_q = rewards + self.gamma * (1 - dones) * self.target_critic(next_states, next_actions, next_pws)
        q_values = self.critic(states, actions, pws)
        critic_loss = torch.nn.MSELoss()(q_values, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states, pws), pws).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._update_target_networks()

    def save_weights(self, path):
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))
        self.actor.eval()
        self.critic.eval()

