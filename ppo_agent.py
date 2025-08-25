import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256]):
        super(PPONetwork, self).__init__()

        # Shared layers
        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # Policy and value heads
        self.actor_mean = nn.Linear(prev_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        self.critic = nn.Linear(prev_size, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            m.bias.data.fill_(0.0)

    def forward(self, state):
        shared_features = self.shared(state)

        action_mean = torch.tanh(self.actor_mean(shared_features))
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean)).clamp(min=1e-6)
        value = self.critic(shared_features)

        return action_mean, action_std, value

    def get_action(self, state):
        action_mean, action_std, value = self.forward(state)

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_action(self, state, action):
        action_mean, action_std, value = self.forward(state)

        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

class PPOAgent:
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, k_epochs=4, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)

        self.states.append(state)
        self.actions.append(action.cpu().numpy()[0])
        self.log_probs.append(log_prob.cpu().numpy()[0])
        self.values.append(value.cpu().numpy()[0])

        return action.cpu().numpy()[0]

    def store_reward(self, reward):
        self.rewards.append(reward)

    def store_done(self, done):
        self.dones.append(done)

    def update(self):
        if len(self.states) == 0:
            return 0.0

        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)

        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards, old_values)
        returns = advantages + old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for epoch in range(self.k_epochs):
            log_probs, entropy, values = self.policy.evaluate_action(states, actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            entropy_loss = -entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        self.clear_memory()
        return total_loss / self.k_epochs

    def _calculate_gae(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0

        values_1d = values.squeeze()
        bootstrap_value = torch.zeros(1).to(self.device)
        values_with_bootstrap = torch.cat([values_1d, bootstrap_value])

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values_with_bootstrap[i + 1] - values_with_bootstrap[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32).to(self.device)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
