import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class Policy:
    def __init__(self, state_size, action_size, num_agents, seed):
        self.policy_net = PolicyNet(state_size, action_size, seed)
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

    def get_action_probs(self, states):
        action_dist = self.policy_net(states)
        actions = action_dist.sample()

        # clamp all actions between -1 and 1
        actions = torch.clamp(actions, -1, 1)

        # calculate log probs from action gaussian distribution
        probs = action_dist.log_prob(actions)

        return actions, probs

    def surrogate(self, old_log_probs, states, rewards, discount, epsilon, beta):
        discount = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        # normalize rewards
        mean = np.mean(rewards_future)
        std = np.std(rewards_future) + 1.0e-10
        rewards_normalized = (rewards_future - mean) / std

        # convert everything into pytorch tensors
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float).unsqueeze(-1)
        assert rewards.shape == torch.Size([1001, self.num_agents, 1])

        states = torch.tensor(states, dtype=torch.float)
        # states = torch.stack(states).float()
        assert states.shape == torch.Size([1001, self.num_agents, self.state_size])

        # convert states to probabilities
        _, new_log_probs = self.get_action_probs(states)
        assert new_log_probs.shape == torch.Size([1001, self.num_agents, self.action_size])

        # ratio for clipping
        ratio = torch.exp(new_log_probs - old_log_probs)

        # clipped function
        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        old_probs = torch.exp(old_log_probs)
        new_probs = torch.exp(new_log_probs)
        entropy = -(new_probs * torch.log(old_probs) + (torch.tensor(1.0) - new_probs) *
                    torch.log(torch.tensor(1.0) - old_probs))

        regularized_surrogate = clipped_surrogate + beta * entropy

        return torch.mean(regularized_surrogate)


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=80, fc2_units=48, fc3_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PolicyNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, fc3_units)

        # std dev of output distribution is a standalone parameter to be optimized
        log_std = -0.5 * torch.ones(action_size, dtype=torch.float)
        self.log_std = torch.nn.Parameter(log_std, requires_grad=True)
        self.mu = nn.Linear(fc2_units, action_size)
        # self.scale = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that parameterises a gaussian distribution for continuous actions."""
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        # x = F.relu(self.fc3(x))

        # take exponential of log_std to guarantee non-negative value
        std_dev = torch.exp(self.log_std)
        return Normal(self.mu(x), std_dev)

