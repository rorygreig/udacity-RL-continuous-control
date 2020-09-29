import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Policy:
    def __init__(self, state_size, action_size, seed):
        self.policy_net = PolicyNet(state_size, action_size, seed)

    def get_action_probs(self, states):
        action_dist = self.policy_net(states)
        actions = action_dist.sample().detach()

        # clip all actions between -1 and 1
        actions = torch.clamp(actions, -1, 1)

        # calculate log probs from action gaussian distribution
        probs = action_dist.log_prob(actions)

        return actions, probs


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32, fc3_units=32):
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
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.loc = nn.Linear(fc3_units, action_size)
        self.scale = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that parameterises a gaussian distribution for continuous actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return Normal(self.loc(x), self.scale(x))

