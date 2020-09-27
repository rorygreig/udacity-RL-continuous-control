import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, learning_rate):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        # self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        pass
        # Save experience in replay memory
        # self.memory.add(state, action, reward, next_state, done)
        #
        # # Learn every update_network_interval time steps.
        # self.t_step += 1
        # if self.t_step % update_network_interval == 0:
        #     # If enough samples are available in memory, get random subset and learn
        #     if len(self.memory) > self.batch_size:
        #         experiences = self.memory.sample()
        #         self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # # Epsilon-greedy action selection
        # if random.random() > eps:
        #     state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #     self.qnetwork_local.eval()
        #     with torch.no_grad():
        #         action_values = self.qnetwork_local(state)
        #     self.qnetwork_local.train()
        #
        #     return np.argmax(action_values.cpu().data.numpy())
        #
        # # else return random action
        # return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma=0.99):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones = experiences
        #
        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #
        # # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        #
        # # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        # # Minimize the loss
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model, tau=1e-3):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
