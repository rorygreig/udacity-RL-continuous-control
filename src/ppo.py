import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch import autograd

from src.policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    """Interacts with and learns from the environment."""

    def __init__(self, env, seed=1, learning_rate=2.5e-4):
        """Initialize an Agent object.
        
        Params
        ======
            env: parallel environment
            num_agents: number of agents in environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.env = env

        # get the default brain
        self.brain_name = env.brain_names[0]
        brain = env.brains[self.brain_name]

        env_info = env.reset(train_mode=True)[self.brain_name]

        state = env_info.vector_observations[0]
        self.state_size = len(state)
        self.action_size = brain.vector_action_space_size
        self.num_agents = len(env_info.agents)

        print(f"\nState size: {self.state_size}, action size: {self.action_size}, number of agents: {self.num_agents}")

        self.policy = Policy(self.state_size, self.action_size, seed)
        self.optimizer = optim.Adam(self.policy.policy_net.parameters(), lr=learning_rate)

    def train(self, n_episodes=2000, discount=0.97, epsilon=0.1, beta=0.01, tmax=1100, sgd_epoch=6):
        """Proximal Policy Optimization.
        Params
        ======
            n_episodes (int): maximum number of training episodes
            epsilon: clipping parameter
            beta: regulation term
            tmax: max number of timesteps per episode
            SGD_epoch
        """

        print("Training PPO on continuous control")

        # keep track of progress
        mean_rewards = []

        for i_episode in tqdm(range(n_episodes)):

            # collect trajectories
            old_probs, states, actions, rewards = self.collect_trajectories(tmax=tmax)
            total_rewards = np.sum(rewards, axis=0)

            # gradient ascent step
            for _ in range(sgd_epoch):
                loss = -self.clipped_surrogate(old_probs, states, rewards, discount, epsilon, beta)

                # with autograd.detect_anomaly():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del loss

            epsilon *= .999
            beta *= .997

            # get the average reward for each agent
            score = np.mean(total_rewards)
            mean_rewards.append(score)

            if i_episode % 20 == 0:
                print(f"Episode: {i_episode}, score: {score}")

        self.env.close()

        return mean_rewards

    # collect trajectories for all unity agents in environment
    def collect_trajectories(self, tmax):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)

        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        for t in range(tmax):
            with torch.no_grad():
                states = torch.from_numpy(states).float()
                actions, probs = self.policy.get_action_probs(states)
                actions = actions.numpy()
                probs = probs.numpy()
                env_info = self.env.step(actions)[self.brain_name]

                state_list.append(states)
                reward_list.append(env_info.rewards)
                prob_list.append(probs)
                action_list.append(actions)

                states = env_info.vector_observations
                scores += env_info.rewards

            if np.any(env_info.local_done):  # exit loop if episode finished
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, action_list, reward_list

    def clipped_surrogate(self, old_log_probs, states, rewards, discount, epsilon, beta):
        discount = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        # normalize rewards
        mean = np.mean(rewards_future)
        std = np.std(rewards_future) + 1.0e-10
        rewards_normalized = (rewards_future - mean) / std

        # convert everything into pytorch tensors and move to gpu if available
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device).unsqueeze(2)

        assert rewards.shape == torch.Size([1001, self.num_agents, 1])

        # convert states to probabilities
        _, new_log_probs = self.policy.get_action_probs(torch.stack(states).float().to(device))

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
        entropy = -(new_probs * torch.log(old_probs) + (1.0 - new_probs) * torch.log(1.0 - old_probs))

        regularized_surrogate = clipped_surrogate + beta * entropy

        return torch.mean(regularized_surrogate)

    def store_weights(self, filename='checkpoint.pth'):
        print("Storing weights")
        torch.save(self.policy.policy_net.state_dict(), "weights/" + filename)

    def run_with_stored_weights(self, filename='"final_weights.pth"'):
        # load stored weights from training
        self.policy.policy_net.load_state_dict(torch.load("weights/" + filename))

        env_info = self.env.reset(train_mode=False)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)

        i = 0
        while True:
            i += 1
            states = torch.from_numpy(states).float().to(device)
            actions, _ = self.policy.get_action_probs(states)
            actions = actions.cpu().detach().numpy()

            env_info = self.env.step(actions)[self.brain_name]

            states = env_info.vector_observations
            dones = env_info.local_done
            scores += env_info.rewards

            if np.any(dones):
                break
        print(f'Ran for {i} episodes. Final score (averaged over agents): {np.mean(scores)}')


