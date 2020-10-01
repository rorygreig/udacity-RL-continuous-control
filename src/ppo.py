import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch import autograd

from src.policy import Policy


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

        self.policy = Policy(self.state_size, self.action_size, self.num_agents, seed)
        self.optimizer = optim.Adam(self.policy.policy_net.parameters(), lr=learning_rate)

    def train(self, n_episodes=2000, discount=0.99, epsilon=0.1, beta=0.01, tmax=1100, sgd_epoch=4):
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
            old_probs, states, actions, rewards = self.collect_trajectories(tmax=tmax)
            total_rewards = np.sum(rewards, axis=0)

            # gradient ascent step
            for _ in range(sgd_epoch):
                # with autograd.detect_anomaly():
                self.optimizer.zero_grad()
                loss = -self.policy.surrogate(old_probs, states, rewards, discount, epsilon, beta)
                loss.backward()
                self.optimizer.step()

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
                env_info = self.env.step(actions)[self.brain_name]

                state_list.append(states.numpy())
                reward_list.append(env_info.rewards)
                prob_list.append(probs.numpy())
                action_list.append(actions)

                states = env_info.vector_observations
                scores += env_info.rewards

            if np.any(env_info.local_done):  # exit loop if episode finished
                break

        return prob_list, state_list, action_list, reward_list

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
            with torch.no_grad():
                states = torch.from_numpy(states).float()
                actions, _ = self.policy.get_action_probs(states)
                actions = actions.numpy()

                env_info = self.env.step(actions)[self.brain_name]

                states = env_info.vector_observations
                dones = env_info.local_done
                scores += env_info.rewards

            if np.any(dones):
                break
        print(f'Ran for {i} episodes. Final score (averaged over agents): {np.mean(scores)}')


