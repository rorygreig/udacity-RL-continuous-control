import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    """Interacts with and learns from the environment."""

    def __init__(self, env, seed=1, learning_rate=1e-4):
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

    def train(self, n_episodes=2000, discount=0.995, epsilon=0.1, beta=0.01, tmax=120, sgd_epoch=4):
        """Proximal Policy Optimization.
        Params
        ======
            n_episodes (int): maximum number of training episodes
            epsilon: clipping parameter
            beta: regulation term
            tmax: max number of timesteps per episode
            SGD_epoch
        """

        print(f"Training PPO on continuous control")

        # keep track of progress
        mean_rewards = []

        for i_episode in tqdm(range(n_episodes)):

            # collect trajectories
            old_probs, states, actions, rewards = self.collect_trajectories(tmax=tmax)
            total_rewards = np.sum(rewards, axis=0)

            # gradient ascent step
            for _ in range(sgd_epoch):
                loss = -self.clipped_surrogate(old_probs, states, rewards, discount=discount, epsilon=epsilon,
                                               beta=beta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del loss

            epsilon *= .999
            beta *= .995

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
            states = torch.from_numpy(states).float().to(device)
            actions, probs = self.policy.get_action_probs(states)
            actions = actions.cpu().detach().numpy()
            probs = probs.cpu().detach().numpy()
            env_info = self.env.step(actions)[self.brain_name]

            state_list.append(states)
            reward_list.append(env_info.rewards)
            prob_list.append(probs)
            action_list.append(actions)

            # update to next states
            states = env_info.vector_observations
            scores += env_info.rewards
            if np.max(env_info.rewards) > 0.0:
                print(f"\nMax reward: {np.max(env_info.rewards)}")

            if np.any(env_info.local_done):  # exit loop if episode finished
                break

        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, action_list, reward_list

    # clipped surrogate function
    # similar as -policy_loss for REINFORCE, but for PPO
    def clipped_surrogate(self, old_probs, states, rewards, discount, epsilon, beta):
        discount = discount ** np.arange(len(rewards))
        rewards = np.asarray(rewards) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        # normalize rewards
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to probabilities
        _, new_probs = self.policy.get_action_probs(torch.stack(states))

        # ratio for clipping
        ratio = new_probs / old_probs

        # clipped function
        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        rewards = rewards.unsqueeze(2)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs * torch.log(old_probs + 1.e-10) + (1.0 - new_probs) * torch.log(1.0 - old_probs + 1.e-10))

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta * entropy)

    def store_weights(self, filename='checkpoint.pth'):
        print("Storing weights")
        torch.save(self.policy.policy_net.state_dict(), "weights/" + filename)

    def run_with_stored_weights(self, filename='"final_weights.pth"'):
        # load stored weights from training
        self.policy.policy_net.load_state_dict(torch.load("weights/" + filename))

        env_info = self.env.reset(train_mode=False)[self.brain_name]  # reset the environment
        scores = np.zeros(self.num_agents)  # initialize the score (for each agent)
        while True:
            actions = np.random.randn(self.num_agents, self.action_size)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            env_info = self.env.step(actions)[self.brain_name]  # send all actions to tne environment
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            if np.any(dones):  # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


