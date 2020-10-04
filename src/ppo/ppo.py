import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from collections import deque

from src.ppo.policy import Policy
from src.plotting import plot_scores


class PPO:
    def __init__(self, env, seed=1, learning_rate=1e-4, target_reward=30.0):
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

        self.policy = Policy(self.env.state_size, self.env.action_size, self.env.num_agents, seed)
        self.optimizer = optim.Adam(self.policy.policy_net.parameters(), lr=learning_rate)

        self.target_reward = target_reward
        self.checkpoint_period = 50

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
        recent_scores = deque(maxlen=100)
        scores = []

        for i_episode in tqdm(range(n_episodes)):
            old_probs, states, actions, rewards = self.collect_trajectories(tmax=tmax)

            # gradient ascent step
            for _ in range(sgd_epoch):
                self.optimizer.zero_grad()
                loss = -self.policy.surrogate(old_probs, states, rewards, discount, epsilon, beta)
                loss.backward()
                self.optimizer.step()

            epsilon *= .999
            beta *= .997

            # get the average reward for each agent
            total_rewards = np.sum(rewards, axis=0)
            score = np.mean(total_rewards) / 10.0
            scores.append(score)
            recent_scores.append(score)
            average_score = np.mean(recent_scores)

            if i_episode % 20 == 0:
                print(f"Episode: {i_episode}, score: {score}")

            print(f"\nEpisode {i_episode}\tAverage Score: {average_score:.2f}\tScore: {score:.2f}")
            if i_episode % self.checkpoint_period == 0:
                self.store_weights('checkpoint')
                plot_scores(scores)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(recent_scores)))

            if average_score > self.target_reward:
                print("Reached target average score, finishing training")
                break

        self.env.close(terminate=True)

        return scores

    # collect trajectories for all unity agents in environment
    def collect_trajectories(self, tmax):
        states = self.env.reset()
        scores = np.zeros(self.env.num_agents)

        state_list = []
        reward_list = []
        prob_list = []
        action_list = []

        for t in range(tmax):
            with torch.no_grad():
                states = torch.from_numpy(states).float()
                actions, probs = self.policy.get_action_probs(states)
                actions = actions.numpy()

                next_states, rewards, dones, _ = self.env.step(actions)

                state_list.append(states.numpy())
                reward_list.append(rewards)
                prob_list.append(probs.numpy())
                action_list.append(actions)

                states = next_states
                scores += rewards

            if np.any(dones):  # exit loop if episode finished
                break

        return prob_list, state_list, action_list, reward_list

    def store_weights(self, filename='checkpoint.pth'):
        print("Storing weights")
        torch.save(self.policy.policy_net.state_dict(), "weights/" + filename)

    def run_with_stored_weights(self, filename='"final_weights.pth"'):
        # load stored weights from training
        self.policy.policy_net.load_state_dict(torch.load("weights/" + filename))

        states = self.env.reset(train_mode=False)
        scores = np.zeros(self.env.num_agents)

        i = 0
        while True:
            i += 1
            with torch.no_grad():
                states = torch.from_numpy(states).float()
                actions, _ = self.policy.get_action_probs(states)
                actions = actions.numpy()

                next_states, rewards, dones, _ = self.env.step(actions)

                states = next_states
                dones = dones
                scores += np.mean(rewards) / 10.0

            if np.any(dones):
                break
        print(f'Ran for {i} episodes. Final score (averaged over agents): {np.mean(scores)}')
        self.env.close(terminate=True)


