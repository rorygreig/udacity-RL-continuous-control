import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from src.ddpg.ddpg_agent import Agent
from src.plotting import plot_scores


class DDPG:
    def __init__(self, env):
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

        self.agent = Agent(self.state_size, self.action_size, random_seed=10)

        self.network_update_period = 20
        self.checkpoint_period = 50

    def train(self, n_episodes=2000, max_t=1100):
        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in tqdm(range(1, n_episodes+1)):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            self.agent.reset()
            episode_scores = np.zeros(self.num_agents)
            for t in range(max_t):
                actions = np.array([self.agent.act(state, add_noise=True) for state in states])
                env_info = self.env.step(actions)[self.brain_name]

                rewards = np.array(env_info.rewards)
                # if np.array(rewards).any():
                #     print("non zero reward")
                rewards = [1.0 if rew > 0.0 else 0.0 for rew in rewards]
                next_states = env_info.vector_observations
                dones = env_info.local_done

                for s, a, r, s_next, d in zip(states, actions, rewards, next_states, dones):
                    self.agent.store_experience(s, a, r, s_next, d)

                if t % self.network_update_period == 0:
                    for i in range(10):
                        self.agent.update_networks()

                states = next_states
                episode_scores += env_info.rewards
                if np.any(env_info.local_done):
                    break

            score = np.mean(episode_scores)
            scores_deque.append(score)
            scores.append(score)

            average_score = np.mean(scores_deque)

            print(f"\nEpisode {i_episode}\tAverage Score: {average_score:.2f}\tScore: {score:.2f}")
            if i_episode % self.checkpoint_period == 0:
                self.store_weights('checkpoint')
                plot_scores(scores)
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

            if average_score > 30:
                print("Reached target average score, finishing training")
                break

        self.store_weights('final')
        return scores

    def store_weights(self, filename_prefix='checkpoint'):
        print("Storing weights")
        torch.save(self.agent.actor_local.state_dict(), "weights/" + filename_prefix + '_actor.pth')
        torch.save(self.agent.critic_local.state_dict(), "weights/" + filename_prefix + '_critic.pth')

    def run_with_stored_weights(self):
        # load stored weights from training
        self.agent.actor_local.load_state_dict(torch.load("weights/final_actor.pth"))
        self.agent.critic_local.load_state_dict(torch.load("weights/final_critic.pth"))

        env_info = self.env.reset(train_mode=False)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)

        i = 0
        while True:
            i += 1
            with torch.no_grad():
                actions = self.agent.act(states, add_noise=False)

                env_info = self.env.step(actions)[self.brain_name]

                states = env_info.vector_observations
                dones = env_info.local_done
                scores += env_info.rewards

            if np.any(dones):
                break
        print(f'Ran for {i} episodes. Final score (averaged over agents): {np.mean(scores)}')
