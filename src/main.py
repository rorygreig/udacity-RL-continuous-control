import argparse
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from src.ppo import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Whether to train or load weights from file",
                        action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    train = parsed_args.train

    env = UnityEnvironment(file_name="./Reacher_Linux_many_agents/Reacher.x86_64")
    ppo = PPO(env, solve_threshold=15.0)

    weights_filename = "final_weights.pth"

    if train:
        scores = ppo.train()
        ppo.store_weights(weights_filename)
        plot_scores(scores)
    else:
        ppo.run_with_stored_weights(weights_filename)


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()