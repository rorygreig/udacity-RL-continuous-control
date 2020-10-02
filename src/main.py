import argparse
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from src.ddpg.ddpg import DDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Whether to train or load weights from file",
                        action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    train = parsed_args.train

    env = UnityEnvironment(file_name="./Reacher_Linux_many_agents/Reacher.x86_64")
    ddpg = DDPG(env)

    weights_filename = "final_weights.pth"

    if train:
        scores = ddpg.train()
        ddpg.store_weights(weights_filename)
        plot_scores(scores)
    else:
        ddpg.run_with_stored_weights(weights_filename)


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == "__main__":
    main()
