import argparse
from unityagents import UnityEnvironment

from src.ddpg.ddpg import DDPG
from src.plotting import plot_scores


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
        ddpg.run_with_stored_weights()


if __name__ == "__main__":
    main()
