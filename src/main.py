import argparse
from src.reacher_env import ReacherMultiAgentEnv

from src.ddpg.ddpg import DDPG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Whether to train or load weights from file",
                        action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    train = parsed_args.train

    env = ReacherMultiAgentEnv("./Reacher_Linux_many/Reacher.x86_64")
    ddpg = DDPG(env)

    if train:
        scores = ddpg.train()
    else:
        ddpg.run_with_stored_weights()


if __name__ == "__main__":
    main()
