import argparse
from src.reacher_env import ReacherMultiAgentEnv

from src.ddpg.ddpg import DDPG
from src.ppo.ppo import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Whether to train or load weights from file",
                        action='store_const', const=True, default=False)
    parser.add_argument("--ppo", help="Whether to use PPO instead of DDPG",
                        action='store_const', const=True, default=False)
    parsed_args = parser.parse_args()
    train = parsed_args.train
    use_ppo = parsed_args.ppo

    env = ReacherMultiAgentEnv("./Reacher_Linux_many/Reacher.x86_64")

    algo = PPO(env) if use_ppo else DDPG(env)

    if train:
        scores = algo.train()
    else:
        algo.run_with_stored_weights()


if __name__ == "__main__":
    main()
