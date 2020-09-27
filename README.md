# udacity-RL-continuous-control
Implementation of "Continuous Control" project from Udacity Deep Reinforcement Learning Nanodegree

This is a Deep Reinforcement Learning algorithm to solve the "Continuous Control" Unity environment, where the aim is 
to keep a robot arm within a target sphere.

### Environment
The size of the state space is ..., which represents the agent's velocity and perception of nearby objects.

The size of the action space is .., which represents ...

The environment is considered solved when the agent can navigate around towards yellow bananas, while avoiding blue 
bananas. Roughly this corresponds to a minimum score of 15.0, which is calculated from the reward received by
 the agent. Once the average score for the agent over the most recent 100 episodes reaches this threshold 
of 15.0 then it is considered solved and the training stops. This threshold can be changed by editing the value
 of `solve_threshold` passed to the DQN class in `src/main.py`.

### Getting Started

#### Install requirements
```
conda activate drlnd
pip install -r requirements.txt
```



#### Run
Run the main script to load the saved neural network weights and have the trained agent act in the environment:
```
python src/main.py
```
**OR** pass the `--train` flag to run in training mode and use DQN to train the networks:
```
python src/main.py --train
```