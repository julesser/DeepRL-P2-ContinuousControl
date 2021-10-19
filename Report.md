# Udacity Deep Reinforcement Learning Nanodegree - <br /> Project 2: Continuous Control
![trained_agent](https://github.com/julesser/DeepRL-P2-ContinuousControl/blob/main/fig/trained_agent.gif)

## Introduction
In this project, I built an reinforcement learning (RL) agent that controls a robotic arm to reach target locations in an environment similar to [Unity's Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher).

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. Consequently, the environment can be categoriezed as episodic, continous control problem.

## Getting Started
### Install Dependencies
    cd /python
    pip install .
### Instructions
- Run `python3 watch_trained_agent.py` to see the trained agent in action.
- Run `python3 watch_random_agent.py` to see an untrained agent performing random actions.
- Run `python3 train_control_agent.py` to re-train the agent.
- `model.py` defines the actor and critic network architectures.
- `agent.py` defines the DDPG agent class.
- `checkpoint_actor.pth` and `checkpoint_critic.pth`are the saved network weights after training.

## Solution Method
The following sections describe the main components of the implemented solution along with design choices. 
### Learning Algorithm & Network Architecture
Deep Deterministic Policy Gradient is used to solve this environment. In this method, two neural networks are used, one as actor and one as critic. The actor network has the state vector as input and action vector as output. The critic network has both state vector and action vector as inputs and estimates the reward. 
### Hyperparameters
The following hyperparameters have been used for training the agent (see `ddpg_agent.py`):

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 128        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-4         # learning rate of the actor 
    LR_CRITIC = 1e-4        # learning rate of the critic

## Results
The implemented RL agent is able to solve the environment in <150 episodes:
![training_results](https://github.com/julesser/DeepRL-P2-ContinuousControl/blob/main/fig/results.png) 
## Ideas for Future Improvements
- Perform a systematic hyperparameter optimization study to improve convergence characteristics.  
- Benchmark against other algorithms, such as TRPO (Trust Region Policy Optimization), PPO (Proximal Policy Optimization) or D4PG (Distributed Distributional Deterministic Policy Gradients), which may obtain more robust results.