from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from ddpg_agent import Agent


def ddpg(n_episodes=200, max_t=700):
    """Deep Deterministic Policy Gradient (DDPG).

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
    """

    scores_deque = deque(maxlen=100)
    scores = []
    avgs = []
    max_score = -np.Inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        avg = np.mean(scores_deque)
        avgs.append(avg)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(
            i_episode, np.mean(scores_deque), score), end="")
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       'checkpoint_critic.pth')
            break
    return scores, avgs


# 1. Create environment
env = UnityEnvironment(
    file_name="simulator/Reacher_Single/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# 2. Create agent
agent = Agent(state_size, action_size, random_seed=10)

# 3. Roll out DQN algorithm
scores, avgs = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DQN')
plt.plot(np.arange(len(scores)), avgs, c='r', label='Average')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()
plt.show()
