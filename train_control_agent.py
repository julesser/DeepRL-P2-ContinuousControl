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
    max_score = -np.Inf
    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)
            
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states                               # roll over states to next time step
            score += rewards

            if np.any(dones):
                break 
        scores_deque.append(score)
        scores.append(np.mean(score))
        
        print('\rEpisode {}\tAverage 100 Score: {:.2f}\t Mean Score: {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), end="")
        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage 100 Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if (np.mean(scores_deque) >= 31):
                break
    return scores


# 1. Create environment
env = UnityEnvironment(
    # file_name="simulator/Reacher_Single/Reacher.x86_64")
    file_name="simulator/Reacher_Multi/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# 2. Create agent
agent = Agent(state_size, action_size, random_seed=14)

# 3. Roll out DQN algorithm
scores, avgs = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
