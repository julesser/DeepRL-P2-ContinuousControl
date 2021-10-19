from unityagents import UnityEnvironment
import numpy as np

# 1. Create environment
env = UnityEnvironment(
    # file_name="simulator/Reacher_Single/Reacher.x86_64")  # one agent
    file_name="simulator/Reacher_Multi/Reacher.x86_64")  # 20 agents
# env = UnityEnvironment(file_name="simulator/Reacher_Parallel/Reacher.x86_64") # 20 parallel agents
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# 2. Perform random actions
# initialize the score (for each agent)
scores = np.zeros(num_agents)
while True:
    # select an action (for each agent)
    actions = np.random.randn(num_agents, action_size)
    actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
    # send all actions to tne environment
    env_info = env.step(actions)[brain_name]
    # get next state (for each agent)
    next_states = env_info.vector_observations
    rewards = env_info.rewards  # get reward (for each agent)
    dones = env_info.local_done  # check if episode is finished
    scores += env_info.rewards  # update the score (for each agent)
    states = next_states  # roll over states to next time step
    if np.any(dones):
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()
