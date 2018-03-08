import numpy as np
import gym
import matplotlib.pyplot as plt
from utils import PolicyGradient

ENV_NAME = 'LunarLander-v2'
EPISODE = 8000
STEP_LIMIT = 300
LR = 0.002
GAMMA = 0.99

env = gym.make(ENV_NAME).unwrapped
#env.seed(1)
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = PolicyGradient(env,learning_rate=LR,reward_decay=GAMMA,step_limit=STEP_LIMIT) # and other params

running_rewards = []
episode_rewards = []

for i_episode in range(EPISODE):
    state = env.reset()

    for i_step in range(STEP_LIMIT):
        action = agent.choose_action(state)
        next_state,reward,done,info = env.step(action)
        if (i_episode+1)%500 == 0:
            env.render()
        agent.perceive(state,action,reward)
        state = next_state
        if done:
            break

    total_ep_reward = agent.ep_reward[:agent.stored_step].sum()
    if 'running_reward' not in globals():
        running_reward = total_ep_reward
    else:
        running_reward = running_reward * 0.99 + total_ep_reward * 0.01
    running_rewards.append(running_reward)
    episode_rewards.append(total_ep_reward)
    if (i_episode+1)%50 == 0:
        print ('stored step:',agent.stored_step,'i_step',i_step)
        print('episode: ', i_episode+1, 'total episode reward:', total_ep_reward, 'running reward', running_reward)
    agent.learn()
    #if len(test_reward)>1 and total_ep_reward>test_reward[-2]:
    #    agent.save()


#-----plot figure------#
fig,ax =  plt.subplots(1,1)

episode_rewards_line, = ax.plot(episode_rewards,'-',label='episode_reward')
running_rewards_line, = ax.plot(running_rewards,'-',label='running_reward')
plt.legend()
#reward_legend1 = ax.legend(handles=[running_rewards_line],loc='upper left')
#reward_legend2 = ax.legend(handles=[episode_rewards_line],loc='upper left')
#ax.add_artist(reward_legend2)
#ax.add_artist(reward_legend1)
plt.show()

