import numpy as np
import gym
import matplotlib.pyplot as plt
from utils import DQN

ENV_NAME = 'LunarLander-v2'
EPISODE = 2000
STEP_LIMIT = 300
TEST = 5

env = gym.make(ENV_NAME)
agent = DQN(env)

test_reward = []

for i_episode in range(EPISODE):
    state = env.reset()
    print (i_episode)
    for i_step in range(STEP_LIMIT):
        action = agent.egreedy_action(state)
        next_state,reward,done,info =env.step(action)
        agent.perceive(state,action,reward,next_state,done)
        state = next_state
        if done:
            break

    if (i_episode+1)%20 == 0:
        print (agent.epsilon,agent.time_step)
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP_LIMIT):
                env.render()
                action = agent.action(state)
                state,reward,done,_ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward/TEST
        print (agent.epsilon,agent.time_step)
        print ('episode: ', i_episode, 'Evaluation Average Reward:', ave_reward)
        test_reward.append(ave_reward)
        if len(test_reward)>1 and ave_reward>test_reward[-2]:
            agent.save()
        #if ave_reward >= 200:


#-----plot figure------#
fig,ax =  plt.subplots(1,1)
reward_line, = ax.plot(test_reward,'o-',label='test_reward')
#plt.legend()
reward_legend = ax.legend(handles=[reward_line],loc='upper left')
ax.add_artist(reward_legend)
plt.show()