import numpy as np
import gym
import matplotlib.pyplot as plt
from utils import a2c

ENV_NAME = 'Pendulum-v0'

EPISODE = 5000
ROLLOUT_PER_EPISODE = 40
STEP_PER_ROLLOUT = 5
LR = 0.0001
GAMMA = 0.9
units = 100


env = gym.make(ENV_NAME).unwrapped
#env.seed(2)
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = a2c(env,reward_decay=GAMMA,lr=LR,hidden_size=units)

episode_rewards = []

def discount_reward(rewards,gamma,final_r):
    R = final_r
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        rewards[i] = R
    return rewards

def rollout(env,init_state,steps,render=False):
    states = np.zeros((steps+1,env.observation_space.shape[0]),dtype=np.float32)  # one more element!!
    rewards = np.zeros((steps,),dtype=np.float32)
    actions = np.zeros((steps,),dtype=np.float32)
    is_done = False
    final_r = 0
    state = init_state

    for j in range(steps):
        action = agent.select_action(state.reshape(1,-1))
        action = action.clip(-2,2)
        next_state, reward, done, _ = env.step(action)
        if render:
            env.render()
        states[j] = state
        actions[j] = action
        rewards[j] = reward
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = env.reset()
            break
    if not is_done:
        final_r = agent.evaluate_state(final_state.reshape(1,-1))[0] #used in n-step TD?

    states[j+1] = state
    return states[:j+2],actions[:j+1],rewards[:j+1],final_r,state,is_done

if __name__=='__main__':
    state = env.reset()
    step_reward = 0
    RENDER = False
    train_rewards = []
    running_rewards = []

    for i_episode in range(EPISODE):
        state = env.reset()
        episode_reward = 0
        for i_rollout in range(ROLLOUT_PER_EPISODE):
            states, actions, rewards, final_r, state, is_done = rollout(env, state, STEP_PER_ROLLOUT, RENDER)
            episode_reward += rewards.sum()
            rewards = discount_reward((rewards+8.1)/8.1,GAMMA,final_r)
            agent.perceive(states, actions, rewards, is_done)
            agent.learn()
            if is_done:
                print('is_done')
                break

        if (i_episode+1)%50 == 0:
            print ("episode: ",i_episode+1, "reward",episode_reward)
        train_rewards.append(episode_reward)
        if len(running_rewards)==0:
            running_rewards.append(episode_reward)
        else:
            running_rewards.append(running_rewards[-1]*.99 + episode_reward*.01)

    fig,ax =  plt.subplots(1,1)
    episode_rewards_line, = ax.plot(train_rewards,'-',label='train_reward')
    episode_rewards_line, = ax.plot(running_rewards, '-', label='running_reward')
    plt.legend()
    plt.show()

