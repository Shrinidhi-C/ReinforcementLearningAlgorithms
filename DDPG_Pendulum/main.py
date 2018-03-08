import numpy as np
import gym
import matplotlib.pyplot as plt
from model import DDPG

###### hyper parameters ######

ENV_NAME = 'Pendulum-v0'
EPISODE = 2000
STEP_LIMIT = 200
TEST_EPISODE = 5
TEST_EVERY = 50
TEST_RENDER = True

######## MAIN ###########
if __name__ == '__main__':
    env = gym.make(ENV_NAME).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim,action_dim,memory_capacity=50000,warm_up=500,
                 tau=0.01,train_every=1,action_max=2,batch_size=64,noise_scale=1,noise_decay=1,
                 gamma=0.9,actor_lr=0.001,critic_lr=0.002)

    for i_episode in range(EPISODE):
        state = env.reset()
        train_rewards = 0
        print (i_episode+1)

        for i_step in range(STEP_LIMIT):
            action = agent.select_action(state,greedy=False)
            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            train_rewards += reward.sum()
            if done:
                agent.exploration_noise.init_process()
                break
        agent.decay_noise()
        print (agent.noise_scale)

        print ("reward :",train_rewards)
        if (i_episode+1)%TEST_EVERY == 0:
            total_reward = 0
            for i in range(TEST_EPISODE):
                state = env.reset()
                for j in range(STEP_LIMIT):
                    if TEST_RENDER:
                        env.render()
                    action = agent.select_action(state,greedy=True)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST_EPISODE
            print('episode: ', i_episode+1, 'Evaluation Average Reward:', ave_reward)

            #test_reward.append(ave_reward)
            #if len(test_reward) > 1 and ave_reward > test_reward[-2]:
            #    agent.save()