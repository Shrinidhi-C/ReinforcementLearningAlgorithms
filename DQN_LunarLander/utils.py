import tensorflow as tf
import numpy as np
import random
from collections import deque
import os

# Hyper Parameters for DQN
GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
REPLAY_SIZE = 50000 # experience replay buffer size
BATCH_SIZE = 64 # size of minibatch
UPDATE_TARGET_EVERY = 100
TRAIN_EVERY = 2
slow_target_burnin = 1000
EPSILON = 0.001
ALPHA = 0.6
BETA = 0.4


class DQN():
    def __init__(self,env):
        self.replay_buffer = SumTree(REPLAY_SIZE)
        self.max_priority = EPSILON
        self.time_step = 0
        self.beta = BETA
        self.beta_increment_per_sampling = 1e-4
        self.abs_err_upper = 1
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_target_Q_network()
        self.create_update_target_ops()
        self.create_training_method()
        self.init_op = tf.global_variables_initializer()
        #Init session

        self.session = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=1)
        self.session.run(self.init_op)
        self.session.run(self.assign_ops)

    def save(self):
        if not os.path.exists('saved_models_priority'):
            os.mkdir('saved_models_priority')
        self.saver.save(self.session,'saved_models_priority/DQN_model.ckpt')

    def restore(self):
        self.saver.restore(self.session,'saved_models_priority/')

    def create_Q_network(self):
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
        with tf.variable_scope('Q_network'):
            layer1 = tf.layers.dense(self.state_input,512,activation=tf.nn.relu,name='layer1')
            layer2 = tf.layers.dense(layer1,512,activation=tf.nn.relu,name='layer2')
            self.Q_value = tf.layers.dense(layer2,self.action_dim,name='Q_value')

    def create_target_Q_network(self):
        with tf.variable_scope('target_Q_network'):
            layer1 = tf.layers.dense(self.state_input,512,activation=tf.nn.relu,
                                     kernel_initializer=tf.zeros_initializer(),
                                     bias_initializer=tf.zeros_initializer(),name='layer1')
            layer2 = tf.layers.dense(layer1,512,activation=tf.nn.relu,
                                     kernel_initializer=tf.zeros_initializer(),
                                     bias_initializer=tf.zeros_initializer(),name='layer2')
            self.target_Q_value = tf.layers.dense(layer2,self.action_dim,
                                        kernel_initializer=tf.zeros_initializer(),
                                        bias_initializer=tf.zeros_initializer(),name='Q_value')

    def create_update_target_ops(self):
        #copy Q_network's weights to target_Q_network
        Q_name_var = {'/'.join(v.name.split('/')[1:]):v for v in tf.trainable_variables('Q_network')}
        target_Q_name_var = {'/'.join(v.name.split('/')[1:]):v for v in tf.trainable_variables('target_Q_network')}
        self.assign_ops = [tf.assign(target_Q_name_var[name],v) for name,v in Q_name_var.items()]

    def create_training_method(self):
        self.action_input = tf.placeholder(tf.float32,[None,self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])
        self.weights = tf.placeholder(tf.float32,[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.delta = self.y_input - Q_action
        self.cost = tf.losses.huber_loss(self.y_input,Q_action,weights=self.weights) #tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.add(self.max_priority,[state, one_hot_action, reward, next_state, done])

        if self.replay_buffer.nonempty > 4*BATCH_SIZE:
            self.train_Q_network()


    def egreedy_action(self,state):
        Q_value = self.session.run(self.Q_value,feed_dict={self.state_input:[state]})[0]

        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self,state):
        return np.argmax(self.session.run(self.Q_value,feed_dict = {
          self.state_input:[state]
          })[0])

    def sample(self,batch_size):
        batch_idx, batch_memory, ISWeights = [], [], []

        total = self.replay_buffer.total()
        segment = total/batch_size
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])

        cap_low = self.replay_buffer.capacity-1
        cap_high = cap_low + self.replay_buffer.nonempty
        min_prob = np.min(self.replay_buffer.tree[cap_low:cap_high]) / total

        maxiwi = np.power(self.replay_buffer.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(batch_size):
            a = segment * i
            b = np.clip(segment * (i + 1),0,total)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.replay_buffer.get(lower_bound)
            prob = p / total
            ISWeights.append(self.replay_buffer.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.array(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def train_Q_network(self):
        self.time_step += 1
        # update target Q network
        if self.time_step%UPDATE_TARGET_EVERY == 0 :
            #print ("Updating target Q network...")
            self.session.run(self.assign_ops)

        if self.time_step%TRAIN_EVERY == 0:
            # obtain random minibatch from replay memory
            batch_idx, minibatch, ISWeights = self.sample(BATCH_SIZE)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            # calculate y
            y_batch = []
            if self.time_step<slow_target_burnin:
                target_Q_value_batch = self.session.run(self.Q_value,feed_dict={self.state_input: next_state_batch})
                Q_value_batch = target_Q_value_batch
            else:
                target_Q_value_batch = self.session.run(self.target_Q_value, feed_dict={self.state_input: next_state_batch}) #target_Q_value
                Q_value_batch = self.session.run(self.Q_value,feed_dict={self.state_input:next_state_batch})
            for i in range(0, BATCH_SIZE):
                done = minibatch[i][4]
                if done:
                    y_batch.append(reward_batch[i])
                else:
                    # double Q-learning
                    y_batch.append(reward_batch[i] + GAMMA * target_Q_value_batch[i][np.argmax(Q_value_batch[i])])

            _,delta = self.session.run([self.optimizer,self.delta],feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch,
                self.weights : ISWeights
            })

            # update priority based on delta
            priorities = self.get_priority_from_delta(delta)
            self.max_priority = np.max(priorities)
            for i in range(0,BATCH_SIZE):
                self.replay_buffer.update(batch_idx[i],priorities[i])

            if self.epsilon>0.05:
                self.epsilon=self.epsilon-0.000005

    def get_priority_from_delta(self,delta):
        priority = abs(delta)
        priority += EPSILON
        priority = np.clip(priority, 0, self.abs_err_upper)
        priority = np.power(priority, ALPHA)
        return priority


class SumTree(object):
    def __init__(self, capacity):
        self.write = 0
        self.nonempty = 0
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        self.nonempty += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.nonempty >= self.capacity:
            self.nonempty = self.capacity

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])