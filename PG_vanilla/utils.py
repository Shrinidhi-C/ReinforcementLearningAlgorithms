import numpy as np
import tensorflow as tf

# reproducible
#np.random.seed(1)
#tf.set_random_seed(1)

class PolicyGradient(object):
    def __init__(self,env,learning_rate = 0.01,reward_decay = 0.95,step_limit = 400):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_state = np.zeros((step_limit,self.state_dim),dtype=np.float32)
        self.ep_action = np.zeros((step_limit,),dtype=np.int32)
        self.ep_reward = np.zeros((step_limit,),dtype=np.float32)
        self.stored_step = 0

        self._build_net()

        self.init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init_op)

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
            self.tf_action = tf.placeholder(tf.int32, [None,], name="action")
            self.tf_value = tf.placeholder(tf.float32, [None,], name="value")
        with tf.variable_scope('network'):
            layer1 = tf.layers.dense(self.tf_state,16,activation=tf.nn.tanh,name='layer1')
            layer2 = tf.layers.dense(layer1,32,activation=tf.nn.tanh,name='layer2')
            self.logits = tf.layers.dense(layer2,self.action_dim,name='logits')
            self.prob = tf.nn.softmax(self.logits,name='act_prob')
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.tf_action)
            loss = tf.reduce_mean(neg_log_prob * self.tf_value)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self,state):
        prob_weights = self.session.run(self.prob, feed_dict={self.tf_state: state[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def perceive(self,state,action,reward):
        self.ep_state[self.stored_step] = state
        self.ep_action[self.stored_step] = action
        self.ep_reward[self.stored_step] = reward
        self.stored_step+=1

    def learn(self):
        discounted_normed_ep_value = self._process_value()
        assert len(discounted_normed_ep_value) == self.stored_step
        self.session.run(self.train_op,feed_dict={self.tf_state:self.ep_state[:self.stored_step],
                                                  self.tf_action:self.ep_action[:self.stored_step],
                                                  self.tf_value:discounted_normed_ep_value})
        self.stored_step = 0
        return discounted_normed_ep_value

    def _process_value(self):
        R = 0
        for i in range(self.stored_step-1,-1,-1):
            R = self.ep_reward[i] + self.gamma * R
            self.ep_reward[i] = R
        rewards = self.ep_reward[:self.stored_step]
        #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        return rewards
