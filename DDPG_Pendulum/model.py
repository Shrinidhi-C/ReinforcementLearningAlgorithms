import numpy as np
import tensorflow as tf
from mechanism.ou_process import OU_Process
class DDPG(object):
    def __init__(self,state_dim,action_dim,memory_capacity,warm_up,tau,train_every,action_max,batch_size,gamma,
                 noise_scale,noise_decay,
                 actor_lr,critic_lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.warm_up = warm_up
        self.tau = tau
        self.train_every = train_every
        self.action_max = action_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.noise_scale = noise_scale
        self.noise_dcay = noise_decay


        self.mem_capacity = memory_capacity
        self.mem_state = np.zeros((self.mem_capacity, state_dim),dtype=np.float32)
        self.mem_n_state=np.zeros((self.mem_capacity,state_dim),dtype=np.float32)
        self.mem_action = np.zeros((self.mem_capacity,action_dim), dtype=np.float32)
        self.mem_reward = np.zeros((self.mem_capacity,),np.float32)
        self.mem_done = np.zeros((self.mem_capacity,),dtype=np.bool)
        self.mem_have_seen = 0
        self.step = 0
        self.index = 0

        self.exploration_noise = OU_Process(self.action_dim,sigma=0.2)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        ## TF graph ##
        self.tf_s_holder = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
        self.tf_sp_holder = tf.placeholder(tf.float32,[None,self.state_dim],name='state_p')
        #self.tf_a_holder = tf.placeholder(tf.float32, [None, self.action_dim],name='action')
        self.tf_r_holder = tf.placeholder(tf.float32, [None,], name='reward')

        self.actor = self.build_actor(self.tf_s_holder,'actor',True)
        self.target_actor = self.build_actor(self.tf_sp_holder,'target_actor',False)

        self.tf_a_maybe_holder = self.actor

        self.critic =         self.build_critic(self.tf_s_holder,self.tf_a_maybe_holder,'critic',True)
        self.target_critic = self.build_critic(self.tf_sp_holder,self.target_actor,'target_critic',False)

        # It is relatively easy to change the loss for a episodic env  i.e.,  r + Q  ->  r  for the end action
        self.td_loss = tf.reduce_mean(tf.square(tf.reshape(self.tf_r_holder,(-1,1))
                                        + self.gamma*self.target_critic-self.critic))
        self.act_loss = -tf.reduce_mean(self.critic)


        self.critic_update = self.create_update_target_ops('critic','target_critic')
        self.actor_update = self.create_update_target_ops('actor','target_actor')

        self.critic_optim = tf.train.AdamOptimizer(self.critic_lr).minimize(self.td_loss,
                                var_list=tf.trainable_variables('critic'))
        self.actor_optim = tf.train.AdamOptimizer(self.actor_lr).minimize(self.act_loss,
                                var_list=tf.trainable_variables('actor'))


        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.sess.run(self.critic_update)
        self.sess.run(self.actor_update)

    def build_actor(self,s,name,trainable):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(s, 32, activation=tf.nn.relu, name='Layer1', trainable=trainable)
            layer2 = tf.layers.dense(layer1,32,activation=tf.nn.relu,name='Layer2',trainable=trainable)
            action = tf.layers.dense(layer2, self.action_dim, activation=tf.nn.tanh, name='action', trainable=trainable)
            return tf.multiply(action, self.action_max, name='output_action')


    def build_critic(self,s,a,name,trainable):
        with tf.variable_scope(name):
            input = tf.concat([s,a],axis=-1)
            layer1 = tf.layers.dense(input,32,activation=tf.nn.relu,name='Layer1',trainable=trainable)
            layer2 = tf.layers.dense(layer1,32,activation=tf.nn.relu,name='Layer2',trainable=trainable)
            layer3 = tf.layers.dense(layer2,1,name='Q',trainable=trainable)
            return layer3

    def create_update_target_ops(self,from_name,target_name):
        target_name_var = {'/'.join(v.name.split('/')[1:]):v for
                           v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,target_name)}
        train_name_var = {'/'.join(v.name.split('/')[1:]): v for
                          v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,from_name)}
        print(target_name_var.keys())
        print (train_name_var.keys())
        pairs = [(target_name_var[name],train_name_var[name]) for name in train_name_var.keys()]
        return [tf.assign(target,self.tau*source+(1-self.tau)*target) for target,source in pairs]

    def select_action(self,state,greedy=False):
        state = state.reshape(1,-1)
        action = self.sess.run(self.actor,feed_dict={self.tf_s_holder:state})[0]
        if not greedy:

            noise = self.exploration_noise.return_noise() * self.noise_scale
            action = np.clip(action + noise , -self.action_max, self.action_max)
        print (action,noise)
        return action

    def decay_noise(self):
        if self.noise_scale > 0.01:
            self.noise_scale = self.noise_scale * self.noise_dcay

    def sample(self):
        indices = np.random.choice(range(min(self.mem_have_seen,self.mem_capacity)),self.batch_size)
        return self.mem_state[indices],self.mem_action[indices],\
               self.mem_n_state[indices],self.mem_reward[indices]  #mem_done is not used

    def perceive(self,state, action, reward, next_state, is_done):
        self.index = self.mem_have_seen % self.mem_capacity
        self.mem_state[self.index] = state
        self.mem_n_state[self.index] = next_state
        self.mem_reward[self.index] = reward
        self.mem_action[self.index] = action
        self.mem_done[self.index] = is_done

        self.mem_have_seen += 1

        if self.mem_have_seen > self.warm_up and self.mem_have_seen%self.train_every==0:
            self.train()

    def train(self):
        states,actions,n_states,rewards = self.sample()

        self.sess.run(self.critic_optim,feed_dict={self.tf_s_holder:states,self.tf_sp_holder:n_states,
                                                   self.tf_r_holder:rewards,self.tf_a_maybe_holder:actions})
        self.sess.run(self.actor_optim,feed_dict={self.tf_s_holder:states})

        self.sess.run(self.critic_update)
        self.sess.run(self.actor_update)