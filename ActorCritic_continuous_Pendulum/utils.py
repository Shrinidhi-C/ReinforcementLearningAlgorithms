import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# reproducible
# np.random.seed(1)

class ActorCriticNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.mu =    nn.Linear(hidden_size,action_size)
        self.sigma = nn.Linear(hidden_size,action_size)
        self.c1 = nn.Linear(input_size,hidden_size)
        self.v = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        mu = F.tanh(self.mu(out))*2
        sigma = F.softplus(self.sigma(out)) + 0.001
        c1 = F.relu(self.c1(x))
        v = self.v(c1)
        return mu,sigma,v


class a2c(object):
    def __init__(self,env,reward_decay,
                 lr,hidden_size):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.lr = lr

        self.gamma = reward_decay

        self.actor_critic =  ActorCriticNetwork(self.state_dim,hidden_size,self.action_dim)
        self.optim = torch.optim.Adam(self.actor_critic.parameters(),lr = lr,betas=(0.9,0.9))

        self.distribution = torch.distributions.Normal


    def select_action(self,state,greedy=False):
        self.actor_critic.eval()
        state = Variable(torch.Tensor(state))
        mu, sigma, v = self.actor_critic(state)
        if greedy:
            action = mu.cpu().data.numpy()
        else:
            m = self.distribution(mean = mu.view(1,).data,std=sigma.view(1,).data)
            action = m.sample().numpy()
        return action

    def evaluate_state(self,state):
        state = Variable(torch.Tensor(state))
        _,_,value = self.actor_critic(state)
        return value.cpu().data.numpy()[0]

    def perceive(self,states, actions, rewards, is_done):
        self.states = Variable(torch.from_numpy(states))
        self.actions = Variable(torch.from_numpy(actions))
        self.rewards = Variable(torch.from_numpy(rewards))
        self.is_done = is_done

    def learn(self):
        self.actor_critic.train()
        self.optim.zero_grad()

        #v_next =    self.critic(self.states[1:]).view(-1).detach()
        #if self.is_done:
        #    v_next[-1] = 0
        mu,sigma,v = self.actor_critic(self.states[:-1])
        mu = mu.view(-1)
        sigma = sigma.view(-1)
        v=v.view(-1)


        td_targets = self.rewards
        td_error = td_targets - v

        critic_loss = td_error.pow(2)
        m = self.distribution(mean = mu,std=sigma)

        actor_loss = -m.log_prob(self.actions) * td_error.detach() - 0.005*torch.log(m.std)

        total_loss = (actor_loss + critic_loss).mean()
        total_loss.backward()
        #nn.utils.clip_grad_norm(self.actor.parameters(), 0.5)
        self.optim.step()