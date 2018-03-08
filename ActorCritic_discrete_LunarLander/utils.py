import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# reproducible
#np.random.seed(1)

class ActorNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)
    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        logits = self.fc3(out)
        probs = F.softmax(logits,dim=-1)
        return probs,logits

class ValueNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class a2c(object):
    def __init__(self,env,reward_decay,
                 actor_lr,critic_lr,
                 actor_hidden_size,critic_hidden_size):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = reward_decay

        self.actor =  ActorNetwork(self.state_dim,actor_hidden_size,self.action_dim)
        self.critic = ValueNetwork(self.state_dim,critic_hidden_size,1)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(),lr = actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(),lr = critic_lr)

        self.loss_ce = nn.CrossEntropyLoss(reduce=False)

    def select_action(self,state,greedy=False):
        state = Variable(torch.Tensor(state))
        probs,logits = self.actor(state)
        if greedy:
            action = np.argmax(probs.cpu().data.numpy()[0])
        else:
            action = np.random.choice(self.action_dim, p=probs.cpu().data.numpy()[0])
        return action

    def evaluate_state(self,state):
        state = Variable(torch.Tensor(state))
        value = self.critic(state).cpu().data.numpy()[0]
        return value

    def perceive(self,states, actions, rewards, is_done):
        self.states = Variable(torch.from_numpy(states))
        self.actions = Variable(torch.from_numpy(actions))
        self.rewards = Variable(torch.from_numpy(rewards))
        self.is_done = is_done

    def learn(self):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        #v_next =    self.critic(self.states[1:]).view(-1).detach()
        #if self.is_done:
        #    v_next[-1] = 0
        v = self.critic(self.states[:-1]).view(-1)

        td_targets = self.rewards
        td_error = (td_targets - v)

        # train critic
        critic_loss = td_error.pow(2).mean()
        critic_loss.backward()
        #nn.utils.clip_grad_norm(self.critic.parameters(), 0.5) #TODO
        self.critic_optim.step()

        #train actor
        probs,logits = self.actor(self.states[:-1])
        actor_loss = torch.mean(self.loss_ce(logits,self.actions) * td_error.detach())
        actor_loss.backward()
        #nn.utils.clip_grad_norm(self.actor.parameters(), 0.5) #TODO
        self.actor_optim.step()
