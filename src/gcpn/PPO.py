import gym
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from .gcpn_policy import GCPN

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


#################################################
#                   GCPN PPO                    #
#################################################

class GCPN_Critic(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(GCPN_Critic, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)


class ActorCriticGCPN(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_nb_hidden_kernel,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(ActorCriticGCPN, self).__init__()

        # action mean range -1 to 1
        self.actor =  GCPN(input_dim,
                           emb_dim,
                           nb_edge_types,
                           gnn_nb_layers,
                           gnn_nb_hidden,
                           gnn_nb_hidden_kernel,
                           mlp_nb_layers,
                           mlp_nb_hidden)
        # critic
        self.critic = GCPN_Critic(emb_dim, mlp_nb_layers, mlp_nb_hidden)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action, probs = self.actor(state)
        action_logprob = torch.log(probs)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action
    
    def evaluate(self, state, action):   
        probs, X_agg = self.actor.evaluate(state, action)
        entropy = 0.0 # TODO get from distribution in gcpn policy
        
        action_logprobs = torch.log(probs)
        state_value = self.critic(X_agg)
        
        return action_logprobs, state_value, entropy


def wrap_state(ob):
    adj = ob['adj']
    nodes = ob['node'].squeeze()

    adj = torch.Tensor(adj)
    nodes = torch.Tensor(nodes)


    adj = [dense_to_sparse(a) for a in adj]
    data = Data(x=nodes, edge_index=adj[0][0], edge_attr=adj[0][1])
    return data


class PPO_GCPN:
    def __init__(self,
                 lr,
                 betas,
                 gamma,
                 K_epochs,
                 eps_clip,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_nb_hidden_kernel,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCriticGCPN(input_dim,
                                      emb_dim,
                                      nb_edge_types,
                                      gnn_nb_layers,
                                      gnn_nb_hidden,
                                      gnn_nb_hidden_kernel,
                                      mlp_nb_layers,
                                      mlp_nb_hidden).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCriticGCPN(input_dim,
                                          emb_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          gnn_nb_hidden_kernel,
                                          mlp_nb_layers,
                                          mlp_nb_hidden).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    
    def select_action(self, state, memory):
        state = wrap_state(state).to(device)
        action = self.policy_old.act(state, memory)
        return action
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = Batch().from_data_list(memory.states).to(device)
        old_actions = torch.squeeze(torch.tensor(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        print("Optimizing...")
        for i in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, _ = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            loss = []
            for j in range(ratios.shape[1]):
                r = ratios[:,j]
                surr1 = r * advantages
                surr2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantages
                l = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards)
                loss.append(l)
            loss = torch.stack(loss)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            if (i%5)==0:
                print("  {:3d}: Loss: {:7.3f}".format(i, loss.mean()))
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n{}".format(repr(self.policy), repr(self.optimizer))





def train_ppo(args, env):
    print("WARNING!!! Only using ONE input bond graph right now")
    print("INFO: Not training with entropy term in loss")

    ############## Hyperparameters ##############
    render = True
    solved_reward = 10          # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 400       # update policy every n timesteps
    K_epochs = 20               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    #############################################

    ob = env.reset()
    nb_edge_types = ob['adj'].shape[0]
    input_dim = ob['node'].shape[2]
    
    ppo = PPO_GCPN(lr,
                   betas,
                   gamma,
                   K_epochs,
                   eps_clip,
                   input_dim,
                   args.emb_size,
                   nb_edge_types,
                   args.layer_num_g,
                   args.num_hidden_g,
                   args.num_hidden_g,
                   args.mlp_num_layer,
                   args.mlp_num_hidden)
    
    print(ppo)
    memory = Memory()
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            



            # update if its time
            if time_step % update_timestep == 0:
                print("updating ppo")
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if (((i_episode+1)%20)==0) and render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = running_reward/log_interval
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

