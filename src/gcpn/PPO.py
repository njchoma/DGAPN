import os
import gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from .gcpn_policy import GCPN_CReM

from utils.general_utils import get_current_datetime
from utils.graph_utils import state_to_pyg


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
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(ActorCriticGCPN, self).__init__()

        # action mean range -1 to 1
        self.actor = GCPN_CReM(input_dim,
                               emb_dim,
                               nb_edge_types,
                               gnn_nb_layers,
                               gnn_nb_hidden,
                               mlp_nb_layers,
                               mlp_nb_hidden)
        # critic
        self.critic = GCPN_Critic(emb_dim, mlp_nb_layers, mlp_nb_hidden)

    def forward(self):
        raise NotImplementedError

    def act(self, states, candidates, surrogate_model, batch_idx):
        with torch.autograd.no_grad():
            states, new_states, actions, probs = self.actor(states, candidates, surrogate_model, batch_idx)

        actions = actions.squeeze_().tolist()
        action_logprobs = torch.log(probs).squeeze_().tolist()
        states = [states.cpu(), new_states.cpu()]

        return states, actions, action_logprobs

    def evaluate(self, states, candidates, actions):   
        probs = self.actor.evaluate(candidates, actions)

        action_logprobs = torch.log(probs)
        state_value = self.critic(states)

        entropy = probs * action_logprobs

        return action_logprobs, state_value, entropy


def wrap_state(ob):
    adj = ob['adj']
    nodes = ob['node'].squeeze()

    adj = torch.Tensor(adj)
    nodes = torch.Tensor(nodes)

    adj = [dense_to_sparse(a) for a in adj]
    data = Data(x=nodes, edge_index=adj[0][0], edge_attr=adj[0][1])
    return data


class PPO_GCPN(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 gamma,
                 eta,
                 upsilon,
                 K_epochs,
                 eps_clip,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 mlp_nb_layers,
                 mlp_nb_hidden):
        super(PPO_GCPN, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eta = eta
        self.upsilon = upsilon
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCriticGCPN(input_dim,
                                      emb_dim,
                                      nb_edge_types,
                                      gnn_nb_layers,
                                      gnn_nb_hidden,
                                      mlp_nb_layers,
                                      mlp_nb_hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCriticGCPN(input_dim,
                                          emb_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          mlp_nb_layers,
                                          mlp_nb_hidden)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def to_device(self, device):
        self.policy.to(device)
        self.policy_old.to(device)

    def select_action(self, state, candidates, surrogate_model, batch_idx=None):
        device = next(self.policy_old.parameters()).device
        if batch_idx is None:
            batch_idx = torch.empty(len(candidates), dtype=torch.long).fill_(0)
        batch_idx = batch_idx.to(device)

        g = Batch.from_data_list([state]).to(device)
        g_candidates = Batch.from_data_list(candidates).to(device)
        states, actions, action_logprobs = self.policy_old.act(g, g_candidates, surrogate_model, batch_idx)
        return states, actions, action_logprobs

    def update(self, memory):
        device = next(self.policy.parameters()).device

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
        old_states = torch.cat(([m[0] for m in memory.states]),dim=0).to(device)
        old_candidates = Batch().from_data_list([Data(x=m[1]) for m in memory.states]).to(device)
        old_actions = torch.tensor(memory.actions).to(device)
        old_logprobs = torch.tensor(memory.logprobs).to(device)

        # Optimize policy for K epochs:
        print("Optimizing...")

        for i in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, entropies = self.policy.evaluate(old_states,
                                                                     old_candidates,
                                                                     old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs)

            # loss
            advantages = rewards - state_values.detach()
            loss = []

            ratios = ratios.unsqueeze(1)
            for j in range(ratios.shape[1]):
                r = ratios[:,j]
                surr1 = r * advantages
                surr2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantages
                l = -torch.min(surr1, surr2)

                if torch.isnan(l).any():
                    print("found nan in loss")
                    print(l)
                    print(torch.isnan(surr1).any())
                    print(torch.isnan(surr2).any())
                    print(torch.isnan(advantages).any())
                    exit()
                loss.append(l)
            loss = torch.stack(loss, 0).sum(0)
            ## entropy
            loss += self.eta*entropies
            ## baseline
            loss = loss.mean() + self.upsilon*self.MseLoss(state_values, rewards)

            ## take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (i%10)==0:
                print("  {:3d}: Loss: {:7.3f}".format(i, loss))

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n{}".format(repr(self.policy), repr(self.optimizer))


#####################################################
#                   FINAL REWARDS                   #
#####################################################

def get_reward(state, surrogate_model, device):
    g = Batch().from_data_list([state])
    g = g.to(device)
    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g, None)
    reward = pred_docking_score.item() * -1
    return reward


#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_ppo(args, surrogate_model, env):
    ############## Hyperparameters ##############
    render = True
    solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    save_interval = 100         # save model in the interval
    max_episodes = 50000        # max training episodes
    
    max_timesteps = 6           # max timesteps in one episode
    update_timestep = 30        # update policy every n timesteps
    
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    #############################################
    print("lr:", lr, "beta:", betas)

    # logging variables
    dt = get_current_datetime()
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + dt)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    ppo = PPO_GCPN(lr,
                   betas,
                   gamma,
                   args.eta,
                   args.upsilon,
                   K_epochs,
                   eps_clip,
                   args.input_size,
                   args.emb_size,
                   args.nb_edge_types,
                   args.layer_num_g,
                   args.num_hidden_g,
                   args.mlp_num_layer,
                   args.mlp_num_hidden)
    ppo.to_device(device)
    print(ppo)

    surrogate_model = surrogate_model.to(device)
    surrogate_model.eval()
    print(surrogate_model)

    running_reward = 0
    avg_length = 0
    time_step = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    # training loop
    for i_episode in range(1, max_episodes+1):
        state, candidates, done = env.reset()
        starting_reward = get_reward(state, surrogate_model, device)

        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            state, action, action_logprob = ppo.select_action(state, candidates, surrogate_model)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

            state, candidates, done = env.step(action)

            # done and reward may not be needed anymore
            reward = 0

            if (t==(max_timesteps-1)) or done:
                surr_reward = get_reward(state, surrogate_model, device)
                reward = surr_reward-starting_reward

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward
            if done:
                break

        # update if it's time
        if time_step > update_timestep:
            print("\n\nupdating ppo @ episode %d..." % i_episode)
            time_step = 0
            ppo.update(memory)
            memory.clear_memory()
            # save running model
            torch.save(ppo.policy.actor, os.path.join(save_dir, 'running_gcpn.pth'))

        writer.add_scalar("EpSurrogate", -1*surr_reward, i_episode-1)
        rewbuffer_env.append(reward)
        avg_length += t

        # write to Tensorboard
        writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format('test'))
            break

        # save every 500 episodes
        if (i_episode-1) % save_interval == 0:
            torch.save(ppo.policy.actor, os.path.join(save_dir, '{:05d}_gcpn.pth'.format(i_episode)))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = running_reward/log_interval
            
            print('Episode {} \t Avg length: {} \t Avg reward: {:5.3f}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

