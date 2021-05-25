import csv
import gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from .gcpn_policy import GCPN, GNN_Embed

from utils.graph_utils import state_to_pyg

from gcpn.adtgpu.get_reward import get_dock_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                 mlp_nb_hidden,
                 prob_redux_factor):
        super(ActorCriticGCPN, self).__init__()

        # action mean range -1 to 1
        self.actor = GCPN(input_dim,
                           emb_dim,
                           nb_edge_types,
                           gnn_nb_layers,
                           gnn_nb_hidden,
                           gnn_nb_hidden_kernel,
                           mlp_nb_layers,
                           mlp_nb_hidden,
                           prob_redux_factor)
        # critic
        self.critic_embed = GNN_Embed(gnn_nb_hidden,
                                   gnn_nb_layers,
                                   gnn_nb_hidden_kernel,
                                   1,
                                   input_dim,
                                   emb_dim)
        self.critic = GCPN_Critic(emb_dim, mlp_nb_layers, mlp_nb_hidden)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action, probs = self.actor(state)
        action_logprob = torch.log(probs)
        
        memory.states.append(state.to_data_list()[0])
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action
    
    def evaluate(self, state, action):   
        probs, _ = self.actor.evaluate(state, action)

        batch = state.batch
        X = self.critic_embed(state)
        X_agg = pyg.nn.global_add_pool(X, batch)
        state_value = self.critic(X_agg)
        
        action_logprobs = torch.log(probs)

        entropy = (probs * action_logprobs).sum(1)
        
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
                 eta,
                 upsilon,
                 K_epochs,
                 eps_clip,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_nb_hidden_kernel,
                 mlp_nb_layers,
                 mlp_nb_hidden,
                 prob_redux_factor):
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
                                      gnn_nb_hidden_kernel,
                                      mlp_nb_layers,
                                      mlp_nb_hidden,
                                      prob_redux_factor).to(device)
        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=lr, betas=betas)
        self.optimizer_critic = torch.optim.Adam(list(self.policy.critic_embed.parameters()) + list(self.policy.critic.parameters()), lr=lr, betas=betas)

        self.policy_old = ActorCriticGCPN(input_dim,
                                          emb_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          gnn_nb_hidden_kernel,
                                          mlp_nb_layers,
                                          mlp_nb_hidden,
                                          prob_redux_factor).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    
    def select_action(self, state, memory, env):
        g = state_to_surrogate_graph(state, env).to(device)
        # state = wrap_state(state).to(device)
        with torch.autograd.no_grad():
            action = self.policy_old.act(g, memory)
        return action
    
    def update(self, memory, i_episode, writer=None):
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
            logprobs, state_values, entropies = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # loss
            advantages = rewards - state_values.detach()   
            loss = []
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
            actor_loss = loss.mean()
            ## take gradient step
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            critic_loss = self.MseLoss(state_values, rewards)
            ## take gradient step
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            if (i%10)==0:
                print("  {:3d}: Actor Loss: {:7.3f}, Critic Loss: {:7.3f}".format(i, actor_loss, critic_loss))

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n{}\n{}".format(repr(self.policy), repr(self.optimizer_actor), repr(self.optimizer_critic))


#####################################################
#                   FINAL REWARDS                   #
#####################################################

def nodes_to_atom_labels(nodes, env, nb_nodes):
    atom_types = env.possible_atom_types
    atom_idx = np.argmax(nodes[:nb_nodes], axis=1)
    node_labels = np.asarray(atom_types)[atom_idx]
    return node_labels

def dense_to_sparse_adj(adj, keep_self_edges):
    # Remove self-edges converting to surrogate input
    if not keep_self_edges:
        adj = adj - np.diag(np.diag(adj))
    sp = np.nonzero(adj)
    sp = np.stack(sp)
    return sp

def state_to_surrogate_graph(state, env, keep_self_edges=True):
    nodes = state['node'].squeeze()
    nb_nodes = int(np.sum(nodes))
    adj = state['adj'][:,:nb_nodes, :nb_nodes]

    atoms = nodes_to_atom_labels(nodes, env, nb_nodes)
    bonds = []
    for a,b in zip(adj, env.possible_bond_types):
        sp = dense_to_sparse_adj(a, keep_self_edges)
        bonds.append((sp, b))
    g = state_to_pyg(atoms, bonds)
    g = Batch.from_data_list(g)
    return g
    

def get_final_reward(state, env, surrogate_model):
    g = state_to_surrogate_graph(state, env, keep_self_edges=False)
    g = g.to(device)
    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g, None)
    reward = pred_docking_score.item() * -1
    return reward



#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_ppo(args, surrogate_model, env, writer=None):
    print("INFO: Not training with entropy term in loss")
    print("{} episodes before surrogate model as final reward".format(
                args.surrogate_reward_timestep_delay))

    ############## Hyperparameters ##############
    render = True
    solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 80           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 2000      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    #############################################

    ob = env.reset()
    nb_edge_types = ob['adj'].shape[0]
    ob = state_to_surrogate_graph(ob, env)
    input_dim = ob.x.shape[1]
    
    ppo = PPO_GCPN(lr,
                   betas,
                   gamma,
                   args.eta,
                   args.upsilon,
                   K_epochs,
                   eps_clip,
                   input_dim,
                   args.emb_size,
                   nb_edge_types,
                   args.layer_num_g,
                   args.num_hidden_g,
                   args.num_hidden_g,
                   args.mlp_num_layer,
                   args.mlp_num_hidden,
                   args.prob_redux_factor)
    
    print(ppo)
    memory = Memory()
    print("lr:", lr, "beta:", betas)

    # surrogate_model = surrogate_model.to(device)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    episode_count = 0

    # variables for plotting rewards

    rewbuffer_env = deque(maxlen=100)
    # training loop
    for i_episode in range(1, max_episodes+1):
        try:
            cur_ep_ret_env = 0
            state = env.reset()
            surr_reward=0.0
            for t in range(max_timesteps):
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory, env)
                state, reward, done, info = env.step(action)
                reward = 0

                if done:
                    smile = info['smile']
                    try:
                        reward = get_dock_score(smile)[0]
                    except Exception as e:
                        print(e)
                        reward = 0.0
                    with open('scores.csv','a') as fd:
                        csv_writer = csv.writer(fd)
                        csv_writer.writerow([smile, reward])

                
                
                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if time_step % update_timestep == 0:
                    print("updating ppo")
                    ppo.update(memory, i_episode, writer)
                    memory.clear_memory()
                    time_step = 0
                running_reward += reward
                cur_ep_ret_env += reward
                if (((i_episode+1)%20)==0) and render:
                    env.render()
                if done:
                    break
            writer.add_scalar("EpSurrogate", -1*surr_reward, episode_count)
            rewbuffer_env.append(cur_ep_ret_env)
            avg_length += t

            # write to Tensorboard
            writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), episode_count)
            # writer.add_scalar("Average Length", avg_length, global_step=episode_count)
            # writer.add_scalar("Running Reward", running_reward, global_step=episode_count)
            episode_count += 1

            # stop training if avg_reward > solved_reward
            if running_reward > (log_interval*solved_reward):
                print("########## Solved! ##########")
                torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format('test'))
                break
            
            # save every 500 episodes
            if i_episode % 500 == 0:
                torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format('test'))

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = running_reward/log_interval
                
                print('Episode {} \t Avg length: {} \t Avg reward: {:5.3f}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0
        except Exception as e:
            print(e)
            continue

