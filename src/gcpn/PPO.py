import os
import re
import sys
import gym
import copy
import yaml
import numpy as np
from collections import deque, OrderedDict

import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from .gcpn_policy import ActorCriticGCPN
from .rnd_explore import RNDistillation

from utils.general_utils import initialize_logger
from utils.graph_utils import mol_to_pyg_graph, get_batch_shift

from gnn_surrogate.model import GNN_MyGAT

#####################################################
#                   HELPER MODULES                  #
#####################################################

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

#################################################
#                    UPDATE                     #
#################################################

class DGCPN(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 eta,
                 gamma,
                 K_epochs,
                 eps_clip,
                 emb_model=None,
                 input_dim=None,
                 emb_dim=None,
                 output_dim=None,
                 nb_edge_types=None,
                 gnn_nb_layers=None,
                 gnn_nb_hidden=None,
                 acp_nb_layers=None,
                 acp_nb_hidden=None,
                 rnd_nb_layers=None,
                 rnd_nb_hidden=None):
        super(DGCPN, self).__init__()
        self.gamma = gamma
        self.K_epochs = K_epochs

        if emb_model is not None:
            input_dim = emb_model.input_dim
            emb_dim = emb_model.emb_dim
            nb_edge_types = emb_model.nb_edge_types
            gnn_nb_layers = emb_model.nb_layers
            gnn_nb_hidden = emb_model.nb_hidden

        self.policy = ActorCriticGCPN(lr[:2],
                                      betas,
                                      eps,
                                      eta,
                                      eps_clip,
                                      emb_model,
                                      input_dim,
                                      emb_dim,
                                      nb_edge_types,
                                      gnn_nb_layers,
                                      gnn_nb_hidden,
                                      acp_nb_layers,
                                      acp_nb_hidden)

        self.policy_old = ActorCriticGCPN(lr[:2],
                                          betas,
                                          eps,
                                          eta,
                                          eps_clip,
                                          emb_model,
                                          input_dim,
                                          emb_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          acp_nb_layers,
                                          acp_nb_hidden)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.explore_critic = RNDistillation(lr[2],
                                             betas,
                                             eps,
                                             input_dim,
                                             emb_dim,
                                             output_dim,
                                             nb_edge_types,
                                             gnn_nb_layers,
                                             gnn_nb_hidden,
                                             rnd_nb_layers,
                                             rnd_nb_hidden)

        self.device = torch.device("cpu")

    def to_device(self, device):
        self.policy.to(device)
        self.policy_old.to(device)
        self.explore_critic.to(device)
        self.device = device

    def forward(self):
        raise NotImplementedError

    def select_action(self, states, candidates, batch_idx=None, return_shifted=False):
        if not isinstance(states, list):
            states = [states]
        if batch_idx is None:
            batch_idx = torch.empty(len(candidates), dtype=torch.long).fill_(0)
        batch_idx = batch_idx.to(self.device)

        g = Batch.from_data_list(states).to(self.device)
        g_candidates = Batch.from_data_list(candidates).to(self.device)
        with torch.autograd.no_grad():
            g_emb, g_next_emb, X_states, action_logprobs, actions, shifted_actions = self.policy_old.select_action(
                g, g_candidates, batch_idx)

        if return_shifted:
            return [g_emb, g_next_emb, X_states], action_logprobs, actions, shifted_actions
        else:
            return [g_emb, g_next_emb, X_states], action_logprobs, actions

    def update(self, memory, eps=1e-5):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        # convert list to tensor
        old_states = torch.cat(([m[0] for m in memory.states]),dim=0).to(self.device)
        old_next_states = torch.cat(([m[1] for m in memory.states]),dim=0).to(self.device)
        old_candidates = Batch().from_data_list([Data(x=m[2]) for m in memory.states]).to(self.device)
        old_actions = torch.tensor(memory.actions).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs).to(self.device)

        old_values = self.policy_old.get_value(old_states)

        # Optimize policy for K epochs:
        print("Optimizing...")

        for i in range(self.K_epochs):
            loss, baseline_loss = self.policy.update(old_states, old_candidates, old_actions, old_logprobs, old_values, rewards)
            rnd_loss = self.explore_critic.update(old_next_states)
            if (i%10)==0:
                print("  {:3d}: Actor Loss: {:7.3f}, Critic Loss: {:7.3f}, RND Loss: {:7.3f}".format(i, loss, baseline_loss, rnd_loss))

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n".format(repr(self.policy))


#####################################################
#                   FINAL REWARDS                   #
#####################################################

def get_surr_reward(states, surrogate_model, device):
    if not isinstance(states, list):
        states = [states]

    states = [mol_to_pyg_graph(state)[0] for state in states]
    g = Batch().from_data_list(states).to(device)

    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g, None)
    return (-pred_docking_score).tolist()

def get_expl_reward(states, emb_model, explore_critic, device):
    if not isinstance(states, list):
        states = [states]
    
    states = [mol_to_pyg_graph(state)[0] for state in states]
    g = Batch().from_data_list(states).to(device)

    X = emb_model.get_embedding(g)
    scores = explore_critic.get_score(X)
    return scores.tolist()


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
    update_timesteps = 30       # update policy every n timesteps

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    eta = 0.01                  # relative weight for entropy loss

    lr = (5e-4, 1e-4, 2e-3)     # learning rate for actor, critic and random network
    betas = (0.9, 0.999)
    eps = 0.01

    #############################################
    print("lr:", lr, "beta:", betas, "eps:", eps) # parameters for Adam optimizer

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    surrogate_model.to(device)
    surrogate_model.eval()
    print(surrogate_model)

    ppo = DGCPN(lr,
                betas,
                eps,
                eta,
                gamma,
                K_epochs,
                eps_clip,
                surrogate_model,
                args.input_size,
                args.emb_size,
                args.output_size,
                args.nb_edge_types,
                args.gnn_nb_layers,
                args.gnn_nb_hidden,
                args.acp_num_layers,
                args.acp_num_hidden,
                args.rnd_num_layers,
                args.rnd_num_hidden)
    ppo.to_device(device)
    print(ppo)

    time_step = 0
    update_count = 0 # for adversarial

    avg_length = 0
    running_reward = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    # training loop
    for i_episode in range(1, max_episodes+1):
        state, candidates, done = env.reset()

        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            state, action_logprob, action = ppo.select_action(
                mol_to_pyg_graph(state)[0], [mol_to_pyg_graph(cand)[0] for cand in candidates])
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

            state, candidates, done = env.step(action)

            # done and reward may not be needed anymore
            reward = 0

            if (t==(max_timesteps-1)) or done:
                surr_reward = get_surr_reward(state, surrogate_model, device)
                reward = surr_reward

            if args.iota > 0 and i_episode > args.innovation_reward_episode_delay:
                expl_reward = get_expl_reward(state, surrogate_model, ppo.explore_critic, device)
                reward += expl_reward

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward
            if done:
                break

        # update if it's time
        if time_step > update_timesteps:
            print("\n\nupdating ppo @ episode %d..." % i_episode)
            time_step = 0
            ppo.update(memory, save_dir)
            memory.clear()

        writer.add_scalar("EpSurrogate", -1*surr_reward, i_episode-1)
        rewbuffer_env.append(reward)
        avg_length += t

        # write to Tensorboard
        writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.actor, './PPO_continuous_solved_{}.pth'.format('test'))
            break

        # save every 500 episodes
        if (i_episode-1) % save_interval == 0:
            torch.save(ppo.policy.actor, os.path.join(save_dir, '{:05d}_gcpn.pth'.format(i_episode)))

        # save running model
        torch.save(ppo.policy.actor, os.path.join(save_dir, 'running_gcpn.pth'))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = running_reward/log_interval
            
            print('Episode {} \t Avg length: {} \t Avg reward: {:5.3f}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

