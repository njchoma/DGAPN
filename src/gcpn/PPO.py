import os
import re
import sys
import gym
import copy
import numpy as np
from collections import deque, OrderedDict

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from .gcpn_policy import ActorCriticGCPN
from .rnd_explore import RNDistillation

from utils.general_utils import get_current_datetime
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

    def extend(self, memory):
        self.actions.extend(memory.actions)
        self.states.extend(memory.states)
        self.logprobs.extend(memory.logprobs)
        self.rewards.extend(memory.rewards)
        self.is_terminals.extend(memory.is_terminals)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


def get_surrogate_dims(surrogate_model):
    state_dict = surrogate_model.state_dict()
    layers_name = [s for s in state_dict.keys() if re.compile('^layers\.[0-9]+\.weight$').match(s)]
    input_dim = state_dict[layers_name[0]].size(0)
    emb_dim = state_dict[layers_name[0]].size(-1)
    nb_edge_types = 1
    nb_layer = len(layers_name)
    nb_hidden = state_dict[layers_name[-1]].size(-1)
    return input_dim, emb_dim, nb_edge_types, nb_layer, nb_hidden

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
            input_dim, emb_dim, nb_edge_types, gnn_nb_layers, gnn_nb_hidden = get_surrogate_dims(emb_model)

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

    def update(self, memory, save_dir, eps=1e-5):
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
            if (i%10)==0:
                print("  {:3d}: Actor Loss: {:7.3f}, Critic Loss: {:7.3f}".format(i, loss, baseline_loss))
        # update RND
        rnd_loss = self.explore_critic.update(old_next_states)
        print("  RND Loss: {:7.3f}".format(rnd_loss))

        # save running model
        torch.save(self.policy.state_dict(), os.path.join(save_dir, 'running_gcpn.pth'))
        torch.save(self.explore_critic.state_dict(), os.path.join(save_dir, 'running_rnd.pth'))
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n".format(repr(self.policy))


#####################################################
#                      REWARDS                      #
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

################## Process ##################
tasks = mp.JoinableQueue()
results = mp.Queue()

# logging variables
running_reward = mp.Value("f", 0)
avg_length = mp.Value("i", 0)

class Worker(mp.Process):
    def __init__(self, env, task_queue, result_queue, max_timesteps):
        super(Worker, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.env = env

        self.max_timesteps = max_timesteps
        self.timestep_counter = 0

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task == None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            index, mol, done = next_task
            if index is None:
                self.result_queue.put((None, None, None, True))
                self.task_queue.task_done()
                continue
            #print('%s: Working' % proc_name)
            if done:
                self.timestep_counter = 0
                mol, candidates, done = self.env.reset()
            else:
                self.timestep_counter += 1
                mol, candidates, done = self.env.reset(mol)
                if self.timestep_counter >= self.max_timesteps:
                    done = True

            self.result_queue.put((index, mol, candidates, done))
            self.task_queue.task_done()
        return
'''
class Task(object):
    def __init__(self, index, action, done):
        self.index = index
        self.action = action
        self.done = done
    def __call__(self):
        return (self.action, self.done)
    def __str__(self):
        return '%d' % self.index

class Result(object):
    def __init__(self, index, state, candidates, done):
        self.index = index
        self.state = state
        self.candidates = candidates
        self.done = done
    def __call__(self):
        return (self.state, self.candidates, self.done)
    def __str__(self):
        return '%d' % self.index
'''
#############################################

def train_ppo(args, surrogate_model, env):

    ############## Hyperparameters ##############
    render = True
    solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    save_interval = 100         # save model in the interval

    max_episodes = 50000        # max training episodes
    max_timesteps = 6           # max timesteps in one episode
    update_timesteps = 500      # update policy every n timesteps

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    eta = 0.01                  # relative weight for entropy loss

    lr = (5e-4, 1e-4, 2e-6)        # learning rate for actor, critic and random network
    betas = (0.9, 0.999)
    eps = 0.01

    #############################################
    print("lr:", lr, "beta:", betas, "eps:", eps) # parameters for Adam optimizer

    print('Creating %d processes' % args.nb_procs)
    workers = [Worker(env, tasks, results, max_timesteps) for i in range(args.nb_procs)]
    for w in workers:
        w.start()

    # logging variables
    dt = get_current_datetime()
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)

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

    sample_count = 0
    episode_count = 0
    update_count = 0 # for adversarial
    save_counter = 0
    log_counter = 0

    avg_length = 0
    running_reward = 0

    memory = Memory()
    memories = [Memory() for _ in range(args.nb_procs)]
    rewbuffer_env = deque(maxlen=100)
    # training loop
    i_episode = 0
    while i_episode < max_episodes:
        print("collecting rollouts")
        for i in range(args.nb_procs):
            tasks.put((i, None, True))
        tasks.join()
        # unpack results
        mols = [None]*args.nb_procs
        done_idx = []
        notdone_idx, candidates, batch_idx = [], [], []
        for i in range(args.nb_procs):
            index, mol, cands, done = results.get()

            mols[index] = mol

            notdone_idx.append(index)
            candidates.extend(cands)
            batch_idx.extend([index]*len(cands))
        batch_idx = torch.LongTensor(batch_idx)
        while True:
            # action selections (for not done)
            if len(notdone_idx) > 0:
                state_embs, action_logprobs, actions, shifted_actions = ppo.select_action(
                    [mol_to_pyg_graph(mols[idx])[0] for idx in notdone_idx], 
                    [mol_to_pyg_graph(cand)[0] for cand in candidates], 
                    batch_idx, return_shifted=True)
            else:
                if sample_count >= update_timesteps:
                    break

            for i, idx in enumerate(notdone_idx):
                tasks.put((idx, candidates[shifted_actions[i]], False))

                memories[idx].states.append([state_embs[0][[i], :], state_embs[1][[i], :], state_embs[2][batch_idx == idx, :]])
                memories[idx].actions.append(actions[i])
                memories[idx].logprobs.append(action_logprobs[i])
            for idx in done_idx:
                if sample_count >= update_timesteps:
                    tasks.put((None, None, True))
                else:
                    tasks.put((idx, None, True))
            tasks.join()
            # unpack results
            mols = [None]*args.nb_procs
            new_done_idx = []
            new_notdone_idx, candidates, batch_idx = [], [], []
            for i in range(args.nb_procs):
                index, mol, cands, done = results.get()

                if index is not None:
                    mols[index] = mol
                if done:
                    new_done_idx.append(index)
                else:
                    new_notdone_idx.append(index)
                    candidates.extend(cands)
                    batch_idx.extend([index]*len(cands))
            batch_idx = torch.LongTensor(batch_idx)
            # get final rewards (for previously not done but now done)
            nowdone_idx = [idx for idx in notdone_idx if idx in new_done_idx]
            stillnotdone_idx = [idx for idx in notdone_idx if idx in new_notdone_idx]
            if len(nowdone_idx) > 0:
                surr_rewards = get_surr_reward(
                    [mols[idx] for idx in nowdone_idx],
                    surrogate_model, device)

            for i, idx in enumerate(nowdone_idx):
                i_episode += 1
                episode_count += 1
                avg_length += 1
                running_reward += surr_rewards[i]
                writer.add_scalar("EpSurrogate", -1*surr_rewards[i], i_episode-1)
                rewbuffer_env.append(surr_rewards[i])
                writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

                memories[idx].rewards.append(surr_rewards[i])
                memories[idx].is_terminals.append(True)
            for idx in stillnotdone_idx:
                avg_length += 1
                running_reward += 0

                memories[idx].rewards.append(0)
                memories[idx].is_terminals.append(False)
            # get exploration rewards
            if args.iota > 0 and i_episode > args.innovation_reward_episode_delay:
                if len(notdone_idx) > 0:
                    expl_rewards = get_expl_reward(
                        [mols[idx] for idx in notdone_idx],
                        surrogate_model, ppo.explore_critic, device)

                for i, idx in enumerate(notdone_idx):
                    running_reward += args.iota * expl_rewards[i]

                    memories[idx].rewards[-1] += args.iota * expl_rewards[i]


            sample_count += len(notdone_idx)
            done_idx = new_done_idx
            notdone_idx = new_notdone_idx

        for m in memories:
            memory.extend(m)
            m.clear()
        # update model
        print("\n\nupdating ppo @ episode %d..." % i_episode)
        ppo.update(memory, save_dir)
        memory.clear()

        update_count += 1
        save_counter += episode_count
        log_counter += episode_count

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), os.path.join(save_dir, 'PPO_continuous_solved_{}.pth'.format('test')))
            break

        # save every 500 episodes
        if save_counter > save_interval:
            torch.save(ppo.policy.state_dict(), os.path.join(save_dir, '{:05d}_gcpn.pth'.format(i_episode)))
            save_counter -= save_interval

        if log_counter > log_interval:
            avg_length = int(avg_length/log_counter)
            running_reward = running_reward/log_counter
            
            print('Episode {} \t Avg length: {} \t Avg reward: {:5.3f}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            log_counter -= log_interval

        episode_count = 0
        sample_count = 0

    # Add a poison pill for each process
    for i in range(args.nb_procs):
        tasks.put(None)
    tasks.join()

