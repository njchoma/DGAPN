import os
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

from .gcpn_policy import GCPN_CReM

from utils.general_utils import get_current_datetime
from utils.graph_utils import state_to_pyg

from gnn_surrogate.model import GNN_MyGAT

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

class Log:
    def __init__(self):
        self.ep_surrogates = []
        self.ep_rewards = []

    def extend(self, log):
        self.ep_surrogates.extend(log.ep_surrogates)
        self.ep_rewards.extend(log.ep_rewards)

    def clear(self):
        del self.ep_surrogates[:]
        del self.ep_rewards[:]


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
    
    def act(self, state, candidates, memory, surrogate_model):
        state = Batch.from_data_list([state])
        with torch.autograd.no_grad():
            state, new_states, action, prob = self.actor(state, candidates, surrogate_model)
        action_logprob = torch.log(prob)
        
        state = [state.cpu(), Data(x=new_states.cpu())]
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action
    
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
    
    def select_action(self, state, candidates, memory, surrogate_model):
        device = next(self.policy_old.parameters()).device

        g = state.to(device)
        g_candidates = candidates.to(device)
        action = self.policy_old.act(g, g_candidates, memory, surrogate_model)
        return action

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
        old_candidates = Batch().from_data_list([m[1] for m in memory.states]).to(device)
        old_actions = torch.tensor(memory.actions).to(device)
        old_logprobs = torch.stack(memory.logprobs).to(device)
        
        # Optimize policy for K epochs:
        print("Optimizing...")

        for i in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, entropies = self.policy.evaluate(old_states,
                                                                     old_candidates,
                                                                     old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

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

def get_reward(state, surrogate_model):
    g = Batch().from_data_list([state])
    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g, None)
    reward = pred_docking_score.item() * -1
    return reward


#####################################################
#                   TRAINING LOOP                   #
#####################################################

################## Process ##################
lock = mp.Lock()
tasks = mp.JoinableQueue()
results = mp.Queue()

episode_count = mp.Value("i", 0)
sample_count = mp.Value("i", 0)

# logging variables
running_reward = mp.Value("f", 0)
avg_length = mp.Value("i", 0)

class Sampler(mp.Process):
    def __init__(self, args, env, task_queue, result_queue, max_episodes, max_timesteps, update_timestep):
        super(Sampler, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.update_timestep = update_timestep

        self.env = env
        self.policy = ActorCriticGCPN(args.input_size,
                             args.emb_size,
                             args.nb_edge_types,
                             args.layer_num_g,
                             args.num_hidden_g,
                             args.mlp_num_layer,
                             args.mlp_num_hidden)
        self.surrogate = GNN_MyGAT(args.input_size,
                          args.emb_size,
                          args.num_hidden_g,
                          args.layer_num_g)
        self.memory = Memory()
        self.log = Log()

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            p_state, s_state = next_task()
            self.policy.load_state_dict(p_state)
            self.surrogate.load_state_dict(s_state)
            self.memory.clear()
            self.log.clear()

            print('%s: Sampling' % proc_name)
            state, candidates, done = self.env.reset()

            while sample_count.value < self.update_timestep and episode_count.value < self.max_episodes:
                n_step = 0
                for t in range(self.max_timesteps):
                    n_step += 1
                    # Running policy_old:
                    action = self.policy.act(state, candidates, self.memory, self.surrogate)
                    state, candidates, done = self.env.step(action)

                    reward = 0
                    if (t==(self.max_timesteps-1)) or done:
                        surr_reward = get_reward(state, self.surrogate)
                        reward = surr_reward

                    # Saving reward and is_terminals:
                    self.memory.rewards.append(reward)
                    self.memory.is_terminals.append(done)

                    if done:
                        break

                lock.acquire() # C[]
                sample_count.value += n_step
                episode_count.value += 1

                running_reward.value += sum(self.memory.rewards)
                avg_length.value += t

                lock.release() # L[]
                self.log.ep_surrogates.append(-1*surr_reward)
                self.log.ep_rewards.append(reward)

            self.result_queue.put(Result(self.memory, self.log))
            self.task_queue.task_done()
        return

class Task(object):
    def __init__(self, p_state, s_state):
        self.p_state = p_state
        self.s_state = s_state
    def __call__(self):
        return (self.p_state, self.s_state)
    #def __str__(self):
    #    return '%s * %s' % (self.a, self.b)

class Result(object):
    def __init__(self, memory, log):
        self.memory = memory
        self.log = log
    def __call__(self):
        return (self.memory, self.log)
    #def __str__(self):
    #    return '%s * %s' % (self.a, self.b)

#############################################

def train_ppo(args, surrogate_model, env):
    ############## Hyperparameters ##############
    render = True
    solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    save_interval = 100         # save model in the interval

    max_episodes = 50000        # max training episodes
    max_timesteps = 6           # max timesteps in one episode
    update_timestep = 500       # update policy every n timesteps

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)

    #############################################
    print("lr:", lr, "beta:", betas)

    print('Creating %d processes' % args.nb_procs)
    samplers = [Sampler(args, env, tasks, results, max_episodes, max_timesteps, update_timestep)
                for i in range(args.nb_procs)]
    for w in samplers:
        w.start()

    # logging variables
    dt = get_current_datetime()
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
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
    print(ppo)

    surrogate_model.eval()
    print(surrogate_model)

    update_count = 0 # for adversarial
    save_counter = 0
    log_counter = 0

    memory = Memory()
    log = Log()
    rewbuffer_env = deque(maxlen=100)
    # training loop
    i_episode = 0
    while i_episode < max_episodes:
        print("collecting rollouts")
        ppo.to_device(torch.device("cpu"))
        # Enqueue jobs
        for i in range(args.nb_procs):
            p_state = ppo.policy_old.state_dict()
            s_state = surrogate_model.state_dict()
            tasks.put(Task(p_state, s_state))
        # Wait for all of the tasks to finish
        tasks.join()
        print("rollouts collected")
        # Start unpacking results
        for i in range(args.nb_procs):
            result = results.get()
            memory.extend(result.memory)
            log.extend(result.log)

        i_episode += episode_count.value

        # write to Tensorboard
        for i in reversed(range(episode_count.value)):
            rewbuffer_env.append(log.ep_rewards[i])
            writer.add_scalar("EpSurrogate", log.ep_surrogates[i], i_episode - i)
            writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode - i)
        log.clear()
        # update model
        print("\n\nupdating ppo @ episode %d..." % i_episode)
        ppo.to_device(device)
        ppo.update(memory)
        memory.clear()

        update_count += 1
        save_counter += episode_count.value
        log_counter += episode_count.value

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > solved_reward:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), os.path.join(save_dir, 'PPO_continuous_solved_{}.pth'.format('test')))
            break

        # save running model
        torch.save(ppo.policy.actor, os.path.join(save_dir, 'running_gcpn.pth'))

        # save every 500 episodes
        if save_counter > save_interval:
            torch.save(ppo.policy.actor, os.path.join(save_dir, '{:05d}_gcpn.pth'.format(i_episode)))
            save_counter -= save_interval

        # logging
        if log_counter > log_interval:
            avg_length.value = int(avg_length.value/log_counter)
            running_reward.value = running_reward.value/log_counter
            
            print('Episode {} \t Avg length: {} \t Avg reward: {:5.3f}'.format(i_episode, avg_length.value, running_reward.value))
            running_reward.value = 0
            avg_length.value = 0
            log_counter = 0

        episode_count.value = 0
        sample_count.value = 0

    # Add a poison pill for each process
    for i in range(args.nb_procs):
        tasks.put(None)
    tasks.join()

