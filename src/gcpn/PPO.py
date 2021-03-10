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

from multiprocessing import Pool, Lock, Barrier, Value, Queue


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
        self.actor = GCPN_CReM(input_dim,
                               emb_dim,
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
                 gnn_nb_hidden_kernel,
                 mlp_nb_layers,
                 mlp_nb_hidden,
                 device):
        super(PPO_GCPN, self).__init__()
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eta = eta
        self.upsilon = upsilon
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
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

    
    def select_action(self, state, candidates, memory, surrogate_model):
        g = state.to(self.device)
        g_candidates = candidates.to(self.device)
        action = self.policy_old.act(g, g_candidates, memory, surrogate_model)
        return action

    def update(self, memory, i_episode, writer=None):
        print("\n\nupdating...")
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        
        # convert list to tensor
        old_states = torch.cat(([m[0] for m in memory.states]),dim=0).to(self.device)
        old_candidates = Batch().from_data_list([m[1] for m in memory.states]).to(self.device)
        old_actions = torch.tensor(memory.actions).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device)
        
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
    print("{} episodes before surrogate model as final reward".format(
                args.surrogate_reward_timestep_delay))
    # logging variables
    dt = get_current_datetime()
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + dt)
    os.makedirs(save_dir, exist_ok=True)

    ############## Hyperparameters ##############
    render = True
    solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    save_interval = 100         # save model in the interval

    max_episodes = 50000        # max training episodes
    max_timesteps = 6           # max timesteps in one episode
    update_timestep = 120       # update policy every n timesteps

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.0001                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    #############################################

    ob, _, _ = env.reset()
    input_dim = ob.x.shape[1]

    # emb_dim = surrogate_model.emb_dim
    emb_dim = 512 # temp fix to use an old surrogate model
    nb_edge_types = 1

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    ppo = PPO_GCPN(lr,
                   betas,
                   gamma,
                   args.eta,
                   args.upsilon,
                   K_epochs,
                   eps_clip,
                   input_dim,
                   emb_dim,
                   nb_edge_types,
                   args.layer_num_g,
                   args.num_hidden_g,
                   args.num_hidden_g,
                   args.mlp_num_layer,
                   args.mlp_num_hidden,
                   device)
    
    print(ppo)
    print("lr:", lr, "beta:", betas)

    surrogate_model = surrogate_model.to(device)
    surrogate_model.eval()

    ################## Process ##################
    pool = Pool(4)
    update_lock = Lock()

    episode_count = Value("i", 0)
    sample_count = Value("i", 0)

    # logging variables
    running_reward = Value("f", 0)
    avg_length = Value("i", 0)
    rewbuffer_env = Queue(100)

    def collect_trajectories(ppo, env, surrogate_model, max_episodes, max_timesteps, update_timestep, device):
        memory = Memory()
        ep_surrogates = []
        ep_rew_env_mean = []

        state, candidates, done = env.reset()
        starting_reward = get_reward(state, surrogate_model, device)

        while episode_count.value < max_episodes and sample_count.value < update_timestep:
            n_step = 0
            for t in range(max_timesteps):
                n_step += 1
                # Running policy_old:
                action = ppo.select_action(state, candidates, memory, surrogate_model)
                state, candidates, done = env.step(action)

                reward = 0
                if (t==(max_timesteps-1)) or done:
                    surr_reward = get_reward(state, surrogate_model, device)
                    reward = surr_reward-starting_reward

                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                if done:
                    break

            update_lock.acquire()

            sample_count.value += n_step
            episode_count.value += 1

            running_reward.value += sum(memory.rewards)
            avg_length.value += t
            rewbuffer_env.put(reward)

            update_lock.release()

            ep_surrogates.append(-1*surr_reward)
            ep_rew_env_mean.append(np.mean(rewbuffer_env))

        return memory, ep_surrogates, ep_rew_env_mean

    #############################################

    memory = Memory()
    ep_surrogates = []
    ep_rew_env_mean = []
    
    update_count = 0 # for adversarial
    save_counter = 0
    log_counter = 0

    # training loop
    i_episode = 0
    while i_episode < max_episodes:
        # parallel runs
        results = []
        for i in range(pool._processes):
            results.append(pool.apply_async(collect_trajectories, 
                [ppo, env, surrogate_model, max_episodes - i_episode, max_timesteps, update_timestep, device]))
        # process results
        for result in results:
            result = result.get()
            memory.rewards.extend(result[0].rewards)
            memory.is_terminals.extend(result[0].is_terminals)
            ep_surrogates.extend(result[1])
            ep_rew_env_mean.extend(result[2])

        # write to Tensorboard
        for i in reversed(range(episode_count.value)):
            writer.add_scalar("EpSurrogate", ep_surrogates[i], i_episode - i)
            writer.add_scalar("EpRewEnvMean", ep_rew_env_mean[i], i_episode - i)
        ep_surrogates = []
        ep_rew_env_mean = []
        # update model
        print("updating ppo")
        ppo.update(memory, i_episode, writer)
        memory.clear_memory()

        update_count += 1
        save_counter += episode_count.value
        log_counter += episode_count.value

        i_episode += episode_count.value

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

