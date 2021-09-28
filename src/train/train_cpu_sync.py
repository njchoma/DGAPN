import os
import gym
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict
from copy import deepcopy

import time
from datetime import datetime

import torch
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from dgapn.DGAPN import DGAPN, save_DGAPN

from reward.get_main_reward import get_main_reward

from utils.general_utils import initialize_logger, close_logger, deque_to_csv
from utils.graph_utils import mols_to_pyg_batch

#####################################################
#                   HELPER MODULES                  #
#####################################################

class Memory:
    def __init__(self):
        self.states = []        # state representations: pyg graph
        self.candidates = []    # next state (candidate) representations: pyg graph
        self.states_next = []   # next state (chosen) representations: pyg graph
        self.actions = []       # action index: long
        self.logprobs = []      # action log probabilities: float
        self.rewards = []       # rewards: float
        self.terminals = []     # trajectory status: logical

    def extend(self, memory):
        self.states.extend(memory.states)
        self.candidates.extend(memory.candidates)
        self.states_next.extend(memory.states_next)
        self.actions.extend(memory.actions)
        self.logprobs.extend(memory.logprobs)
        self.rewards.extend(memory.rewards)
        self.terminals.extend(memory.terminals)

    def clear(self):
        del self.states[:]
        del self.candidates[:]
        del self.states_next[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminals[:]


class Log:
    def __init__(self):
        self.ep_mols = []
        self.ep_lengths = []
        self.ep_rewards = []
        self.ep_main_rewards = []

    def extend(self, log):
        self.ep_mols.extend(log.ep_mols)
        self.ep_lengths.extend(log.ep_lengths)
        self.ep_rewards.extend(log.ep_rewards)
        self.ep_main_rewards.extend(log.ep_main_rewards)

    def clear(self):
        del self.ep_mols[:]
        del self.ep_lengths[:]
        del self.ep_rewards[:]
        del self.ep_main_rewards[:]

#####################################################
#                     SUBPROCESS                    #
#####################################################

tasks = mp.JoinableQueue()
results = mp.Queue()


class Sampler(mp.Process):
    def __init__(self, args, env, task_queue, result_queue,
                    nb_procs, max_timesteps, update_timesteps):
        super(Sampler, self).__init__()
        self.args = args
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.nb_procs = nb_procs
        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps
        self.sample_count = 0
        self.episode_count = 0

        #self.env = deepcopy(env)
        self.env = env
        self.model = DGAPN(args.lr,
                    args.betas,
                    args.eps,
                    args.eta,
                    args.gamma,
                    args.eps_clip,
                    args.k_epochs,
                    args.embed_state,
                    args.emb_nb_inherit,
                    args.input_size,
                    args.nb_edge_types,
                    args.use_3d,
                    args.gnn_nb_layers,
                    args.gnn_nb_shared,
                    args.gnn_nb_hidden,
                    args.enc_num_layers,
                    args.enc_num_hidden,
                    args.enc_num_output,
                    args.rnd_num_layers,
                    args.rnd_num_hidden,
                    args.rnd_num_output)

        self.memory = Memory()
        self.log = Log()

    def run(self):
        proc_name = self.name
        self.args.run_id = self.args.run_id + proc_name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            i_episode, model_state = next_task()
            self.model.load_state_dict(model_state)
            self.memory.clear()
            self.log.clear()
            self.sample_count = 0
            self.episode_count = 0

            print('%s: Sampling' % proc_name)

            while self.sample_count*self.nb_procs < self.update_timesteps:
                state, candidates, done = self.env.reset()

                for t in range(self.max_timesteps):
                    # Running policy:
                    state_emb, candidates_emb, action_logprob, action = self.model.select_action(
                        mols_to_pyg_batch(state, self.model.emb_3d, device=self.model.device),
                        mols_to_pyg_batch(candidates, self.model.emb_3d, device=self.model.device))
                    self.memory.states.append(state_emb[0])
                    self.memory.candidates.append(candidates_emb)
                    self.memory.states_next.append(candidates_emb[action])
                    self.memory.actions.append(action)
                    self.memory.logprobs.append(action_logprob)

                    state, candidates, done = self.env.step(action)

                    reward = 0
                    if (t==(self.max_timesteps-1)) or done:
                        main_reward = get_main_reward(state, reward_type=self.args.reward_type, args=self.args)[0]
                        reward = main_reward
                        done = True
                    if (self.args.iota > 0 and 
                        i_episode + self.episode_count*self.nb_procs > self.args.innovation_reward_episode_delay and 
                        i_episode + self.episode_count*self.nb_procs < self.args.innovation_reward_episode_cutoff):
                        inno_reward = self.model.get_inno_reward(mols_to_pyg_batch(state, self.model.emb_3d, device=self.model.device))
                        reward += inno_reward

                    # Saving rewards and terminals:
                    self.memory.rewards.append(reward)
                    self.memory.terminals.append(done)

                    if done:
                        break

                self.sample_count += (t+1)
                self.episode_count += 1

                self.log.ep_lengths.append(t+1)
                self.log.ep_rewards.append(sum(self.memory.rewards))
                self.log.ep_main_rewards.append(main_reward)
                self.log.ep_mols.append(Chem.MolToSmiles(state))

            self.result_queue.put(Result(self.episode_count, self.memory, self.log))
            self.task_queue.task_done()
        return

class Task(object):
    def __init__(self, i_episode, model_state):
        self.i_episode = i_episode
        self.model_state = model_state
    def __call__(self):
        return (self.i_episode, self.model_state)
    def __str__(self):
        return '%d' % self.i_episode

class Result(object):
    def __init__(self, episode_count, memory, log):
        self.episode_count = episode_count
        self.memory = memory
        self.log = log
    def __call__(self):
        return (self.episode_count, self.memory, self.log)
    def __str__(self):
        return '%d' % self.episode_count

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_cpu_sync(args, env, model):
    # initiate subprocesses
    print('Creating %d processes' % args.nb_procs)
    workers = [Sampler(args, env, tasks, results,
                args.nb_procs, args.max_timesteps, args.update_timesteps) for i in range(args.nb_procs)]
    for w in workers:
        w.start()

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)
    logging.info(model)

    episode_count = 0
    save_counter = 0
    log_counter = 0

    running_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    log = Log()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=1000)
    # training loop
    i_episode = 0
    while i_episode < args.max_episodes:
        logging.info("\n\ncollecting rollouts")
        model.to_device(torch.device("cpu"))
        # Enqueue jobs
        for i in range(args.nb_procs):
            tasks.put(Task(i_episode, model.state_dict()))
        # Wait for all of the tasks to finish
        tasks.join()
        # Start unpacking results
        for i in range(args.nb_procs):
            result = results.get()
            e, m, l = result()
            episode_count += e
            memory.extend(m)
            log.extend(l)

        i_episode += episode_count
        model.to_device(args.device)

        # log results
        for i in reversed(range(episode_count)):
            running_length += log.ep_lengths[i]
            running_reward += log.ep_rewards[i]
            running_main_reward += log.ep_main_rewards[i]
            rewbuffer_env.append(log.ep_main_rewards[i])
            molbuffer_env.append(log.ep_mols[i])
            writer.add_scalar("EpMainRew", log.ep_main_rewards[i], i_episode - 1)
            writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode - 1)
        log.clear()

        # update model
        logging.info("\nupdating model @ episode %d..." % i_episode)
        model.update(memory)
        memory.clear()

        save_counter += episode_count
        log_counter += episode_count

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            save_DGAPN(model, os.path.join(save_dir, 'DGAPN_continuous_solved_{}.pt'.format('test')))
            break

        # save every 500 episodes
        if save_counter >= args.save_interval:
            save_DGAPN(model, os.path.join(save_dir, '{:05d}_dgapn.pt'.format(i_episode)))
            deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'))
            save_counter = 0

        # save running model
        save_DGAPN(model, os.path.join(save_dir, 'running_dgapn.pt'))

        if log_counter >= args.log_interval:
            logging.info('Episode {} \t Avg length: {} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, running_length/log_counter, running_reward/log_counter, running_main_reward/log_counter))

            running_reward = 0
            running_main_reward = 0
            running_length = 0
            log_counter = 0

        episode_count = 0

    close_logger()
    writer.close()
    # Add a poison pill for each process
    for i in range(args.nb_procs):
        tasks.put(None)
    tasks.join()
