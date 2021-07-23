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

#####################################################
#                     SUBPROCESS                    #
#####################################################

tasks = mp.JoinableQueue()
results = mp.Queue()


class Sampler(mp.Process):
    def __init__(self, env, task_queue, result_queue, max_timesteps):
        super(Sampler, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue

        self.max_timesteps = max_timesteps
        self.timestep_count = 0

        #self.env = deepcopy(env)
        self.env = env

    def run(self):
        # input:
        ## None:                                kill
        ## (None, None, True):                  dummy task
        ## (index, state, done):                trajectory id, molecule smiles, trajectory status
        #
        # output:
        ## (None, None, None, True):            dummy task
        ## (index, state, candidates, done):    trajectory id, molecule smiles, candidate smiles, trajectory status
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task == None:
                # Poison pill means shutdown
                print('\n%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            index, state, done = next_task
            if index is None:
                self.result_queue.put((None, None, None, True))
                self.task_queue.task_done()
                continue
            # print('%s: Working' % proc_name)
            if done:
                self.timestep_count = 0
                state, candidates, done = self.env.reset(return_type='smiles')
            else:
                self.timestep_count += 1
                state, candidates, done = self.env.reset(state, return_type='smiles')
                if self.timestep_count >= self.max_timesteps:
                    done = True

            self.result_queue.put((index, state, candidates, done))
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

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_gpu_sync(args, env, model):
    # initiate subprocesses
    print('Creating %d processes' % args.nb_procs)
    workers = [Sampler(env, tasks, results, args.max_timesteps) for i in range(args.nb_procs)]
    for w in workers:
        w.start()

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)
    logging.info(model)

    sample_count = 0
    episode_count = 0
    save_counter = 0
    log_counter = 0

    running_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    memories = [Memory() for _ in range(args.nb_procs)]
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=1000)
    # training loop
    i_episode = 0
    while i_episode < args.max_episodes:
        logging.info("\n\ncollecting rollouts")
        for i in range(args.nb_procs):
            tasks.put((i, None, True))
        tasks.join()
        # unpack results
        states = [None] * args.nb_procs
        done_idx = []
        notdone_idx, candidates, batch_idx = [], [], []
        for i in range(args.nb_procs):
            index, state, cands, done = results.get()

            states[index] = state

            notdone_idx.append(index)
            candidates.append(cands)
            batch_idx.extend([index] * len(cands))
        while True:
            # action selections (for not done)
            if len(notdone_idx) > 0:
                states_emb, candidates_emb, action_logprobs, actions = model.select_action(
                    mols_to_pyg_batch([Chem.MolFromSmiles(states[idx])
                        for idx in notdone_idx], model.emb_3d, device=model.device),
                    mols_to_pyg_batch([Chem.MolFromSmiles(item) 
                        for sublist in candidates for item in sublist], model.emb_3d, device=model.device),
                    batch_idx)
                if not isinstance(actions, list):
                    action_logprobs = [action_logprobs]
                    actions = [actions]
            else:
                if sample_count >= args.update_timesteps:
                    break

            for i, idx in enumerate(notdone_idx):
                tasks.put((idx, candidates[i][actions[i]], False))
                cands = [data for j, data in enumerate(candidates_emb) if batch_idx[j] == idx]

                memories[idx].states.append(states_emb[i])
                memories[idx].candidates.append(cands)
                memories[idx].states_next.append(cands[actions[i]])
                memories[idx].actions.append(actions[i])
                memories[idx].logprobs.append(action_logprobs[i])
            for idx in done_idx:
                if sample_count >= args.update_timesteps:
                    tasks.put((None, None, True))
                else:
                    tasks.put((idx, None, True))
            tasks.join()
            # unpack results
            states = [None] * args.nb_procs
            new_done_idx = []
            new_notdone_idx, candidates, batch_idx = [], [], []
            for i in range(args.nb_procs):
                index, state, cands, done = results.get()

                if index is not None:
                    states[index] = state
                if done:
                    new_done_idx.append(index)
                else:
                    new_notdone_idx.append(index)
                    candidates.append(cands)
                    batch_idx.extend([index] * len(cands))
            # get final rewards (for previously not done but now done)
            nowdone_idx = [idx for idx in notdone_idx if idx in new_done_idx]
            stillnotdone_idx = [idx for idx in notdone_idx if idx in new_notdone_idx]
            if len(nowdone_idx) > 0:
                main_rewards = get_main_reward(
                    [Chem.MolFromSmiles(states[idx]) for idx in nowdone_idx], reward_type=args.reward_type, args=args)
                if not isinstance(main_rewards, list):
                    main_rewards = [main_rewards]

            for i, idx in enumerate(nowdone_idx):
                main_reward = main_rewards[i]

                i_episode += 1
                running_reward += main_reward
                running_main_reward += main_reward
                writer.add_scalar("EpMainRew", main_reward, i_episode - 1)
                rewbuffer_env.append(main_reward)
                molbuffer_env.append((states[idx], main_reward))
                writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode - 1)

                memories[idx].rewards.append(main_reward)
                memories[idx].terminals.append(True)
            for idx in stillnotdone_idx:
                running_reward += 0

                memories[idx].rewards.append(0)
                memories[idx].terminals.append(False)
            # get innovation rewards
            if (args.iota > 0 and 
                i_episode > args.innovation_reward_episode_delay and 
                i_episode < args.innovation_reward_episode_cutoff):
                if len(notdone_idx) > 0:
                    inno_rewards = model.get_inno_reward(
                        mols_to_pyg_batch([Chem.MolFromSmiles(states[idx]) 
                            for idx in notdone_idx], model.emb_3d, device=model.device))
                    if not isinstance(inno_rewards, list):
                        inno_rewards = [inno_rewards]

                for i, idx in enumerate(notdone_idx):
                    inno_reward = args.iota * inno_rewards[i]

                    running_reward += inno_reward

                    memories[idx].rewards[-1] += inno_reward

            sample_count += len(notdone_idx)
            episode_count += len(nowdone_idx)
            running_length += len(notdone_idx)

            done_idx = new_done_idx
            notdone_idx = new_notdone_idx

        for m in memories:
            memory.extend(m)
            m.clear()

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

            running_length = 0
            running_reward = 0
            running_main_reward = 0
            log_counter = 0

        episode_count = 0
        sample_count = 0

    close_logger()
    writer.close()
    # Add a poison pill for each process
    for i in range(args.nb_procs):
        tasks.put(None)
    tasks.join()
