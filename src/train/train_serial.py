import os
import gym
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict

import time
from datetime import datetime

import torch
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
#                   TRAINING LOOP                   #
#####################################################

def train_serial(args, env, model):
    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)
    logging.info(model)

    time_step = 0

    running_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=1000)
    # training loop
    for i_episode in range(1, args.max_episodes+1):
        if time_step == 0:
            logging.info("\n\ncollecting rollouts")
        state, candidates, done = env.reset()

        for t in range(args.max_timesteps):
            time_step += 1
            # Running policy:
            state_emb, candidates_emb, action_logprob, action = model.select_action(
                mols_to_pyg_batch(state, model.emb_3d, device=model.device),
                mols_to_pyg_batch(candidates, model.emb_3d, device=model.device))
            memory.states.append(state_emb[0])
            memory.candidates.append(candidates_emb)
            memory.states_next.append(candidates_emb[action])
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

            state, candidates, done = env.step(action)

            reward = 0
            if (t==(args.max_timesteps-1)) or done:
                main_reward = get_main_reward(state, reward_type=args.reward_type, args=args)[0]
                reward = main_reward
                running_main_reward += main_reward
                done = True
            if (args.iota > 0 and 
                i_episode > args.innovation_reward_episode_delay and 
                i_episode < args.innovation_reward_episode_cutoff):
                inno_reward = model.get_inno_reward(mols_to_pyg_batch(state, model.emb_3d, device=model.device))
                reward += inno_reward
            running_reward += reward

            # Saving rewards and terminals:
            memory.rewards.append(reward)
            memory.terminals.append(done)

            if done:
                break

        # update if it's time
        if time_step >= args.update_timesteps:
            logging.info("\nupdating model @ episode %d..." % i_episode)
            time_step = 0
            model.update(memory)
            memory.clear()

        writer.add_scalar("EpMainRew", main_reward, i_episode-1)
        rewbuffer_env.append(main_reward) # reward
        molbuffer_env.append((Chem.MolToSmiles(state), main_reward))
        running_length += (t+1)

        # write to Tensorboard
        writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            save_DGAPN(model, os.path.join(save_dir, 'DGAPN_continuous_solved_{}.pt'.format('test')))
            break

        # save every save_interval episodes
        if (i_episode-1) % args.save_interval == 0:
            save_DGAPN(model, os.path.join(save_dir, '{:05d}_dgapn.pt'.format(i_episode)))
            deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'))

        # save running model
        save_DGAPN(model, os.path.join(save_dir, 'running_dgapn.pt'))

        # logging
        if i_episode % args.log_interval == 0:
            logging.info('Episode {} \t Avg length: {} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, running_length/args.log_interval, running_reward/args.log_interval, running_main_reward/args.log_interval))

            running_length = 0
            running_reward = 0
            running_main_reward = 0

    close_logger()
    writer.close()
