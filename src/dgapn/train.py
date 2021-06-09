import os
import gym
import logging
import numpy as np
from rdkit import Chem
from collections import deque, OrderedDict

import time
from datetime import datetime

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from .DGAPN import DGAPN, Memory

from reward.get_main_reward import get_main_reward

from utils.general_utils import initialize_logger, close_logger, deque_to_csv
from utils.graph_utils import mols_to_pyg_batch

#####################################################
#                      PROCESS                      #
#####################################################

tasks = mp.JoinableQueue()
results = mp.Queue()


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
                print('\n%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            index, mol, done = next_task
            if index is None:
                self.result_queue.put((None, None, None, True))
                self.task_queue.task_done()
                continue
            # print('%s: Working' % proc_name)
            if done:
                self.timestep_counter = 0
                mol, candidates, done = self.env.reset(return_type='smiles')
            else:
                self.timestep_counter += 1
                mol, candidates, done = self.env.reset(mol, return_type='smiles')
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

#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_gpu_sync(args, embed_model, env):
    lr = (args.actor_lr, args.critic_lr, args.rnd_lr)
    betas = (args.beta1, args.beta2)
    eps = args.eps
    print("lr:", lr, "beta:", betas, "eps:", eps)  # parameters for Adam optimizer

    print('Creating %d processes' % args.nb_procs)
    workers = [Worker(env, tasks, results, args.max_timesteps) for i in range(args.nb_procs)]
    for w in workers:
        w.start()

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    policy = DGAPN(lr,
                betas,
                eps,
                args.eta,
                args.gamma,
                args.K_epochs,
                args.eps_clip,
                embed_model,
                args.emb_nb_shared,
                args.input_size,
                args.nb_edge_types,
                args.use_3d,
                args.gnn_nb_layers,
                args.gnn_nb_hidden,
                args.enc_num_layers,
                args.enc_num_hidden,
                args.enc_num_output,
                args.rnd_num_layers,
                args.rnd_num_hidden,
                args.rnd_num_output)
    if args.running_model_path != '':
        policy = torch.load(args.running_model_path)
    policy.to_device(device)
    logging.info(policy)

    sample_count = 0
    episode_count = 0
    save_counter = 0
    log_counter = 0

    avg_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    memories = [Memory() for _ in range(args.nb_procs)]
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=1000)
    # training loop
    i_episode = 0
    while i_episode < args.max_episodes:
        logging.info("collecting rollouts")
        for i in range(args.nb_procs):
            tasks.put((i, None, True))
        tasks.join()
        # unpack results
        mols = [None] * args.nb_procs
        done_idx = []
        notdone_idx, candidates, batch_idx = [], [], []
        for i in range(args.nb_procs):
            index, mol, cands, done = results.get()

            mols[index] = mol

            notdone_idx.append(index)
            candidates.append(cands)
            batch_idx.extend([index] * len(cands))
        while True:
            # action selections (for not done)
            if len(notdone_idx) > 0:
                states_emb, candidates_emb, action_logprobs, actions = policy.select_action(
                    [Chem.MolFromSmiles(mols[idx]) for idx in notdone_idx],
                    [Chem.MolFromSmiles(item) for sublist in candidates for item in sublist], batch_idx)
                if not isinstance(action_logprobs, list):
                    action_logprobs = [action_logprobs]
                    actions = [actions]
            else:
                if sample_count >= args.update_timesteps:
                    break

            for i, idx in enumerate(notdone_idx):
                tasks.put((idx, candidates[i][actions[i]], False))

                memories[idx].states.append(states_emb[i])
                memories[idx].candidates.append([data for j, data in enumerate(candidates_emb) if batch_idx[j] == idx])
                memories[idx].actions.append(actions[i])
                memories[idx].logprobs.append(action_logprobs[i])
            for idx in done_idx:
                if sample_count >= args.update_timesteps:
                    tasks.put((None, None, True))
                else:
                    tasks.put((idx, None, True))
            tasks.join()
            # unpack results
            mols = [None] * args.nb_procs
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
                    candidates.append(cands)
                    batch_idx.extend([index] * len(cands))
            # get final rewards (for previously not done but now done)
            nowdone_idx = [idx for idx in notdone_idx if idx in new_done_idx]
            stillnotdone_idx = [idx for idx in notdone_idx if idx in new_notdone_idx]
            if len(nowdone_idx) > 0:
                main_rewards = get_main_reward(
                    [Chem.MolFromSmiles(mols[idx]) for idx in nowdone_idx], reward_type=args.reward_type, args=args)
                if not isinstance(main_rewards, list):
                    main_rewards = [main_rewards]

            for i, idx in enumerate(nowdone_idx):
                main_reward = main_rewards[i]

                i_episode += 1
                running_reward += main_reward
                running_main_reward += main_reward
                writer.add_scalar("EpMainRew", main_reward, i_episode - 1)
                rewbuffer_env.append(main_reward)
                molbuffer_env.append((mols[idx], main_reward))
                writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode - 1)

                memories[idx].rewards.append(main_reward)
                memories[idx].is_terminals.append(True)
            for idx in stillnotdone_idx:
                running_reward += 0

                memories[idx].rewards.append(0)
                memories[idx].is_terminals.append(False)
            # get innovation rewards
            if (args.iota > 0 and 
                i_episode > args.innovation_reward_episode_delay and 
                i_episode < args.innovation_reward_episode_cutoff):
                if len(notdone_idx) > 0:
                    inno_rewards = policy.get_inno_reward(
                        [Chem.MolFromSmiles(mols[idx]) for idx in notdone_idx])
                    if not isinstance(inno_rewards, list):
                        inno_rewards = [inno_rewards]

                for i, idx in enumerate(notdone_idx):
                    inno_reward = args.iota * inno_rewards[i]

                    running_reward += inno_reward

                    memories[idx].rewards[-1] += inno_reward

            sample_count += len(notdone_idx)
            avg_length += len(notdone_idx)
            episode_count += len(nowdone_idx)

            done_idx = new_done_idx
            notdone_idx = new_notdone_idx

        for m in memories:
            memory.extend(m)
            m.clear()
        # update model
        logging.info("\n\nupdating policy @ episode %d..." % i_episode)
        policy.update(memory)
        memory.clear()

        save_counter += episode_count
        log_counter += episode_count

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            torch.save(policy, os.path.join(save_dir, 'DGAPN_continuous_solved_{}.pth'.format('test')))
            break

        # save every 500 episodes
        if save_counter >= args.save_interval:
            torch.save(policy, os.path.join(save_dir, '{:05d}_dgapn.pth'.format(i_episode)))
            deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'))
            save_counter = 0

        # save running model
        torch.save(policy, os.path.join(save_dir, 'running_dgapn.pth'))

        if log_counter >= args.log_interval:
            avg_length = int(avg_length / log_counter)
            running_reward = running_reward / log_counter
            running_main_reward = running_main_reward / log_counter

            logging.info('Episode {} \t Avg length: {} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, avg_length, running_reward, running_main_reward))
            running_reward = 0
            running_main_reward = 0
            avg_length = 0
            log_counter = 0

        episode_count = 0
        sample_count = 0

    close_logger()
    writer.close()
    # Add a poison pill for each process
    for i in range(args.nb_procs):
        tasks.put(None)
    tasks.join()

def train_serial(args, embed_model, env):
    lr = (args.actor_lr, args.critic_lr, args.rnd_lr)
    betas = (args.beta1, args.beta2)
    eps = args.eps
    print("lr:", lr, "beta:", betas, "eps:", eps) # parameters for Adam optimizer

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/' + args.name + '_' + dt))
    save_dir = os.path.join(args.artifact_path, 'saves/' + args.name + '_' + dt)
    os.makedirs(save_dir, exist_ok=True)
    initialize_logger(save_dir)

    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    policy = DGAPN(lr,
                betas,
                eps,
                args.eta,
                args.gamma,
                args.K_epochs,
                args.eps_clip,
                embed_model,
                args.emb_nb_shared,
                args.input_size,
                args.nb_edge_types,
                args.use_3d,
                args.gnn_nb_layers,
                args.gnn_nb_hidden,
                args.enc_num_layers,
                args.enc_num_hidden,
                args.enc_num_output,
                args.rnd_num_layers,
                args.rnd_num_hidden,
                args.rnd_num_output)
    if args.running_model_path != '':
        policy = torch.load(args.running_model_path)
    policy.to_device(device)
    logging.info(policy)

    time_step = 0

    avg_length = 0
    running_reward = 0
    running_main_reward = 0

    memory = Memory()
    rewbuffer_env = deque(maxlen=100)
    molbuffer_env = deque(maxlen=1000)
    # training loop
    for i_episode in range(1, args.max_episodes+1):
        state, candidates, done = env.reset()

        for t in range(args.max_timesteps):
            time_step += 1
            # Running policy_old:
            state_emb, candidates_emb, action_logprob, action = policy.select_action(state, candidates)
            memory.states.append(state_emb[0])
            memory.candidates.append(candidates_emb)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

            state, candidates, done = env.step(action)

            # done and reward may not be needed anymore
            reward = 0

            if (t==(args.max_timesteps-1)) or done:
                main_reward = get_main_reward(state, reward_type=args.reward_type, args=args)
                reward = main_reward
                running_main_reward += main_reward

            if (args.iota > 0 and 
                i_episode > args.innovation_reward_episode_delay and 
                i_episode < args.innovation_reward_episode_cutoff):
                inno_reward = policy.get_inno_reward(state)
                reward += inno_reward

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            running_reward += reward
            if done:
                break

        # update if it's time
        if time_step >= args.update_timesteps:
            logging.info("\n\nupdating policy @ episode %d..." % i_episode)
            time_step = 0
            policy.update(memory)
            memory.clear()

        writer.add_scalar("EpMainRew", main_reward, i_episode-1)
        rewbuffer_env.append(main_reward) # reward
        molbuffer_env.append((Chem.MolToSmiles(state), main_reward))
        avg_length += (t+1)

        # write to Tensorboard
        writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), i_episode-1)

        # stop training if avg_reward > solved_reward
        if np.mean(rewbuffer_env) > args.solved_reward:
            logging.info("########## Solved! ##########")
            torch.save(policy, os.path.join(save_dir, 'DGAPN_continuous_solved_{}.pth'.format('test')))
            break

        # save every save_interval episodes
        if (i_episode-1) % args.save_interval == 0:
            torch.save(policy, os.path.join(save_dir, '{:05d}_dgapn.pth'.format(i_episode)))
            deque_to_csv(molbuffer_env, os.path.join(save_dir, 'mol_dgapn.csv'))

        # save running model
        torch.save(policy, os.path.join(save_dir, 'running_dgapn.pth'))

        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length/args.log_interval)
            running_reward = running_reward/args.log_interval
            running_main_reward = running_main_reward/args.log_interval
            
            logging.info('Episode {} \t Avg length: {} \t Avg reward: {:5.3f} \t Avg main reward: {:5.3f}'.format(
                i_episode, avg_length, running_reward, running_main_reward))
            running_reward = 0
            running_main_reward = 0
            avg_length = 0

    close_logger()
    writer.close()

