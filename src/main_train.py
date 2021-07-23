import os
import argparse
import resource

import torch
import multiprocessing as mp
import torch.multiprocessing as tmp

from train.train_serial import train_serial
from train.train_cpu_sync import train_cpu_sync
from train.train_cpu_async import train_cpu_async
from train.train_gpu_sync import train_gpu_sync
from train.train_gpu_async import train_gpu_async

from dgapn.DGAPN import DGAPN, load_DGAPN

from utils.general_utils import load_model

from environment.env import CReM_Env

def read_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--run_id', default='')
    add_arg('--use_cpu', action='store_true')
    add_arg('--gpu', default='0')
    add_arg('--nb_procs', type=int, default=4)
    add_arg('--mode', default='gpu_sync', help='cpu_sync;cpu_async;gpu_sync;gpu_async')
    #add_arg('--seed', help='RNG seed', type=int, default=666)

    add_arg('--warm_start_dataset', default='')
    add_arg('--running_model_path', default='')
    add_arg('--log_interval', type=int, default=20)         # print avg reward in the interval
    add_arg('--save_interval', type=int, default=400)       # save model in the interval

    add_arg('--reward_type', type=str, default='plogp', help='logp;plogp;dock')

    add_arg('--iota', type=float, default=0.05, help='relative weight for innovation reward')
    add_arg('--innovation_reward_episode_delay', type=int, default=100)
    add_arg('--innovation_reward_episode_cutoff', type=int, default=1000)

    # TRAINING PARAMETERS
    add_arg('--solved_reward', type=float, default=100)     # stop training if avg_reward > solved_reward
    add_arg('--max_episodes', type=int, default=50000)      # max training episodes
    add_arg('--max_timesteps', type=int, default=12)        # max timesteps in one episode
    add_arg('--update_timesteps', type=int, default=200)    # update policy every n timesteps
    add_arg('--k_epochs', type=int, default=50)             # update policy for K epochs
    add_arg('--eps_clip', type=float, default=0.2)          # clip parameter for PPO
    add_arg('--gamma', type=float, default=0.99)            # discount factor
    add_arg('--eta', type=float, default=0.01)              # relative weight for entropy loss
    add_arg('--actor_lr', type=float, default=5e-4)         # learning rate for actor
    add_arg('--critic_lr', type=float, default=1e-4)        # learning rate for critic
    add_arg('--rnd_lr', type=float, default=2e-3)           # learning rate for random network
    add_arg('--beta1', type=float, default=0.9)             # beta1 for Adam optimizer
    add_arg('--beta2', type=float, default=0.999)           # beta2 for Adam optimizer
    add_arg('--eps', type=float, default=0.01)              # eps for Adam optimizer

    # NETWORK PARAMETERS
    add_arg('--embed_model_url', default='')
    add_arg('--embed_model_path', default='')
    add_arg('--emb_nb_inherit', type=int, default=2)         # number of layers to inherit from the embedding model

    add_arg('--input_size', type=int, default=121)
    add_arg('--nb_edge_types', type=int, default=1)
    add_arg('--use_3d', action='store_true')
    add_arg('--gnn_nb_layers', type=int, default=3)         # number of gnn layers on top of the inherited layers
    add_arg('--gnn_nb_shared', type=int, default=2)         # number of shared layers for Q, K within the gnn layers
    add_arg('--gnn_nb_hidden', type=int, default=256, help='hidden size of Graph Networks')
    add_arg('--enc_num_layers', type=int, default=3)
    add_arg('--enc_num_hidden', type=int, default=256, help='hidden size of Encoding Networks')
    add_arg('--enc_num_output', type=int, default=256)
    add_arg('--rnd_num_layers', type=int, default=1)
    add_arg('--rnd_num_hidden', type=int, default=256, help='hidden size of Random Networks')
    add_arg('--rnd_num_output', type=int, default=8)

    # AUTODOCK PARAMETERS
    add_arg('--obabel_path', default='')
    add_arg('--adt_path', default='')
    add_arg('--receptor_file', default='')

    return parser.parse_args()

if __name__ == '__main__':
    args = read_args()
    print("====args====\n", args)

    # Process
    #args.nb_procs = mp.cpu_count()

    # Optimizer
    args.lr = (args.actor_lr, args.critic_lr, args.rnd_lr)
    args.betas = (args.beta1, args.beta2)
    print("lr:", args.lr, "beta:", args.betas, "eps:", args.eps)

    # Input
    embed_state = None
    if args.embed_model_url != '' or args.embed_model_path != '':
        embed_state = load_model(args.artifact_path,
                                    args.embed_model_url,
                                    args.embed_model_path,
                                    name='embed_model')
        assert args.emb_nb_inherit <= embed_state['nb_layers']
        if not embed_state['use_3d']:
            assert not args.use_3d
        args.input_size = embed_state['nb_hidden']
        args.nb_edge_types = embed_state['nb_edge_types']
    args.embed_state = embed_state

    # Environment
    env = CReM_Env(args.data_path, args.warm_start_dataset, mode='mol')
    #ob, _, _ = env.reset(return_type='pyg')
    #assert ob.x.shape[1] == args.input_size

    # Model
    if args.running_model_path != '':
        model = load_DGAPN(args.running_model_path)
    else:
        model = DGAPN(args.lr,
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

    # Device
    args.device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")
    model.to_device(args.device)

    # Training
    if args.nb_procs > 1:
        if args.mode == 'cpu_sync':
            mp.set_start_method('fork', force=True)
            train_cpu_sync(args, env, model)
        elif args.mode == 'cpu_async':
            mp.set_start_method('fork', force=True)
            train_cpu_async(args, env, model)
        elif args.mode == 'gpu_sync':
            mp.set_start_method('fork', force=True)
            train_gpu_sync(args, env, model)
        elif args.mode == 'gpu_async':
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
            #tmp.set_sharing_strategy('file_system')

            tmp.set_start_method('spawn', force=True)
            manager = mp.Manager()
            model.share_memory()
            train_gpu_async(args, env, model, manager)
        else:
            raise ValueError("Mode not recognized.")
    else:
        train_serial(args, env, model)
