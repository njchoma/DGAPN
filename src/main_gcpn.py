import os
import sys
import argparse

import torch
import torch.multiprocessing as mp

from gcpn.PPO import train_ppo

from utils.general_utils import maybe_download_file
from gnn_surrogate import model
from gcpn.env import CReM_Env

def molecule_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--use_cpu', action='store_true')
    add_arg('--gpu', default='0')
    add_arg('--nb_procs', type=int, default=4)
    #add_arg('--seed', help='RNG seed', type=int, default=666)

    add_arg('--warm_start_dataset_path', default='')
    add_arg('--surrogate_model_url', default='')
    add_arg('--surrogate_model_path', default='')

    add_arg('--iota', type=float, default=0.5, help='relative weight for exploration reward')
    add_arg('--innovation_reward_episode_delay', type=int, default=1000)

    add_arg('--alpha', type=float, default=5., help='relative weight for adversarial reward')
    add_arg('--adversarial_reward_episode_cutoff', type=int, default=100)

    # ENVIRONMENT PARAMETERS
    #add_arg('--dataset', type=str, default='zinc', help='caveman; grid; ba; zinc; gdb')
    #add_arg('--logp_ratio', type=float, default=1)
    #add_arg('--qed_ratio', type=float, default=1)
    #add_arg('--sa_ratio', type=float, default=1)
    #add_arg('--reward_step_total', type=float, default=0.5)
    #add_arg('--normalize_adj', type=int, default=0)
    #add_arg('--reward_type', type=str, default='qed', help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
    #add_arg('--reward_target', type=float, default=0.5, help='target reward value')
    #add_arg('--has_feature', type=int, default=0)
    #add_arg('--is_conditional', type=int, default=0) # default 0
    #add_arg('--conditional', type=str, default='low') # default 0
    #add_arg('--max_action', type=int, default=128) # default 0
    #add_arg('--min_action', type=int, default=20) # default 0

    # NETWORK PARAMETERS
    #add_arg('--input_size', type=int, default=256)
    #add_arg('--emb_size', type=int, default=256)
    add_arg('--output_size', type=int, default=4, help='output size of RND')
    #add_arg('--nb_edge_types', type=int, default=1)
    #add_arg('--gnn_nb_layers', type=int, default=4)
    #add_arg('--gnn_nb_hidden', type=int, default=512, help='hidden size of Graph Neural Network')
    add_arg('--acp_num_layers', type=int, default=4)
    add_arg('--acp_num_hidden', type=int, default=128, help='hidden size of Actor-Critic Policy')
    add_arg('--rnd_num_layers', type=int, default=2)
    add_arg('--rnd_num_hidden', type=int, default=256, help='hidden size of Random Network Distillation')

    return parser

def load_surrogate_model(artifact_path, surrogate_model_url, surrogate_model_path):
    if surrogate_model_url != '':
        surrogate_model_path = os.path.join(artifact_path, 'surrogate_model.pth')

        maybe_download_file(surrogate_model_path,
                            surrogate_model_url,
                            'Surrogate model')
    surrogate_model = torch.load(surrogate_model_path, map_location='cpu')
    print("Surrogate model loaded")
    return surrogate_model

def main():
    args = molecule_arg_parser().parse_args()
    #args.nb_procs = mp.cpu_count()

    surrogate_model = load_surrogate_model(args.artifact_path,
                                           args.surrogate_model_url,
                                           args.surrogate_model_path)
    args.input_size, args.emb_size, args.nb_edge_types, args.gnn_nb_layers, args.gnn_nb_hidden = None, None, None, None, None

    env = CReM_Env(args.data_path, args.warm_start_dataset_path, mode='mol')
    #ob, _, _ = env.reset()
    #args.input_size = ob.x.shape[1]

    print("====args====\n", args)

    train_ppo(args, surrogate_model, env)

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
