import os
import sys
import argparse

import torch
import torch.multiprocessing as mp

from dgapn.train import train_gpu_sync, train_serial

from utils.general_utils import maybe_download_file
from gnn_embed import model
from environment.env import CReM_Env

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
    add_arg('--embed_model_url', default='')
    add_arg('--embed_model_path', default='')

    add_arg('--iota', type=float, default=0.5, help='relative weight for innovation reward')
    add_arg('--innovation_reward_episode_delay', type=int, default=100)

    # ENVIRONMENT PARAMETERS
    #add_arg('--dataset', type=str, default='zinc', help='caveman; grid; ba; zinc; gdb')
    #add_arg('--logp_ratio', type=float, default=1)
    #add_arg('--qed_ratio', type=float, default=1)
    #add_arg('--sa_ratio', type=float, default=1)
    #add_arg('--reward_step_total', type=float, default=0.5)
    #add_arg('--normalize_adj', type=int, default=0)
    add_arg('--reward_type', type=str, default='plogp', help='plogp;logp;dock')
    #add_arg('--reward_target', type=float, default=0.5, help='target reward value')
    #add_arg('--has_feature', type=int, default=0)
    #add_arg('--is_conditional', type=int, default=0) # default 0
    #add_arg('--conditional', type=str, default='low') # default 0
    #add_arg('--max_action', type=int, default=128) # default 0
    #add_arg('--min_action', type=int, default=20) # default 0

    # NETWORK PARAMETERS
    add_arg('--input_size', type=int, default=121)
    #add_arg('--emb_size', type=int, default=128)
    add_arg('--nb_edge_types', type=int, default=1)
    add_arg('--gnn_nb_layers', type=int, default=1) # number of layers on top of the embedding model
    add_arg('--gnn_nb_hidden', type=int, default=256, help='hidden size of Graph Networks')
    add_arg('--enc_num_layers', type=int, default=4)
    add_arg('--enc_num_hidden', type=int, default=64, help='hidden size of Encoding Networks')
    add_arg('--enc_num_output', type=int, default=64)
    add_arg('--rnd_num_layers', type=int, default=3)
    add_arg('--rnd_num_hidden', type=int, default=128, help='hidden size of Random Networks')
    add_arg('--rnd_num_output', type=int, default=4)

    return parser

def load_embed_model(artifact_path, embed_model_url, embed_model_path):
    if embed_model_url != '':
        embed_model_path = os.path.join(artifact_path, 'embed_model.pth')

        maybe_download_file(embed_model_path,
                            embed_model_url,
                            'embed model')
    embed_model = torch.load(embed_model_path, map_location='cpu')
    print("embed model loaded")
    return embed_model

def main():
    args = molecule_arg_parser().parse_args()
    #args.nb_procs = mp.cpu_count()

    try:
        embed_model = load_embed_model(args.artifact_path,
                                            args.embed_model_url,
                                            args.embed_model_path)
        embed_model.eval()
        print(embed_model)
        args.input_size = embed_model.nb_hidden
        args.nb_edge_types = embed_model.nb_edge_types
    except Exception as e:
        print(e)
        embed_model = None

    env = CReM_Env(args.data_path, args.warm_start_dataset_path, mode='mol')
    #ob, _, _ = env.reset()
    #args.input_size = ob.x.shape[1]

    print("====args====\n", args)

    if args.nb_procs > 1:
        train_gpu_sync(args, embed_model, env)
    else:
        train_serial(args, embed_model, env)

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
