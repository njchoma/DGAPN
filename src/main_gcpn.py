import os
import re
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
    add_arg('--surrogate_reward_timestep_delay', type=int, default=0)

    # ENVIRONMENT PARAMETERS
    #add_arg('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    #add_arg('--logp_ratio', type=float, default=1)
    #add_arg('--qed_ratio', type=float, default=1)
    #add_arg('--sa_ratio', type=float, default=1)
    #add_arg('--reward_step_total', type=float, default=0.5)
    #add_arg('--normalize_adj', type=int, default=0)
    #add_arg('--reward_type', type=str, default='qed',help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
    #add_arg('--reward_target', type=float, default=0.5,help='target reward value')
    #add_arg('--has_feature', type=int, default=0)
    #add_arg('--is_conditional', type=int, default=0) # default 0
    #add_arg('--conditional', type=str, default='low') # default 0
    #add_arg('--max_action', type=int, default=128) # default 0
    #add_arg('--min_action', type=int, default=20) # default 0

    # NETWORK PARAMETERS
    #add_arg('--input_size', type=int, default=256)
    #add_arg('--emb_size', type=int, default=256) # default 64
    #add_arg('--nb_edge_types', type=int, default=1)
    #add_arg('--layer_num_g', type=int, default=4)
    #add_arg('--num_hidden_g', type=int, default=512)
    add_arg('--mlp_num_layer', type=int, default=3)
    add_arg('--mlp_num_hidden', type=int, default=128)

    # LOSS PARAMETERS
    add_arg('--eta', type=float, default=0.01, help='relative weight for entropy loss')
    add_arg('--upsilon', type=float, default=0.5, help='relative weight for baseline loss')

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

def get_surrogate_dims(surrogate_model):
    state_dict = surrogate_model.state_dict()
    layers_name = [s for s in state_dict.keys() if re.compile('^layers\.[0-9]+\.weight$').match(s)]
    input_dim = state_dict[layers_name[0]].size(0)
    emb_dim = state_dict[layers_name[0]].size(-1)
    nb_edge_types = 1
    nb_hidden = state_dict[layers_name[-1]].size(-1)
    nb_layer = len(layers_name)
    return input_dim, emb_dim, nb_edge_types, nb_hidden, nb_layer

def main():
    args = molecule_arg_parser().parse_args()
    #args.nb_procs = mp.cpu_count()

    surrogate_model = load_surrogate_model(args.artifact_path,
                                           args.surrogate_model_url,
                                           args.surrogate_model_path)
    args.input_size, args.emb_size, args.nb_edge_types, args.num_hidden_g, args.layer_num_g = get_surrogate_dims(surrogate_model)

    env = CReM_Env(args.data_path, args.warm_start_dataset_path)
    #ob, _, _ = env.reset()
    #args.input_size = ob.x.shape[1]

    print("====args====\n", args)
    print("{} episodes before surrogate model as final reward".format(
                    args.surrogate_reward_timestep_delay))

    train_ppo(args, surrogate_model, env)

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
