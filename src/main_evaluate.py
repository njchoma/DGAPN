import os
import argparse

import torch

from utils.general_utils import maybe_download_file

from gnn_surrogate import model

from gcpn.env import CReM_Env
from gcpn.gcpn_policy import GCPN_Actor

from evaluate.eval_gcpn_crem import eval_gcpn_crem
from evaluate.eval_greedy import eval_greedy

def molecule_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--warm_start_dataset_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--greedy', action='store_true')

    add_arg('--surrogate_model_url', default='')
    add_arg('--surrogate_guide_path', default='')
    add_arg('--surrogate_eval_path', default='')
    add_arg('--gcpn_path', default='')

    add_arg('--nb_sample_crem', type=int, default=128)

    add_arg('--nb_test', type=int, default=50)
    add_arg('--nb_bad_steps', type=int, default=6) # match max_timesteps in ppo

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

def load_gcpn(gcpn_path):
    gcpn_model = torch.load(gcpn_path, map_location='cpu')
    print("GCPN model loaded")
    return gcpn_model

def main():
    args = molecule_arg_parser().parse_args()
    print("====args====\n", args)

    env = CReM_Env(args.data_path,
                args.warm_start_dataset_path,
                nb_sample_crem = args.nb_sample_crem,
                mode='mol')

    surrogate_guide = load_surrogate_model(args.artifact_path,
                                           args.surrogate_model_url,
                                           args.surrogate_guide_path)
    surrogate_eval  = load_surrogate_model(args.artifact_path,
                                           '',
                                           args.surrogate_eval_path)
    print(surrogate_guide)

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    if args.greedy is True:
        # Greedy
        eval_greedy(artifact_path,
                    surrogate_guide,
                    surrogate_eval,
                    env,
                    N = args.nb_test,
                    K = args.nb_bad_steps)
    else:
        # GCPN_CReM
        policy = load_gcpn(args.gcpn_path)
        print(policy)
        eval_gcpn_crem(artifact_path,
                        policy,
                        surrogate_guide,
                        surrogate_eval,
                        env,
                        N = args.nb_test,
                        K = args.nb_bad_steps)


if __name__ == '__main__':
    main()

