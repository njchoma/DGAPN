import os
import argparse

import torch

from utils.general_utils import maybe_download_file

from gnn_embed import model

from environment.env import CReM_Env
from dgapn.gapn_policy import GAPN_Actor

from evaluate.eval_dgapn import eval_dgapn
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

    add_arg('--policy_path', default='')

    add_arg('--reward_type', type=str, default='logp', help='logp;dock')

    add_arg('--nb_sample_crem', type=int, default=128)

    add_arg('--nb_test', type=int, default=50)
    add_arg('--nb_bad_steps', type=int, default=5)

    return parser

def load_dgapn(policy_path):
    dgapn_model = torch.load(policy_path, map_location='cpu')
    print("DGAPN model loaded")
    return dgapn_model

def main():
    args = molecule_arg_parser().parse_args()
    print("====args====\n", args)

    env = CReM_Env(args.data_path,
                args.warm_start_dataset_path,
                nb_sample_crem = args.nb_sample_crem,
                mode='mol')

    artifact_path = os.path.join(args.artifact_path, args.name)
    os.makedirs(artifact_path, exist_ok=True)

    if args.greedy is True:
        # Greedy
        eval_greedy(artifact_path,
                    env,
                    args.reward_type,
                    N = args.nb_test,
                    K = args.nb_bad_steps)
    else:
        # DGAPN
        dgapn = load_dgapn(args.policy_path)
        policy = dgapn.policy.actor
        emb_model = dgapn.emb_model
        print(policy)
        eval_dgapn(artifact_path,
                    policy,
                    emb_model,
                    env,
                    args.reward_type,
                    N = args.nb_test,
                    K = args.nb_bad_steps)


if __name__ == '__main__':
    main()

