import os
import argparse

import torch

from dgapn.DGAPN import init_DGAPN
from utils.general_utils import load_model
from environment.env import CReM_Env

from evaluate.eval_dgapn import eval_dgapn
from evaluate.eval_greedy import eval_greedy

def read_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--warm_start_dataset', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--run_id', default='')
    add_arg('--use_cpu', action='store_true')
    add_arg('--gpu', default='0')

    add_arg('--greedy', action='store_true')
    add_arg('--model_url', default='')
    add_arg('--model_path', default='')

    add_arg('--reward_type', type=str, default='plogp', help='logp;plogp;dock')

    add_arg('--nb_sample_crem', type=int, default=128)

    add_arg('--nb_test', type=int, default=50)
    add_arg('--nb_bad_steps', type=int, default=5)

    # AUTODOCK PARAMETERS
    add_arg('--obabel_path', default='')
    add_arg('--adt_path', default='')
    add_arg('--receptor_file', default='')

    return parser.parse_args()

def main():
    args = read_args()
    print("====args====\n", args)

    env = CReM_Env(args.data_path,
                args.warm_start_dataset,
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
                    K = args.nb_bad_steps,
                    args = args)
    else:
        # DGAPN
        state = load_model(artifact_path, args.model_url, args.model_path, name='model')
        model = init_DGAPN(state)
        print(model)
        eval_dgapn(artifact_path,
                    model,
                    env,
                    args.reward_type,
                    N = args.nb_test,
                    K = args.nb_bad_steps,
                    args = args)


if __name__ == '__main__':
    main()
