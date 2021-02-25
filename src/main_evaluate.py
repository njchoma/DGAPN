import os
import argparse
from datetime import datetime

import torch

from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import maybe_download_file
from gnn_surrogate import model
from gcpn.env import CReM_Env
# from evaluate.eval_gcpn_crem import eval_gcpn_crem
from evaluate.eval_greedy import eval_greedy

def molecule_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)

    add_arg('--surrogate_model_url', default='')
    add_arg('--surrogate_model_path', default='')

    add_arg('--nb_sample_crem', type=int, default=32)

    return parser


def get_current_datetime():
    now = datetime.now()
    dt_string = now.strftime("%Y.%m.%d_%H:%M:%S")
    return dt_string

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
    print("====args====", args)
    dt = get_current_datetime()
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/'+dt))
    
    surrogate_model = load_surrogate_model(args.artifact_path,
                                           args.surrogate_model_url,
                                           args.surrogate_model_path)

    env = CReM_Env(args.data_path,
                   nb_sample_crem = args.nb_sample_crem)

    print(surrogate_model)

    eval_greedy(surrogate_model, env)


if __name__ == '__main__':
    main()
