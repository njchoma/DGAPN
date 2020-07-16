import os
import argparse
from mpi4py import MPI

from gcpn.train import train

from torch.utils.tensorboard import SummaryWriter

def molecule_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_arg = parser.add_argument

    # EXPERIMENT PARAMETERS
    add_arg('--data_path', required=True)
    add_arg('--artifact_path', required=True)
    add_arg('--name', default='default_run')
    add_arg('--seed', help='RNG seed', type=int, default=666)

    # ENVIRONMENT PARAMETERS
    add_arg('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    add_arg('--logp_ratio', type=float, default=1)
    add_arg('--qed_ratio', type=float, default=1)
    add_arg('--sa_ratio', type=float, default=1)
    add_arg('--reward_step_total', type=float, default=0.5)
    add_arg('--normalize_adj', type=int, default=0)
    add_arg('--reward_type', type=str, default='logppen',help='logppen;logp_target;qed;qedsa;qed_target;mw_target;gan')
    add_arg('--reward_target', type=float, default=0.5,help='target reward value')
    add_arg('--has_feature', type=int, default=0)
    add_arg('--is_conditional', type=int, default=0) # default 0
    add_arg('--conditional', type=str, default='low') # default 0
    add_arg('--max_action', type=int, default=128) # default 0
    add_arg('--min_action', type=int, default=20) # default 0

    # POLICY PARAMETERS
    add_arg('--emb_size', type=int, default=128) # default 64
    add_arg('--layer_num_g', type=int, default=3)
    add_arg('--num_hidden_g', type=int, default=128)
    add_arg('--mlp_num_layer', type=int, default=3)
    add_arg('--mlp_num_hidden', type=int, default=128)


    # add_arg('--env', type=str, help='environment name: molecule; graph',
    #                     default='molecule')
    # add_arg('--num_steps', type=int, default=int(5e7))
    # add_arg('--name', type=str, default='test_conditional')
    # add_arg('--name_load', type=str, default='0new_concatno_mean_layer3_expert1500')
    # # add_arg('--name_load', type=str, default='test')
    # add_arg('--dataset_load', type=str, default='zinc')
    # add_arg('--gan_step_ratio', type=float, default=1)
    # add_arg('--gan_final_ratio', type=float, default=1)
    # add_arg('--lr', type=float, default=1e-3)
    # # add_arg('--has_rl', type=int, default=1)
    # # add_arg('--has_expert', type=int, default=1)
    # add_arg('--has_d_step', type=int, default=1)
    # add_arg('--has_d_final', type=int, default=1)
    # add_arg('--has_ppo', type=int, default=1)
    # add_arg('--rl_start', type=int, default=250)
    # add_arg('--rl_end', type=int, default=int(1e6))
    # add_arg('--expert_start', type=int, default=0)
    # add_arg('--expert_end', type=int, default=int(1e6))
    # add_arg('--save_every', type=int, default=200)
    # add_arg('--load', type=int, default=0)
    # add_arg('--load_step', type=int, default=250)
    # # add_arg('--load_step', type=int, default=0)
    # add_arg('--curriculum', type=int, default=0)
    # add_arg('--curriculum_num', type=int, default=6)
    # add_arg('--curriculum_step', type=int, default=200)
    # add_arg('--supervise_time', type=int, default=4)
    # add_arg('--layer_num_d', type=int, default=3)
    # add_arg('--graph_emb', type=int, default=0)
    # add_arg('--stop_shift', type=int, default=-3)
    # add_arg('--has_residual', type=int, default=0)
    # add_arg('--has_concat', type=int, default=0)
    # add_arg('--gcn_aggregate', type=str, default='mean')# sum, mean, concat
    # add_arg('--gan_type', type=str, default='normal')# normal, recommend, wgan
    # add_arg('--gate_sum_d', type=int, default=0)
    # add_arg('--mask_null', type=int, default=0)
    # add_arg('--bn', type=int, default=0)
    # add_arg('--name_full',type=str,default='')
    # add_arg('--name_full_load',type=str,default='')

    return parser

def main():
    args = molecule_arg_parser().parse_args()
    print("====args====", args)
    writer = SummaryWriter(log_dir=os.path.join(args.artifact_path, 'runs/'))

    train(args,seed=args.seed,writer=writer)

if __name__ == '__main__':
    main()
