import gym
from gym_molecule.envs.molecule import GraphEnv

from .gcpn_policy import GCPN
from .PPO import train_ppo

def train(args, surrogate_model, device, seed, writer=None):
    # MAKE ENVIRONMENT
    env = gym.make('molecule-v0')
    env.init(data_type=args.dataset,
             logp_ratio=args.logp_ratio,
             qed_ratio=args.qed_ratio,
             sa_ratio=args.sa_ratio,
             reward_step_total=args.reward_step_total,
             is_normalize=args.normalize_adj,
             reward_type=args.reward_type,
             reward_target=args.reward_target,
             has_feature=bool(args.has_feature),
             is_conditional=bool(args.is_conditional),
             conditional=args.conditional,
             max_action=args.max_action,
             min_action=args.min_action) # remember call this after gym.make!!
    print(env.observation_space)

    env.seed(seed)
    # ob = env.reset()
    # nb_edge_types = ob['adj'].shape[0]
    # input_dim = ob['node'].shape[2]

    # # INITIALIZE POLICY
    # policy = GCPN(input_dim,
    #               args.emb_size,
    #               nb_edge_types,
    #               args.layer_num_g,
    #               args.num_hidden_g,
    #               args.num_hidden_g,
    #               args.mlp_num_layer,
    #               args.mlp_num_hidden)

    # print(policy)


    # TRAIN
    train_ppo(args, surrogate_model, env, device, writer=writer)

    env.close()
    print("Environment successfully closed.")
