import os
import numpy as np

import time
from datetime import datetime

from rdkit import Chem

import torch

from reward.get_main_reward import get_main_reward

from utils.graph_utils import mols_to_pyg_batch

def dgapn_rollout(save_path,
                    model,
                    env,
                    reward_type,
                    K,
                    max_rollout=20,
                    args=None):
    device = torch.device("cpu") if args.use_cpu else torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")

    model.to_device(device)
    model.eval()

    mol, mol_candidates, done = env.reset()
    smile_best = Chem.MolToSmiles(mol, isomericSmiles=False)
    emb_model_3d = model.emb_model.use_3d if model.emb_model is not None else model.use_3d

    g = mols_to_pyg_batch(mol, emb_model_3d, device=device)
    if model.emb_model is not None:
        with torch.autograd.no_grad():
            g = model.emb_model.get_embedding(g, n_layers=model.emb_nb_shared, return_3d=model.use_3d, aggr=False)
    new_rew = get_main_reward(mol, reward_type, args=args)[0]
    start_rew = new_rew
    best_rew = new_rew
    steps_remaining = K

    for i in range(max_rollout):
        print("  {:3d} {:2d} {:4.1f}".format(i+1, steps_remaining, best_rew))
        steps_remaining -= 1
        g_candidates = mols_to_pyg_batch(mol_candidates, emb_model_3d, device=device)
        if model.emb_model is not None:
            with torch.autograd.no_grad():
                g_candidates = model.emb_model.get_embedding(g_candidates, n_layers=model.emb_nb_shared, return_3d=model.use_3d, aggr=False)
        # next_rewards = get_main_reward(mol_candidates, reward_type, args=args)

        with torch.autograd.no_grad():
            probs, _, _ = model.policy.actor(g, g_candidates, torch.zeros(len(mol_candidates), dtype=torch.long).to(device))
        probs = probs.cpu().numpy()

        max_action = np.argmax(probs)
        min_action = np.argmin(probs)
        # print(next_rewards[max_action], next_rewards[min_action])
        # print(probs.shape, next_rewards.shape)
        # p = probs.unsqueeze(1)
        # r = torch.FloatTensor(next_rewards).unsqueeze(1).to(device)
        # c = torch.cat((p, r), dim=1)
        # for s in c:
        #     print("{:5.3f} {:4.1f}".format(s[0], s[1]))
        # # print(c)
        # exit()

        action = max_action
        mol, mol_candidates, done = env.step(action, include_current_state=False)

        try:
            # new_rew = next_rewards[action]
            new_rew = get_main_reward([mol], reward_type,args=args)[0]
        except Exception as e:
            print(e)
            break

        g = mols_to_pyg_batch(mol, emb_model_3d, device=device)
        if model.emb_model is not None:
            with torch.autograd.no_grad():
                g = model.emb_model.get_embedding(g, n_layers=model.emb_nb_shared, return_3d=model.use_3d, aggr=False)

        if new_rew > best_rew:
            smile_best = Chem.MolToSmiles(mol, isomericSmiles=False)
            best_rew = new_rew
            steps_remaining = K

        if (steps_remaining == 0) or done:
            break

    with open(save_path, 'a') as f:
        print("Writing SMILE molecules!")

        print(smile_best, best_rew)
        row = ''.join(['{},'] * 2)[:-1] + '\n'
        f.write(row.format(smile_best, best_rew))

    return start_rew, best_rew

def eval_dgapn(artifact_path, model, env, reward_type, N=120, K=1, args=None):
    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_path = os.path.join(artifact_path, dt + '_dgapn.csv')

    print("\nStarting dgapn eval...\n")
    avg_improvement = []
    avg_best = []
    for i in range(N):
        start_rew, best_rew = dgapn_rollout(save_path,
                                            model,
                                            env,
                                            reward_type,
                                            K,
                                            args=args)
        improvement = best_rew - start_rew
        print("Improvement ", improvement)
        print("{:2d}: {:4.1f} {:4.1f} {:4.1f}".format(i+1,
                                                      start_rew,
                                                      best_rew,
                                                      improvement))
        avg_improvement.append(improvement)
        avg_best.append(best_rew)
    avg_improvement = sum(avg_improvement) / len(avg_improvement)
    avg_best = sum(avg_best) / len(avg_best)
    print("Avg improvement over {} samples: {:5.2f}".format(N, avg_improvement))
    print("Avg best        over {} samples: {:5.2f}".format(N, avg_best))
