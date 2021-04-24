import os
import numpy as np

import time
from datetime import datetime

from rdkit import Chem

import torch
from torch_geometric.data import Batch

from utils.graph_utils import mol_to_pyg_graph

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_rewards(g_batch, surrogate_model):
    with torch.autograd.no_grad():
        scores = surrogate_model(g_batch.to(DEVICE))
    return scores.cpu().numpy()*-1

def gcpn_crem_rollout(save_path,
                      policy,
                      env,
                      surrogate_guide,
                      surrogate_eval,
                      K,
                      max_rollout=6):
    mol, mol_candidates, done = env.reset()
    mol_start = mol
    mol_best = mol

    g = Batch().from_data_list([mol_to_pyg_graph(mol)[0]])
    new_rew = get_rewards(g, surrogate_guide)
    start_rew = new_rew
    best_rew = new_rew
    steps_remaining = K

    for i in range(max_rollout):
        print("  {:3d} {:2d} {:4.1f}".format(i+1, steps_remaining, best_rew))
        steps_remaining -= 1
        g_candidates = Batch().from_data_list([mol_to_pyg_graph(cand)[0] for cand in mol_candidates])
        next_rewards = get_rewards(g_candidates, surrogate_guide)

        with torch.autograd.no_grad():
            _, _, _, probs, _, _, _ = policy(g, g_candidates, torch.empty(len(mol_candidates), dtype=torch.long).fill_(0))
        max_action = np.argmax(probs.cpu().numpy())
        min_action = np.argmin(probs.cpu().numpy())

        # print(next_rewards[max_action], next_rewards[min_action])
        action = max_action
        # print(probs.shape, next_rewards.shape)
        # p = probs.unsqueeze(1)
        # r = torch.FloatTensor(next_rewards).unsqueeze(1).to(DEVICE)
        # c = torch.cat((p, r), dim=1)
        # for s in c:
        #     print("{:5.3f} {:4.1f}".format(s[0], s[1]))
        # # print(c)
        # exit()

        try:
            new_rew = next_rewards[action]
        except Exception as e:
            print(e)
            break

        mol, mol_candidates, done = env.step(action, include_current_state=False)
        g = Batch().from_data_list([mol_to_pyg_graph(mol)[0]])

        if new_rew > best_rew:
            mol_best = mol
            best_rew = new_rew
            steps_remaining = K

        if (steps_remaining == 0) or done:
            break

    with open(save_path, 'a') as f:
        print("Writing SMILE molecules!")

        smile = Chem.MolToSmiles(mol_best, isomericSmiles=False)
        print(smile, new_rew)
        row = ''.join(['{},'] * 2)[:-1] + '\n'
        f.write(row.format(smile, new_rew))

    return start_rew, best_rew

def eval_gcpn_crem(artifact_path, policy, surrogate_guide, surrogate_eval, env, N=120, K=1):
    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_path = os.path.join(artifact_path, dt + '.csv')

    surrogate_guide = surrogate_guide.to(DEVICE)
    surrogate_guide.eval()
    surrogate_eval = surrogate_eval.to(DEVICE)
    surrogate_eval.eval()

    policy = policy.to(DEVICE)
    policy.eval()

    print("\nStarting gcpn_crem eval...\n")
    avg_improvement = []
    avg_best = []
    for i in range(N):
        start_rew, best_rew = gcpn_crem_rollout(save_path,
                                                policy,
                                                env,
                                                surrogate_guide,
                                                surrogate_eval,
                                                K)
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
