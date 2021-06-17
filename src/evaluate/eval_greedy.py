import os
import numpy as np

import time
from datetime import datetime

from rdkit import Chem

from reward.get_main_reward import get_main_reward

def greedy_rollout(save_path, env, reward_type, K, max_rollout=6, args=None):
    mol, mol_candidates, done = env.reset()
    mol_best = mol

    new_rew = get_main_reward(mol, reward_type, args=args)[0]
    start_rew = new_rew
    best_rew = new_rew
    steps_remaining = K

    for i in range(max_rollout):
        print("  {:3d} {:2d} {:4.1f}".format(i+1,
                                             steps_remaining,
                                             new_rew))
        steps_remaining -= 1
        next_rewards = get_main_reward(mol_candidates, reward_type, args=args)

        action = np.argmax(next_rewards)

        try:
            new_rew = next_rewards[action]
        except Exception as e:
            print(e)
            break

        mol, mol_candidates, done = env.step(action, include_current_state=False)

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

def eval_greedy(artifact_path, env, reward_type, N=30, K=1, args=None):
    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_path = os.path.join(artifact_path, dt + '_greedy.csv')

    print("\nStarting greedy...\n")
    avg_improvement = []
    avg_best = []
    for i in range(N):
        start_rew, best_rew = greedy_rollout(save_path, env, reward_type, K, args=args)
        improvement = best_rew - start_rew
        print("{:2d}: {:4.1f} {:4.1f} {:4.1f}\n".format(i+1,
                                                      start_rew,
                                                      best_rew,
                                                      improvement))
        avg_improvement.append(improvement)
        avg_best.append(best_rew)
    avg_improvement = sum(avg_improvement) / len(avg_improvement)
    avg_best = sum(avg_best) / len(avg_best)
    print("Avg improvement over {} samples: {:5.2f}".format(N, avg_improvement))
    print("Avg best        over {} samples: {:5.2f}".format(N, avg_best))
