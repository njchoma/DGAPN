import numpy as np

import torch
from torch_geometric.data import Batch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_rewards(g_batch, surrogate_model):
    with torch.autograd.no_grad():
        scores = surrogate_model(g_batch.to(DEVICE))
    return scores.cpu().numpy()*-1

def greedy_rollout(env, surrogate_guide, surrogate_eval, K, max_rollout=5):
    g_start, g_candidates, done = env.reset()
    g_best = g_start

    g = Batch.from_data_list([g_start])
    best_rew = get_rewards(g, surrogate_guide)
    new_rew = best_rew
    steps_remaining = K


    for i in range(max_rollout):
        print("  {:3d} {:2d} {:4.1f}".format(i+1,
                                             steps_remaining,
                                             new_rew))
        steps_remaining -= 1
        next_rewards = get_rewards(g_candidates, surrogate_guide)
        action = np.argmax(next_rewards)
        
        try:
            new_rew = next_rewards[action]
        except Exception as e:
            print(e)
            break


        g, g_candidates, done = env.step(action, include_current_state=False)

        if new_rew > best_rew:
            g_best = g
            best_rew = new_rew
            steps_remaining = K
        
        if (steps_remaining == 0) or done:
            break

        
    start_rew = get_rewards(Batch.from_data_list([g_start]), surrogate_eval)
    try:
        final_rew = get_rewards(Batch.from_data_list([g_best]), surrogate_eval)
    except Exception as e:
        print(e)
        final_rew = start_rew
    return start_rew, final_rew

def eval_greedy(surrogate_guide, surrogate_eval, env, N=30, K=1):

    surrogate_guide = surrogate_guide.to(DEVICE)
    surrogate_guide.eval()
    surrogate_eval  = surrogate_eval.to(DEVICE)
    surrogate_eval.eval()
    print("\nStarting greedy...\n")
    avg_improvement = []
    avg_best = []
    for i in range(N):
        start_rew, best_rew = greedy_rollout(env, surrogate_guide, surrogate_eval, K)
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
