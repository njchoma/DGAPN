import numpy as np

import torch
from torch_geometric.data import Batch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_rewards(g_batch, surrogate_model):
    with torch.autograd.no_grad():
        scores = surrogate_model(g_batch.to(DEVICE))
    return scores.cpu().numpy()*-1

def greedy_rollout(env, surrogate_model, K, max_rollout=1000):
    g, g_candidates, done = env.reset()

    g = Batch.from_data_list([g])
    start_rew = get_rewards(g, surrogate_model)
    best_rew = start_rew
    steps_remaining = K


    for i in range(max_rollout):
        print("  {:3d} {:2d} {:4.1f}".format(i+1, steps_remaining, best_rew))
        steps_remaining -= 1
        next_rewards = get_rewards(g_candidates, surrogate_model)
        action = np.argmax(next_rewards)
        
        try:
            new_rew = next_rewards[action]
        except Exception as e:
            print(e)
            break


        g, g_candidates, done = env.step(action, include_current_state=False)

        if new_rew > best_rew:
            best_rew = new_rew
            steps_remaining = K
        
        if (steps_remaining == 0) or done:
            break

        
    return start_rew, best_rew

def eval_greedy(surrogate_model, env, N=30, K=4):

    surrogate_model = surrogate_model.to(DEVICE)
    surrogate_model.eval()
    print("\nStarting greedy...\n")
    avg_improvement = []
    avg_best = []
    for i in range(N):
        start_rew, best_rew = greedy_rollout(env, surrogate_model, K)
        improvement = best_rew - start_rew
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
