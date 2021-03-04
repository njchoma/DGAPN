import numpy as np

import torch
from torch_geometric.data import Batch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_rewards(g_batch, surrogate_model):
    with torch.autograd.no_grad():
        scores = surrogate_model(g_batch.to(DEVICE))
    return scores.cpu().numpy()*-1

def gcpn_crem_rollout(policy, env, surrogate_model, K, max_rollout=5):
    g, g_candidates, done = env.reset()

    g = Batch.from_data_list([g])
    start_rew = get_rewards(g, surrogate_model)
    best_rew = start_rew
    steps_remaining = K


    for i in range(max_rollout):
        print("  {:3d} {:2d} {:4.1f}".format(i+1, steps_remaining, best_rew))
        steps_remaining -= 1
        next_rewards = get_rewards(g_candidates, surrogate_model)
        # action = np.argmax(next_rewards)

        with torch.autograd.no_grad():
            _, _, probs = policy(g, g_candidates, surrogate_model)
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


        g, g_candidates, done = env.step(action, include_current_state=False)
        g = Batch.from_data_list([g]).to(DEVICE)

        if new_rew > best_rew:
            best_rew = new_rew
            steps_remaining = K
        
        if (steps_remaining == 0) or done:
            break

        
    return start_rew, best_rew
def eval_gcpn_crem(policy, surrogate_model, env, N=120, K=1):

    surrogate_model = surrogate_model.to(DEVICE)
    surrogate_model.eval()
    
    policy = policy.to(DEVICE)
    policy.eval()

    print("\nStarting gcpn_crem eval...\n")
    avg_improvement = []
    avg_best = []
    for i in range(N):
        start_rew, best_rew = gcpn_crem_rollout(policy, env, surrogate_model, K)
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
