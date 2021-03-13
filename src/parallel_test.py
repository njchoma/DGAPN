import torch
import torch.nn as nn

from torch.multiprocessing import Pool, Process, Lock, Barrier, Value, Queue


NUM_LAYERS =  8
NUM_HIDDEN = 2048

class Critic(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(Critic, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers-1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)



def collect_trajectories(p_state_dict, i):
    
    model = Critic(NUM_HIDDEN, NUM_LAYERS, NUM_HIDDEN)
    model.load_state_dict(p_state_dict)
    if i==0:
        print(model)
    print(i)
    return 0



def main():

    ppo = Critic(NUM_HIDDEN, NUM_LAYERS, NUM_HIDDEN)

    p_state_dict = ppo.state_dict()

    # parallel runs
    pool = Pool(4)
    results = []
    for i in range(pool._processes):
        res = pool.apply_async(collect_trajectories, [p_state_dict, i])
        results.append(res)
    pool.close()
    pool.join()
    print("\nDone.")

if __name__ == "__main__":
    main()
