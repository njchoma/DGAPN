import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

from .gapn_policy import ActorCriticGAPN
from .rnd_explore import RNDistillation
from gnn_embed.model import MyGNN

from utils.graph_utils import mols_to_pyg_batch

#####################################################
#                   HELPER MODULES                  #
#####################################################

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.candidates = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def extend(self, memory):
        self.actions.extend(memory.actions)
        self.states.extend(memory.states)
        self.candidates.extend(memory.candidates)
        self.logprobs.extend(memory.logprobs)
        self.rewards.extend(memory.rewards)
        self.is_terminals.extend(memory.is_terminals)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.candidates[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

#################################################
#                  MAIN MODEL                   #
#################################################

class DGAPN(nn.Module):
    def __init__(self,
                 lr,
                 betas,
                 eps,
                 eta,
                 gamma,
                 K_epochs,
                 eps_clip,
                 emb_model,
                 input_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 enc_nb_layers,
                 enc_nb_hidden,
                 enc_nb_output,
                 rnd_nb_layers,
                 rnd_nb_hidden,
                 rnd_nb_output):
        super(DGAPN, self).__init__()
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.emb_model = emb_model
        self.emb_3d = emb_model.use_3d if emb_model is not None else False

        self.policy = ActorCriticGAPN(lr[:2],
                                      betas,
                                      eps,
                                      eta,
                                      eps_clip,
                                      input_dim,
                                      nb_edge_types,
                                      gnn_nb_layers,
                                      gnn_nb_hidden,
                                      enc_nb_layers,
                                      enc_nb_hidden,
                                      enc_nb_output)

        self.policy_old = ActorCriticGAPN(lr[:2],
                                          betas,
                                          eps,
                                          eta,
                                          eps_clip,
                                          input_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          enc_nb_layers,
                                          enc_nb_hidden,
                                          enc_nb_output)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.explore_critic = RNDistillation(lr[2],
                                             betas,
                                             eps,
                                             input_dim,
                                             nb_edge_types,
                                             gnn_nb_layers,
                                             gnn_nb_hidden,
                                             rnd_nb_layers,
                                             rnd_nb_hidden,
                                             rnd_nb_output)

        self.device = torch.device("cpu")

    def to_device(self, device):
        self.emb_model = self.emb_model.to(device) if self.emb_model is not None else None
        self.policy.to(device)
        self.policy_old.to(device)
        self.explore_critic.to(device)
        self.device = device

    def forward(self):
        raise NotImplementedError

    def select_action(self, states, candidates, batch_idx=None):
        if batch_idx is None:
            batch_idx = torch.zeros(len(candidates), dtype=torch.long)
        batch_idx = torch.LongTensor(batch_idx).to(self.device)

        states = mols_to_pyg_batch(states, self.emb_3d, device=self.device)
        candidates = mols_to_pyg_batch(candidates, self.emb_3d, device=self.device)

        with torch.autograd.no_grad():
            if self.emb_model is not None:
                states = self.emb_model.get_embedding(states, aggr=False)
                candidates = self.emb_model.get_embedding(candidates, aggr=False)
            action_logprobs, actions = self.policy_old.select_action(
                states, candidates, batch_idx)

        return states, candidates, action_logprobs, actions
    
    def get_inno_reward(self, states):
        states = mols_to_pyg_batch(states, self.emb_3d, device=self.device)

        if self.emb_model is not None:
            with torch.autograd.no_grad():
                states = self.emb_model.get_embedding(states, aggr=False)
        scores = self.explore_critic.get_score(states)
        return scores.tolist()

    def update(self, memory, eps=1e-5):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        # candidates batch
        batch_idx = []
        for i, cands in enumerate(memory.candidates):
            batch_idx.extend([i]*len(cands))
        batch_idx = torch.LongTensor(batch_idx).to(self.device)

        # convert list to tensor Batch().from_data_list([graph[0] for graph in graphs])
        old_states = Batch().from_data_list(memory.states).to(self.device)
        old_next_states = Batch().from_data_list([cands[a] for a, cands in zip(memory.actions, memory.candidates)]).to(self.device)
        old_candidates = Batch().from_data_list([item for sublist in memory.candidates for item in sublist]).to(self.device)
        old_actions = torch.tensor(memory.actions).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs).to(self.device)

        old_values = self.policy_old.get_value(old_states)

        # Optimize policy for K epochs:
        print("Optimizing...")

        for i in range(self.K_epochs):
            loss, baseline_loss = self.policy.update(old_states, old_candidates, old_actions, old_logprobs, old_values, rewards, batch_idx)
            rnd_loss = self.explore_critic.update(old_next_states)
            if (i%10)==0:
                print("  {:3d}: Actor Loss: {:7.3f}, Critic Loss: {:7.3f}, RND Loss: {:7.3f}".format(i, loss, baseline_loss, rnd_loss))

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n".format(repr(self.policy))

