import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

from .gapn_policy import ActorCriticGAPN
from .rnd_explore import RNDistillation
from gnn_surrogate.model import GNN_MyGAT

from utils.graph_utils import mols_to_pyg_batch

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
                 emb_model=None,
                 input_dim=None,
                 emb_dim=None,
                 nb_edge_types=None,
                 gnn_nb_layers=None,
                 gnn_nb_hidden=None,
                 use_3d=None,
                 enc_nb_layers=None,
                 enc_nb_hidden=None,
                 enc_nb_output=None,
                 rnd_nb_layers=None,
                 rnd_nb_hidden=None,
                 rnd_nb_output=None):
        super(DGAPN, self).__init__()
        self.gamma = gamma
        self.K_epochs = K_epochs

        if emb_model is not None:
            input_dim = emb_model.input_dim
            emb_dim = emb_model.emb_dim
            nb_edge_types = emb_model.nb_edge_types
            gnn_nb_layers = emb_model.nb_layers
            gnn_nb_hidden = emb_model.nb_hidden
            use_3d = emb_model.use_3d

        self.policy = ActorCriticGAPN(lr[:2],
                                      betas,
                                      eps,
                                      eta,
                                      eps_clip,
                                      emb_model,
                                      input_dim,
                                      emb_dim,
                                      nb_edge_types,
                                      gnn_nb_layers,
                                      gnn_nb_hidden,
                                      use_3d,
                                      enc_nb_layers,
                                      enc_nb_hidden,
                                      enc_nb_output)

        self.policy_old = ActorCriticGAPN(lr[:2],
                                          betas,
                                          eps,
                                          eta,
                                          eps_clip,
                                          emb_model,
                                          input_dim,
                                          emb_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          use_3d,
                                          enc_nb_layers,
                                          enc_nb_hidden,
                                          enc_nb_output)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.explore_critic = RNDistillation(lr[2],
                                             betas,
                                             eps,
                                             input_dim,
                                             emb_dim,
                                             nb_edge_types,
                                             gnn_nb_layers,
                                             gnn_nb_hidden,
                                             use_3d,
                                             rnd_nb_layers,
                                             rnd_nb_hidden,
                                             rnd_nb_output)

        self.device = torch.device("cpu")

    def to_device(self, device):
        self.policy.to(device)
        self.policy_old.to(device)
        self.explore_critic.to(device)
        self.device = device

    def forward(self):
        raise NotImplementedError

    def select_action(self, states, candidates, batch_idx=None, return_shifted=False):
        if batch_idx is None:
            batch_idx = torch.zeros(len(torch.unique(candidates[0].batch)), dtype=torch.long)
        batch_idx = batch_idx.to(self.device)

        with torch.autograd.no_grad():
            g_emb, g_next_emb, g_candidates_emb, action_logprobs, actions, shifted_actions = self.policy_old.select_action(
                states, candidates, batch_idx)

        if return_shifted:
            return [g_emb, g_next_emb, g_candidates_emb], action_logprobs, actions, shifted_actions
        else:
            return [g_emb, g_next_emb, g_candidates_emb], action_logprobs, actions

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

        # convert list to tensor
        old_states = torch.cat(([m[0] for m in memory.states]),dim=0).to(self.device)
        old_next_states = torch.cat(([m[1] for m in memory.states]),dim=0).to(self.device)
        old_candidates = Batch().from_data_list([Data(x=m[2]) for m in memory.states]).to(self.device)
        old_actions = torch.tensor(memory.actions).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs).to(self.device)

        old_values = self.policy_old.get_value(old_states)

        # Optimize policy for K epochs:
        print("Optimizing...")

        for i in range(self.K_epochs):
            loss, baseline_loss = self.policy.update(old_states, old_candidates, old_actions, old_logprobs, old_values, rewards)
            rnd_loss = self.explore_critic.update(old_next_states)
            if (i%10)==0:
                print("  {:3d}: Actor Loss: {:7.3f}, Critic Loss: {:7.3f}, RND Loss: {:7.3f}".format(i, loss, baseline_loss, rnd_loss))

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n".format(repr(self.policy))


#####################################################
#                      REWARDS                      #
#####################################################

def get_surr_reward(states, surrogate_model, device):
    g = mols_to_pyg_batch(states, surrogate_model.use_3d, device=device)

    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g)
    return (-pred_docking_score).tolist()

def get_inno_reward(states, emb_model, explore_critic, device):
    g = mols_to_pyg_batch(states, emb_model.use_3d, device=device)

    with torch.autograd.no_grad():
        X = emb_model.get_embedding(g)
    scores = explore_critic.get_score(X)
    return scores.tolist()

