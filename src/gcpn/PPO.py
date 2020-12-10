import gym
import numpy as np
from collections import deque
import gzip
import shutil
import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from .gcpn_policy import GCPN, GCPN_crem
from .MLP import Critic, Discriminator

from utils.general_utils import load_surrogate_model, maybe_download_file
from utils.graph_utils import mol_to_pyg_graph
from utils.state_utils import wrap_state, nodes_to_atom_labels, dense_to_sparse_adj, state_to_graph, state_to_mol


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


#################################################
#                   GCPN PPO                    #
#################################################

class Critic(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(Critic, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers - 1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_layer(X).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, emb_dim, nb_layers, nb_hidden):
        super(Discriminator, self).__init__()
        layers = [nn.Linear(emb_dim, nb_hidden)]
        for _ in range(nb_layers - 1):
            layers.append(nn.Linear(nb_hidden, nb_hidden))

        self.layers = nn.ModuleList(layers)
        self.final_layer = nn.Linear(nb_hidden, 1)
        self.act = nn.ReLU()
        self.final_act = nn.Sigmoid()

    def forward(self, X):
        for i, l in enumerate(self.layers):
            X = self.act(l(X))
        return self.final_act(self.final_layer(X)).squeeze(1)


class ActorCriticGCPN(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 mlp_nb_layers,
                 mlp_nb_hidden,
                 crem,
                 sample_crem,
                 device):
        super(ActorCriticGCPN, self).__init__()

        self.crem = crem
        # action mean range -1 to 1
        if self.crem:
            self.actor = GCPN_crem(input_dim,
                                   gnn_nb_layers,
                                   gnn_nb_hidden,
                                   gnn_heads,
                                   emb_dim,
                                   mlp_nb_layers,
                                   mlp_nb_hidden,
                                   sample_crem,
                                   device)
        else:
            self.actor = GCPN(input_dim,
                              emb_dim,
                              nb_edge_types,
                              gnn_nb_layers,
                              gnn_nb_hidden,
                              gnn_heads,
                              mlp_nb_layers,
                              mlp_nb_hidden)
        # critic
        self.critic = Critic(emb_dim, mlp_nb_layers, mlp_nb_hidden)

        # discriminator
        self.discriminator = Discriminator(emb_dim, mlp_nb_layers, mlp_nb_hidden)

    def forward(self):
        raise NotImplementedError

    # TODO: Change what action is, does it need to be a mol? Do I need to return a list of mols?
    def act(self, state, memory):
        # If Crem, state is a pyg graph and action is a new pyg_graph. Otherwise, state is from molecule.py and action is an index.
        if self.crem:
            # states is a list of rdkit mols generated by CReM
            action, prob, states = self.actor(state)
            memory.states.append(states)
        else:
            action, prob = self.actor(state)
            memory.states.append(state.to_data_list()[0])

        action_logprob = torch.log(prob)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action

    def evaluate(self, state, action):
        # X_agg will be (n_step, hidden_dim). Probs is (n_step,4). If self.crem X_agg is same size, probs is (2000, 1).
        probs, X_agg = self.actor.evaluate(state, action)

        action_logprobs = torch.log(probs)
        state_value = self.critic(X_agg)
        fidelity = self.discriminator(X_agg)

        if self.crem:
            entropy = probs * action_logprobs.sum()
        else:
            entropy = (probs * action_logprobs).sum(1)
        entropy = (probs * action_logprobs).sum(1)

        return action_logprobs, state_value, entropy, fidelity

    def evaluate_disc(self, state):
        _, X_agg = self.actor.evaluate(state)
        fidelity = self.discriminator(X_agg)

        return fidelity


class PPO_GCPN:
    def __init__(self,
                 lr,
                 betas,
                 gamma,
                 eta,
                 upsilon,
                 alpha,
                 K_epochs,
                 eps_clip,
                 input_dim,
                 emb_dim,
                 nb_edge_types,
                 gnn_nb_layers,
                 gnn_nb_hidden,
                 gnn_heads,
                 mlp_nb_layers,
                 mlp_nb_hidden,
                 crem,
                 sample_crem,
                 device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eta = eta
        self.upsilon = upsilon
        self.alpha = alpha
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.crem = crem

        self.policy = ActorCriticGCPN(input_dim,
                                      emb_dim,
                                      nb_edge_types,
                                      gnn_nb_layers,
                                      gnn_nb_hidden,
                                      gnn_heads,
                                      mlp_nb_layers,
                                      mlp_nb_hidden,
                                      crem,
                                      sample_crem,
                                      device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCriticGCPN(input_dim,
                                          emb_dim,
                                          nb_edge_types,
                                          gnn_nb_layers,
                                          gnn_nb_hidden,
                                          gnn_heads,
                                          mlp_nb_layers,
                                          mlp_nb_hidden,
                                          crem,
                                          sample_crem,
                                          device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()  # combine operation for numerical stability

    def select_action(self, state, memory, env):
        if self.crem:
            g = state_to_mol(state, env, False)
        else:
            g = state_to_graph(state, env)
        # state = wrap_state(state).to(self.device)
        action = self.policy_old.act(g, memory)
        return action

    def update(self, memory, truth, i_episode, writer=None):
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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        if self.crem:
            old_states = [Batch().from_data_list([mol_to_pyg_graph(mol) for mol in mol_list]).to(self.device) \
                          for mol_list in memory.states]
            old_logprobs = torch.squeeze(torch.tensor(memory.logprobs)).to(self.device).detach()
            old_actions = torch.tensor(memory.actions).to(self.device).detach()
        else:
            old_states = Batch().from_data_list(memory.states).to(self.device)
            old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
            old_actions = torch.squeeze(torch.tensor(memory.actions).to(self.device), 1).detach()

        # Optimize policy for K epochs:
        print("Optimizing...")

        for i in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, entropies, fidelity = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # loss
            ## policy
            advantages = rewards - state_values.detach()
            loss = []

            for j in range(ratios.shape[1]):
                r = ratios[:, j]
                surr1 = r * advantages
                surr2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                l = -torch.min(surr1, surr2)
                if torch.isnan(l).any():
                    print("found nan in loss")
                    print(l)
                    print(torch.isnan(surr1).any())
                    print(torch.isnan(surr2).any())
                    print(torch.isnan(advantages).any())
                    exit()
                loss.append(l)
            loss = torch.stack(loss, 0).sum(0)
            ## entropy
            loss += self.eta * entropies

            loss = loss.mean()
            ## baseline
            loss += self.upsilon * self.MseLoss(state_values, rewards)
            if (i % 10) == 0:
                print("  {:3d}: Loss: {:7.3f}".format(i, loss))
            ## adversarial
            if (i + 1) == self.K_epochs:
                truth_fidelity = self.policy.evaluate_disc(truth)
                score = torch.cat((truth_fidelity, fidelity))
                objective = torch.cat((torch.ones_like(truth_fidelity), torch.zeros_like(fidelity)))

                adversarial_loss = self.alpha * self.BCEWithLogitsLoss(score, objective)
                print("  {:3d}: Adversarial Loss: {:7.3f}".format(i, adversarial_loss))
                loss += adversarial_loss

            ## take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update_disc(self, truth, i_episode, writer=None):
        fidelity = self.policy.evaluate_disc(truth)

        ## adversarial
        adversarial_loss = self.alpha * self.BCEWithLogitsLoss(fidelity, torch.ones_like(fidelity))
        print("  Adversarial Loss (Real): {:7.3f}".format(adversarial_loss))

        ## take gradient step
        self.optimizer.zero_grad()
        adversarial_loss.backward()
        self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def __repr__(self):
        return "{}\n{}".format(repr(self.policy), repr(self.optimizer))


#####################################################
#                   FINAL REWARDS                   #
#####################################################

def get_surrogate_reward(state, env, surrogate_model, device):
    #g = state_to_graph(state, env, keep_self_edges=False)
    # state is a pyg_graph now, temporary fix
    g = state.to(device)
    with torch.autograd.no_grad():
        pred_docking_score = surrogate_model(g, None)
    reward = pred_docking_score.item() * -1
    return reward


def get_adversarial_reward(state, env, policy_model, device):
    g = state_to_graph(state, env, keep_self_edges=False)
    g = g.to(device)
    with torch.autograd.no_grad():
        reward = torch.mean(policy_model.evaluate_disc(g))
    return reward


#####################################################
#                   TRAINING LOOP                   #
#####################################################

def train_ppo(args, env, writer=None):
    ############## Hyperparameters ##############
    render = True
    solved_reward = 100  # stop training if avg_reward > solved_reward
    log_interval = 80  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 2000  # update policy every n timesteps
    truth_frequency = 5  # frequency of feeding truth to discriminator

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0001  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    #############################################

    ob = env.reset()
    nb_edge_types = ob['adj'].shape[0]
    ob = state_to_graph(ob, env)
    input_dim = ob.x.shape[1]

    device = torch.device("cpu") if args.cpu else \
        torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else "cpu")
    ground_truth = pd.read_csv(args.conditional, header=None)

    ppo = PPO_GCPN(lr,
                   betas,
                   gamma,
                   args.eta,
                   args.upsilon,
                   args.alpha,
                   K_epochs,
                   eps_clip,
                   input_dim,
                   args.emb_size,
                   nb_edge_types,
                   args.layer_num_g,
                   args.num_hidden_g,
                   args.heads_g,
                   args.mlp_num_layer,
                   args.mlp_num_hidden,
                   args.use_crem,
                   args.sample_crem,
                   device)

    print(ppo)
    memory = Memory()
    print("lr:", lr, "beta:", betas)

    if args.use_crem:
        print("Using CReM")
        url = "http://www.qsar4u.com/files/cremdb/replacements02_sc2.db.gz"
        crem_gz_path = os.path.join(os.getcwd(), "replacements02_sc2.db.gz")
        crem_path = os.path.join(os.getcwd(), "replacements02_sc2.db")
        maybe_download_file(crem_gz_path, url, "CReM Database")
        with gzip.open(crem_gz_path, 'rb') as f_in:
            with open(crem_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if args.use_surrogate:
        print("{} episodes before surrogate model as final reward".format(args.surrogate_reward_episode_delay))

        surrogate_model = load_surrogate_model(args.artifact_path,
                                               args.surrogate_model_url,
                                               args.surrogate_model_path,
                                               device)
        print(surrogate_model)
        surrogate_model = surrogate_model.to(device)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    episode_count = 0

    # variables for plotting rewards

    rewbuffer_env = deque(maxlen=100)
    # training loop
    for i_episode in range(1, max_episodes + 1):
        cur_ep_ret_env = 0
        #tf_state
        state = env.reset()
        surr_reward = 0.0
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory, env)
            #Now state is pyg_graph
            state, reward, done, info = env.step(action, memory, crem=args.use_crem)

            if done:
                # surrogate
                if args.use_surrogate and (i_episode > args.surrogate_reward_episode_delay):
                    try:
                        surr_reward = get_surrogate_reward(state, env, surrogate_model, device)
                        reward += surr_reward / 5
                        info['surrogate_reward'] = surr_reward
                    except Exception as e:
                        print(e)
                        info['surrogate_reward'] = None
                        pass
                else:
                    info['surrogate_reward'] = None
                # adversarial
                if i_episode > args.adversarial_reward_episode_delay:
                    try:
                        advers_reward = get_adversarial_reward(state, env, ppo.policy, device)
                        # TODO for Nick: Rescale this reward.
                        #  Currently a random scale.
                        reward += advers_reward * 0.5
                        info['adversarial_reward'] = advers_reward
                    except Exception as e:
                        print(e)
                        info['adversarial_reward'] = None
                        pass
                else:
                    info['adversarial_reward'] = None
                # final
                info['final_reward'] = reward

                # From rl-baselines/baselines/ppo1/pposgd_simple_gcn.py in rl_graph_generation
                with open('molecule_gen/' + args.name + '.csv', 'a') as f:
                    if args.is_conditional:
                        start_mol, end_mol = Chem.MolFromSmiles(info['start_smile']), Chem.MolFromSmiles(info['smile'])
                        start_fingerprint, end_fingerprint = FingerprintMols.FingerprintMol(
                            start_mol), FingerprintMols.FingerprintMol(end_mol)
                        sim = DataStructs.TanimotoSimilarity(start_fingerprint, end_fingerprint)

                        row = ''.join(['{},'] * 12)[:-1] + '\n'
                        f.write(row.format(info['start_smile'], info['smile'], sim, info['reward_valid'],
                                           info['reward_qed'], \
                                           info['reward_sa'], info['final_stat'], info['flag_steric_strain_filter'],
                                           info['flag_zinc_molecule_filter'], \
                                           info['stop'], info['surrogate_reward'], info['final_reward']))
                    else:
                        row = ''.join(['{},'] * 10)[:-1] + '\n'
                        f.write(row.format(info['smile'], info['reward_valid'], info['reward_qed'], info['reward_sa'], \
                                           info['final_stat'], info['flag_steric_strain_filter'],
                                           info['flag_zinc_molecule_filter'], \
                                           info['stop'], info['surrogate_reward'], info['final_reward']))

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                print("updating ppo")
                # TODO: import a batch of true molecules here.
                #   The data type of `truth` should be `Batch`.
                #   `MyDataset` in `load_data.py` might be useful.
                truth_smi = random.sample(ground_truth, 100)
                truth = [mol_to_pyg_graph(Chem.MolFromSmiles(smi)) for smi in truth_smi]
                ppo.update(memory, truth, i_episode, writer)
                memory.clear_memory()

                if time_step % (update_timestep * truth_frequency) == 0:
                    # TODO for Nick: import a batch of true molecules here.
                    #   The structure of `truth` should be the same as `memory`.

                    truth_smi = random.sample(ground_truth, 100)
                    truth = [mol_to_pyg_graph(Chem.MolFromSmiles(smi)) for smi in truth_smi]
                    ppo.update_disc(truth, i_episode, writer)

            running_reward += reward
            cur_ep_ret_env += reward
            if (((i_episode + 1) % 20) == 0) and render:
                env.render()
            if done:
                break
        writer.add_scalar("EpSurrogate", -1 * surr_reward, episode_count)
        rewbuffer_env.append(cur_ep_ret_env)
        avg_length += time_step

        # write to Tensorboard
        writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env), episode_count)
        # writer.add_scalar("Average Length", avg_length, global_step=episode_count)
        # writer.add_scalar("Running Reward", running_reward, global_step=episode_count)
        episode_count += 1

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format('test'))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format('test'))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = running_reward / log_interval

            print('Episode {} \t Avg length: {} \t Avg reward: {:5.3f}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
