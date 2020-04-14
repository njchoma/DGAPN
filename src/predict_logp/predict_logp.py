import os
import logging
import numpy as np

import time

from rdkit import Chem

import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric as pyg


import utils.graph_utils as graph_utils
import utils.general_utils as general_utils
from .model import GNN

if torch.cuda.is_available():
    DEVICE='cuda'
else:
    DEVICE='cpu'

#############################################
#                   DATA                    #
#############################################

class MolData(Dataset):
    def __init__(self, logp, smiles):
        super(MolData, self).__init__()
        self.logp = logp
        self.smiles = smiles

    def __getitem__(self, index):
        logp = self.logp[index]
        smiles = self.smiles[index]

        mol = Chem.MolFromSmiles(smiles)
        g = graph_utils.mol_to_pyg_graph(mol)
        return g, torch.FloatTensor([logp])

    def __len__(self):
        return len(self.logp)

    def get_input_dim(self):
        g, y = self[0]
        input_dim = g.x.shape[1]
        return input_dim

    def compute_baseline_error(self):
        logp = np.array(self.logp)
        mean = logp.mean()
        sq_sum = np.sum(np.square(logp-mean)) / len(logp)
        logging.info("{:5.3f} baseline L2 loss\n".format(sq_sum))

def create_datasets(logp, smiles):
    nb_samples = len(logp)
    assert nb_samples > 10

    nb_train = int(nb_samples * 0.6)
    nb_valid = int(nb_samples * 0.2)

    sample_order = np.random.permutation(nb_samples)

    logp = np.asarray(logp)[sample_order].tolist()
    smiles = np.asarray(smiles)[sample_order].tolist()

    train_data = MolData(logp[:nb_train], smiles[:nb_train])
    valid_data = MolData(logp[nb_train:nb_train+nb_valid],
                         smiles[nb_train:nb_train+nb_valid])
    test_data  = MolData(logp[nb_train+nb_valid:], smiles[nb_train+nb_valid:])
    return train_data, valid_data, test_data

def my_collate(samples):
    g = [s[0] for s in samples]
    y = [s[1] for s in samples]

    G = pyg.data.Batch().from_data_list(g)
    y = torch.cat(y, dim=0)
    return G, y

#################################################
#                   TRAINING                    #
#################################################

def proc_one_epoch(net,
		   criterion,
                   batch_size,
                   loader,
                   optim=None,
                   train=False):
    print_freq = 10 if train else 4
    nb_batch = len(loader)
    nb_samples = nb_batch * batch_size

    epoch_loss = 0.0
    elapsed = 0.0

    if train:
        net.train()
    else:
        net.eval()

    t0 = time.time()
    logging.info("  {} batches, {} samples".format(nb_batch, nb_samples))
    for i, (G, y) in enumerate(loader):
        t1 = time.time()
        if train:
            optim.zero_grad()
        y = y.to(DEVICE, non_blocking=True)
        G = G.to(DEVICE)
        y_pred = net(G)

        loss = criterion(y_pred, y)
        if train:
            loss.backward()
            optim.step()
        epoch_loss += loss.item()

        if ((i+1)%(nb_batch//print_freq))==0:
            nb_proc = (i+1)*batch_size
            logging.info("    {:8d}: {:4.2f}".format(nb_proc, epoch_loss / (i+1)))
        elapsed += time.time() - t1

    logging.info("  Model elapsed:  {:.2f}".format(elapsed))
    logging.info("  Loader elapsed: {:.2f}".format(time.time()-t0-elapsed))
    logging.info("  Total elapsed:  {:.2f}".format(time.time()-t0))
    return epoch_loss / nb_batch


def train(net,
          criterion,
          batch_size,
          train_loader,
          valid_loader,
          optim,
          starting_epoch=0,
          best_loss=10**10):

    current_lr = optim.param_groups[0]['lr']
    lr_end = current_lr / 10**3

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
    scheduler.step(best_loss)
    for i in range(starting_epoch, 100):
        t0 = time.time()
        logging.info("\n\nEpoch {}".format(i+1))
        logging.info("Learning rate: {0:.3g}".format(current_lr))
        logging.info("  Train:")
        train_loss = proc_one_epoch(net,
                                    criterion,
                                    batch_size,
                                    train_loader,
                                    optim,
                                    train=True)
        logging.info("\n  Valid:")
        valid_loss = proc_one_epoch(net,
                                    criterion,
                                    batch_size,
                                    valid_loader)
        logging.info("Train MSE: {:3.2f}".format(train_loss))
        logging.info("Valid MSE: {:3.2f}".format(valid_loss))
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            logging.info("Best performance on valid set")
            best_loss = valid_loss
        logging.info("{:6.1f} seconds, this epoch".format(time.time() - t0))

        current_lr = optim.param_groups[0]['lr']
        if current_lr < lr_end:
            break


#############################################
#                   MAIN                    #
#############################################

def main(artifact_path,
         logp,
         smiles,
         batch_size=128,
         num_workers=24,
         nb_hidden=256,
         nb_layer=6,
         lr=0.001):
    artifact_path = os.path.join(artifact_path, 'predict_logp')
    os.makedirs(artifact_path, exist_ok=True)
    general_utils.initialize_logger(artifact_path)

    train_data, valid_data, test_data = create_datasets(logp, smiles)
    train_loader = DataLoader(train_data,
                              shuffle=True,
                              collate_fn=my_collate,
                              batch_size=batch_size,
                              num_workers=num_workers)
    valid_loader = DataLoader(valid_data,
                              collate_fn=my_collate,
                              batch_size=batch_size*2,
                              num_workers=num_workers)
    test_loader =  DataLoader(test_data,
                              collate_fn=my_collate,
                              batch_size=batch_size*2,
                              num_workers=num_workers)

    valid_data.compute_baseline_error()

    net = GNN(input_dim = train_data.get_input_dim(),
              nb_hidden = nb_hidden,
              nb_layer  = nb_layer)
    net = net.to(DEVICE)
    logging.info(net)

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train(net,
          criterion,
          batch_size,
          train_loader,
          valid_loader,
          optim)

    general_utils.close_logger()
