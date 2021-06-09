import os
import sys
import yaml
import logging
import pickle
import numpy as np

import time
from datetime import datetime

from rdkit import Chem

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch_geometric as pyg

import utils.graph_utils as graph_utils
import utils.general_utils as general_utils
from .sGAT import sGAT, save_sGAT, load_sGAT

torch.multiprocessing.set_sharing_strategy('file_system')

#####################################################
#                   MODEL HANDLING                  #
#####################################################
def load_current_model(model_path):
    net = load_sGAT(os.path.join(model_path, 'current_model.pt'))
    return net

def load_best_model(model_path):
    net = load_sGAT(os.path.join(model_path, 'best_model.pt'))
    return net

def save_current_model(net, model_path):
    save_sGAT(net, os.path.join(model_path, 'current_model.pt'))

def save_best_model(net, model_path):
    save_sGAT(net, os.path.join(model_path, 'best_model.pt'))

#############################################
#            Custom Functions               #
#############################################
def dock_score_weights(scores):
    """If sample has a docking score south of a random split point, it is more likely to be sampled in the batch."""
    weights = np.zeros(len(scores))
    for idx, score in enumerate(scores):
        if score < split:
            weight = upsampled_weight
        else:
            weight = 1 - upsampled_weight
        weights[idx] = weight
    return weights


def exp_weighted_mse(output, target):
    """Custom loss function assigning greater weight to errors at the top of the ranked list."""
    epsilon = 0.001  # To avoid nan's?
    weight = torch.clamp((torch.exp(-(target - exp_loc) / exp_scale) / exp_scale), min=0.0, max=1)
    loss = torch.mean(weight * (output - target) ** 2) + epsilon
    return loss


#############################################
#                   DATA                    #
#############################################

def get_dense_edges(n):
    x = np.arange(n)
    src, dst = [np.tile(x, len(x)), np.repeat(x, len(x))]
    return torch.tensor([src, dst], dtype=torch.long)


class MolData(Dataset):
    def __init__(self, logp, smiles, use_3d):
        super(MolData, self).__init__()
        self.logp = logp
        self.smiles = smiles
        self.use_3d = use_3d

    def __getitem__(self, index):
        logp = self.logp[index]
        smiles = self.smiles[index]
        # Hot fix, get first in list if mol is none...
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            smiles = self.smiles[0]
            logp = self.logp[0]
            mol = Chem.MolFromSmiles(smiles)
            print("Invalid SMILE encountered. Using first row instead.")

        g = graph_utils.mol_to_pyg_graph(mol, self.use_3d)
        return torch.FloatTensor([logp]), g[0], g[1]

    def __len__(self):
        return len(self.logp)

    def get_graph_spec(self):
        y, g, _ = self[0]
        nb_node_feats = g.x.shape[1]
        try:
            nb_edge_feats = g.edge_attr.shape[1]
        except Exception as e:
            nb_edge_feats = 1
        return nb_node_feats, nb_edge_feats

    def compute_baseline_error(self):
        logp = np.array(self.logp)
        mean = logp.mean()
        sq_sum = np.sum(np.square(logp - mean)) / len(logp)
        logging.info("{:5.3f} baseline L2 loss\n".format(sq_sum))


def create_datasets(logp, smiles, use_3d, np_seed=0):
    nb_samples = len(logp)
    assert nb_samples > 10

    nb_train = int(nb_samples * 0.6)
    nb_valid = int(nb_samples * 0.2)

    np.random.seed(np_seed)
    sample_order = np.random.permutation(nb_samples)

    logp = np.asarray(logp)[sample_order].tolist()
    smiles = np.asarray(smiles)[sample_order].tolist()

    train_data = MolData(logp[:nb_train], smiles[:nb_train], use_3d)
    valid_data = MolData(logp[nb_train:nb_train + nb_valid],
                         smiles[nb_train:nb_train + nb_valid],
                         use_3d)
    test_data = MolData(logp[nb_train + nb_valid:],
                        smiles[nb_train + nb_valid:],
                        use_3d)
    return train_data, valid_data, test_data


def my_collate(samples):
    y = [s[0] for s in samples]
    g1 = [s[1] for s in samples]
    g2 = [s[2] for s in samples]

    y = torch.cat(y, dim=0)
    G1 = pyg.data.Batch().from_data_list(g1)
    if None in g2:
        return y, G1, None
    else:
        G2 = pyg.data.Batch().from_data_list(g2)
        return y, G1, G2




def parse_raw_data(raw_dataset):
    batch_size = 32
    loader = DataLoader(raw_dataset,
                        shuffle=False,
                        collate_fn=my_collate,
                        batch_size=batch_size,
                        num_workers=16)
    all_data = []
    print("\nPreprocessing {} samples".format(len(raw_dataset)))
    for i, d in enumerate(loader):
        if (i % 3)==0:
            print("{:7d}".format(i*batch_size))
            print(len(all_data))
        all_data.append(d)
        if i==20:
            break
    return all_data


def parse_data_path(data_path, use_3d):
    path_split = data_path.split('/')
    parent_path = '/'.join(path_split[:-1])
    data_name = path_split[-1].split('.')[0]
    storage_path = os.path.join(parent_path, data_name)

    if use_3d:
        storage_path += '_with_3d'
    else:
        storage_path += '_no_3d'

    os.makedirs(storage_path, exist_ok=True)
    return storage_path
    
def preprocess_data(raw_dataset, storage_path, dataset_name):
    dataset_path = os.path.join(storage_path, dataset_name+'.pkl')
    print(dataset_path)
    try:
        with open(dataset_path, 'rb') as f:
            parsed_data = pickle.load(f)
        print("Preprocessed {} set loaded".format(dataset_name))
    except Exception as e:
        print("Preprocessed {} set not found. Parsing...".format(dataset_name))
        t0 = time.time()
        parsed_data = parse_raw_data(raw_dataset)
        print("{:5.2f}s for {} samples".format(time.time()-t0, len(parsed_data)))
        with open(dataset_path, 'wb') as f:
            pickle.dump(parsed_data, f)
        print("Done.")
    exit()
    return parsed_data


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
    for i, (y, G1, G2) in enumerate(loader):
        t1 = time.time()
        if train:
            optim.zero_grad()
        y = y.to(DEVICE, non_blocking=True)
        G1 = G1.to(DEVICE)
        G2 = G2.to(DEVICE) if G2 is not None else None

        y_pred = net(G1, G2)

        loss = criterion(y_pred, y)
        with torch.autograd.set_detect_anomaly(True):
            if train:
                loss.backward()
                optim.step()
        epoch_loss += loss.item()

        if ((i + 1) % (nb_batch // print_freq)) == 0:
            nb_proc = (i + 1) * batch_size
            logging.info("    {:8d}: {:4.2f}".format(nb_proc, epoch_loss / (i + 1)))
        elapsed += time.time() - t1

    logging.info("  Model elapsed:  {:.2f}".format(elapsed))
    logging.info("  Loader elapsed: {:.2f}".format(time.time() - t0 - elapsed))
    logging.info("  Total elapsed:  {:.2f}".format(time.time() - t0))
    return epoch_loss / nb_batch


def train(net,
          criterion,
          batch_size,
          train_loader,
          valid_loader,
          optim,
          arg_handler,
          save_dir,
          writer):
    current_lr = optim.param_groups[0]['lr']
    lr_end = current_lr / 10 ** 3

    best_loss = arg_handler('best_loss')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3, verbose=True)
    scheduler.step(best_loss)
    for i in range(arg_handler('current_epoch'), 1000):
        t0 = time.time()
        logging.info("\n\nEpoch {}".format(i + 1))
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
        writer.add_scalar('lr', current_lr, i)
        writer.add_scalar('train_loss', train_loss, i)
        writer.add_scalar('valid_loss', valid_loss, i)
        #writer.add_scalars('loss',
        #                   {'train': train_loss, 'valid': valid_loss},
        #                   i)
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            logging.info("Best performance on valid set")
            best_loss = valid_loss
            save_best_model(net, save_dir)
        logging.info("{:6.1f} seconds, this epoch".format(time.time() - t0))

        current_lr = optim.param_groups[0]['lr']
        arg_handler.update_args(current_lr, i + 1, best_loss)
        save_current_model(net, save_dir)
        if current_lr < lr_end:
            break


#############################################
#                   ARGS                    #
#############################################
class ArgumentHandler:
    def __init__(self, experiment_dir, starting_lr):
        self.arg_file = os.path.join(experiment_dir, 'args.yaml')
        try:
            self.load_args()
            logging.info("Arguments loaded.")
        except Exception as e:
            self.initialize_args(starting_lr)
            logging.info("Arguments initialized.")

    def load_args(self):
        with open(self.arg_file, 'r') as f:
            self.args = yaml.load(f, Loader=yaml.FullLoader)

    def initialize_args(self, starting_lr):
        args = {}
        args['current_epoch'] = 0
        args['current_lr'] = starting_lr
        args['best_loss'] = 10 ** 10
        self.args = args
        self.save_args()

    def save_args(self):
        with open(self.arg_file, 'w') as f:
            yaml.dump(self.args, f)

    def update_args(self, current_lr, current_epoch, best_loss):
        self.args['current_lr'] = current_lr
        self.args['current_epoch'] = current_epoch
        self.args['best_loss'] = best_loss
        self.save_args()

    def __call__(self, param):
        return self.args[param]


#############################################
#                   MAIN                    #
#############################################

def main(artifact_path,
         logp,
         smiles,
         gpu_num=0,
         upsample=False,
         exp_loss=False,
         use_3d=False,
         batch_size=128,
         num_workers=12,
         nb_hidden=256,
         nb_layers=4,
         lr=0.001,
         store_preprocessed=False,
         data_path=None):
    # Global variables: GPU Device, random splits for upsampling, loc and scale parameter for exp weighted loss.
    global DEVICE
    global split
    global upsampled_weight
    global exp_loc
    global exp_scale

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(gpu_num))
    else:
        DEVICE = 'cpu'

    # logging variables
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    writer = SummaryWriter(log_dir=os.path.join(artifact_path, 'runs/' + dt))
    save_dir = os.path.join(artifact_path, 'saves/' + dt)
    os.makedirs(save_dir, exist_ok=True)
    general_utils.initialize_logger(save_dir)

    arg_handler = ArgumentHandler(save_dir, lr)

    train_data, valid_data, test_data = create_datasets(logp, smiles, use_3d)
    valid_data.compute_baseline_error()
    print("Dataset created")

    if (data_path is not None) and store_preprocessed:
        print("Using stored dataset. Preprocessing if necessary.")
        storage_path = parse_data_path(data_path, use_3d)
        train_data = preprocess_data(train_data, storage_path, 'train')
        valid_data = preprocess_data(valid_data, storage_path, 'valid')
        test_data  = preprocess_data(test_data, storage_path, 'test')

    if upsample:
        # Percentiles used in dock score weights.
        # Reset randomness
        np.random.seed()
        #train_25 = np.percentile(train_data.logp, 25)
        #train_75 = np.percentile(train_data.logp, 75)
        upsampled_weight = np.random.uniform(0.5, 1, 1)[0]
        #split = np.random.uniform(train_25, train_75, 1)[0]
        split = np.percentile(train_data.logp, 1)
        logging.info("Upsampling weights: {:3.2f}".format(upsampled_weight))
        logging.info("Upsampling split: {:3.2f}".format(split))

        # Initialize weighted sampler
        train_weights = torch.DoubleTensor(dock_score_weights(train_data.logp))
        valid_weights = torch.DoubleTensor(dock_score_weights(valid_data.logp))
        # test_weights = torch.DoubleTensor(dock_score_weights(test_data.logp))

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, len(valid_weights))
        # test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

        train_loader = DataLoader(train_data,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers)
        valid_loader = DataLoader(valid_data,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(train_data,
                                  shuffle=True,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  num_workers=num_workers)
        valid_loader = DataLoader(valid_data,
                                  collate_fn=my_collate,
                                  batch_size=batch_size,
                                  num_workers=num_workers)


    try:
        net = load_current_model(save_dir)
        logging.info("Model restored")
    except Exception as e:
        input_dim, nb_edge_types = train_data.get_graph_spec()
        net = sGAT(input_dim=input_dim,
                        nb_hidden=nb_hidden,
                        nb_layers=nb_layers,
                        nb_edge_types=nb_edge_types,
                        use_3d=use_3d)
        logging.info(net)
        logging.info("New model created")
    net = net.to(DEVICE)

    optim = torch.optim.Adam(net.parameters(), lr=arg_handler('current_lr'))

    if exp_loss:
        np.random.seed()
        exp_loc = min(train_data.logp)
        exp_scale = np.random.uniform(1, 4, 1)[0]
        logging.info("Exponential loc: {:3.2f}".format(exp_loc))
        logging.info("Exponential scale: {:3.2f}".format(exp_scale))
        criterion = exp_weighted_mse
    else:
        criterion = torch.nn.MSELoss()

    train(net,
          criterion,
          batch_size,
          train_loader,
          valid_loader,
          optim,
          arg_handler,
          save_dir,
          writer)

    general_utils.close_logger()
    writer.close()
    return load_best_model(save_dir)
