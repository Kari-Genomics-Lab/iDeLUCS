################ I CHANGED THE LEARNING RATE  ---------------------------
# lr=5e-4

import sys
import torch
import random
import math
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

sys.path.append('src/')

from LossFunctions import IID_loss, simCLR_loss, info_nce_loss, Cluster_loss
from torch.utils.data import DataLoader
from PytorchUtils import myNet, NetLinear

from utils import byte_iterfasta, FastaEntry, \
    SequenceDataset, create_dataloaders, \
    transition_transversion, cluster_acc

# Random Seeds for reproducibility.
# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(0)


# from torch.utils.tensorboard import SummaryWriter

global dtype
global EPS

dtype = torch.FloatTensor
EPS = sys.float_info.epsilon


def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class IID_model():
    def __init__(self, args):

        d = {4: 135, 5: 511, 6: 2079}  # This is based on dinemsionality reduction

        self.k = args['k']
        self.n_features = d[args['k']]
        self.net = myNet(self.n_features, args['n_clusters'])
        self.net.apply(weights_init)
        self.epoch = 0
        self.EPS = sys.float_info.epsilon
        self.sequence_file = args['sequence_file']
        self.GT_file = args['GT_file']
        self.n_mimics = args['n_mimics']
        self.batch_sz = args['batch_sz']

        self.optimizer = args['optimizer']
        self.l = args['lambda']
        self.noise = args['noise']
        self.mutate = False

        # self.writer = SummaryWriter()

        print(self.net)
        print("Number of Trainable Parameters: ",
              sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def build_dataloaders(self):
        # Data Files
        data_path = self.sequence_file
        GT_file = self.GT_file
        print(data_path)
        # Retrieve the numer and define the mutations
        # You can include the whatever callable transformation you want.

        n_mimics = self.n_mimics

        # Building the data loaders.
        batch_size = self.batch_sz
        assert batch_size % n_mimics == 0, "batch_size must be a multiple of n_mimics"

        if self.mutate:
            mutations = []
            for i in range(n_mimics):
                mutations.append(transition_transversion(1e-4, 0.5e-4))
        else:
            mutations = [None] * n_mimics

        self.idx, self.dataloaders = create_dataloaders(data_path,
                                                        mutations, k=self.k,
                                                        batch_sz=batch_size // n_mimics,
                                                        GT_file=GT_file)

        random.shuffle(self.idx)

    def unsupervised_training_epoch(self):

        optimizer = optim.Adam(self.net.parameters(), lr=2.8e-4, weight_decay=0.01)
        n_features = self.n_features

        batch_size = self.batch_sz
        batch_sz = batch_size // self.n_mimics
        num_train_batches = len(self.dataloaders[0])

        n_mimics = self.n_mimics
        self.net.train()

        iterators = (d for d in self.dataloaders)
        running_loss = 0.0

        random.shuffle(self.idx)

        for i_batch, batch in enumerate(zip(*iterators)):

            # Batch is a tuple with the structure: [original, mimic_1, ... , mimic_n]
            # Each of the kmer tensor in the tuple is of size [mini_b_sz, 4**k]
            # So we need to resize the data into a "true batch" to feed into the network

            original_true_batch = torch.zeros(batch_size, 1, n_features).type(dtype)
            mimics_true_batch = torch.zeros(batch_size, 1, n_features).type(dtype)

            original = (batch[0]['kmer']).view(-1, 1, n_features).type(dtype)  # The first in the tuple
            mini_b_sz = original.shape[0]

            for d_i in range(n_mimics):
                mimics = batch[d_i + 1]['kmer']
                assert mimics.shape[0] == mini_b_sz

                start = d_i * mini_b_sz
                original_true_batch[start: start + mini_b_sz, :, :] = original
                mimics_true_batch[start: start + mini_b_sz, :, :] = mimics.view(-1, 1, n_features).type(dtype)

            if (mini_b_sz != batch_size // n_mimics) and (self.epoch == 0):
                print(f'Last mini batch size: {mini_b_sz}')

            true_size = mini_b_sz * n_mimics
            original_true_batch = original_true_batch[:true_size, :, :]
            mimics_true_batch = mimics_true_batch[:true_size, :, :]
            # zero the gradients
            self.net.zero_grad()

            # Forward + Backwards + Optimize

            z1, h1 = self.net(original_true_batch)
            z2, h2 = self.net(mimics_true_batch)

            loss = IID_loss(z1, z2, lamb=self.l) + info_nce_loss(h1, h2, 1)
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)

            # self.writer.add_scalar('Loss', loss, (self.epoch*num_train_batches)+ i_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # running_loss /= i_batch
        self.epoch += 1
        print(f'Epoch: {self.epoch} \t Loss: {running_loss / (i_batch + 1)}')
        return running_loss / (i_batch + 1)
        # print(f'Epoch: {self.epoch} \t Loss: {running_loss}')
        # self.writer.flush()

    def predict(self, data=None):

        n_features = self.n_features

        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.batch_sz,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False)
        y_pred = []
        probabilities = []
        latent = []

        with torch.no_grad():
            self.net.eval()

            for test in test_dataloader:
                kmers = test['kmer'].view(-1, 1, n_features).type(dtype)
                outputs, logits = self.net(kmers)
                probs, predicted = torch.max(outputs, 1)

                # Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())
                latent.extend(logits.cpu().tolist())

        return np.array(y_pred), np.array(probabilities), np.array(latent)

    # def evaluate(self):
    #     df = pd.read_csv(self.GT_file, sep='\t')
    #     y_true = df['cluster_id'].to_numpy()
    #     unique_labels = list(np.unique(y_true))
    #     numClasses = len(unique_labels)
    #     y = np.array(list(map(lambda x: unique_labels.index(x), y_true)))
    #
    #     # HUNGARIAN
    #     ind, acc = cluster_acc(y, self.assignments)
    #     d = {}
    #     for i, j in ind:
    #         d[i] = j
    #
    #     for i in range(y.shape[0]):
    #         self.assignments[i] = d[self.assignments[i]]
    #
    #     return self.assignments, self.assignments

    def calculate_probs(self, data=None):

        n_features = self.n_features
        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.batch_sz,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False)

        probabilities = []
        with torch.no_grad():
            self.net.eval()
            for test in test_dataloader:
                kmers = test['kmer'].view(-1, 1, n_features).type(dtype)
                # calculate the prediction by running through the network
                outputs, logits = self.net(kmers)
                probabilities.extend(outputs.cpu().tolist())

        return np.array(probabilities)