# Dependencies
import sys
sys.path.append('src/')

import numpy as np
import random
import time
import pandas as pd

import os
import pickle
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from PytorchUtils import myNet, NetLinear
from LossFunctions import IID_loss, InstanceLoss, ClusterLoss
from utils import byte_iterfasta, FastaEntry, \
                  SequenceDataset, create_dataloaders, \
                  transition_transversion, cluster_acc
from scipy import stats


# Random Seeds for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


def eval_training(net, idx, dataloaders, num_epochs, batch_size, l=1.0, _lr=0.0001, device = 'cuda'):
    """
    :param x_test: features of the new sequences to be tested
    :param y_test: "ground truth" of the sequences-optional.
    :param net: Network to be trained
    :param training_set: Dataset with pairs of CGRs of the form (original, mimic)
    :param l: hyperparameter to favor conditional entropy
    :param _lr: Learning Rate
    :param k: word length in k-mer counts.
    :return: Trained Network.
    """
    # Training parameters:
    optimizer = optim.Adam(net.parameters(), lr=_lr)
    dtype = torch.cuda.FloatTensor
    u, s = 0, 0
    EPS=sys.float_info.epsilon
    n_features = 2079
    net.train()

    # -------------------Training the network----------------------------------
    for epoch in range(num_epochs):
    
        iterators = (d for d in dataloaders)
        running_loss = 0.0
        random.shuffle(idx)


        for i_batch, batch in enumerate(zip(*iterators)):

            #Batch is a tuple with the structure: [original, mimic_1, ... , mimic_n]
            #Each of the kmer tensor in the tuple is of size [mini_b_sz, 4**k]
            #So we need to resize the data into a "true batch" to feed into the network

            original_true_batch = torch.zeros(batch_size, 1, n_features).type(dtype)
            mimics_true_batch = torch.zeros(batch_size, 1, n_features).type(dtype)

            original = (batch[0]['kmer']).view(-1, 1, n_features).type(dtype) #The first in the tuple
            mini_b_sz = original.shape[0]

            u = 0.1 * u + 0.9 * original.mean(axis=0)
            s = 0.1 * s + 0.9 * original.std(axis=0)

            for d_i in range(n_mimics):
                mimics = batch[d_i + 1]['kmer']
                assert mimics.shape[0] == mini_b_sz

                start = d_i * mini_b_sz
                original_true_batch[start: start + mini_b_sz, :,:] = (original - u ) / (s+EPS) 
                mimics_true_batch[start: start + mini_b_sz, :,:] = (mimics.view(-1, 1, n_features).type(dtype) - u) / (s+EPS)

            if (mini_b_sz != batch_size // n_mimics) and (epoch == 0):
                print(f'Last mini batch size: {mini_b_sz}')

            true_size = mini_b_sz * n_mimics    
            original_true_batch = original_true_batch[:true_size,:,:]
            mimics_true_batch = mimics_true_batch[:true_size,:,:]

            #zero the gradients
            net.zero_grad()

            #Forward + Backwards + Optimize

            z1, h1 = net(original_true_batch)
            z2, h2 = net(mimics_true_batch)

            loss = IID_loss(z1, z2, lamb=l)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= i_batch

        if epoch % 30 == 0 and epoch != 0:
            print(f'Epoch: {epoch} \t Loss: {running_loss}')
            with torch.no_grad():
                for param in net.parameters():
                    param.add_(torch.randn(param.size()).type(dtype) * 0.09)


    # ------------------- Testing Process -------------------------------------
    
    test_dataset = SequenceDataset(data_path, k=k, transform=None, GT_file=GT_file)
    
    test_dataloader = DataLoader(test_dataset, 
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)
    y_true = []
    y_pred = []

    with torch.no_grad():
        net.eval()
        for data in test_dataloader:

            kmers = data['kmer'].view(-1, 1, n_features).type(dtype)
            cluster_id = data['cluster_id']

            #calculate the prediction by running through the network
            outputs, _ = net((kmers  - u) / (s+EPS))
            #outputs, _ = net(kmers)

            #The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            #Extend our list with predictions and groud truth
            y_true.extend(cluster_id)
            y_pred.extend(predicted.cpu().tolist())
            
    
    classes = list(np.unique(y_true))
    numClasses = len(classes)
    y_true = list(map(lambda x: classes.index(x), y_true))
    ind, acc = cluster_acc(np.array(y_true), np.array(y_pred))

    d = {}
    for i, j in ind:
        d[i] = j

    for i in range(len(y_true)):  # we do this for each sample or sample batch
        y_pred[i] = d[y_pred[i]]

    w = np.zeros((numClasses, numClasses), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_true[i], y_pred[i]] += 1

    print(f'Network accuracy: {acc}')
    return y_pred, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_file', action='store', type=str, default='None')
    parser.add_argument('--GT_file', action='store', type=str, default='None')
    parser.add_argument('--n_clusters', action='store', type=int, default='None')
    parser.add_argument('--n_mimics', action='store', type=int, default=3)
    parser.add_argument('--batch_sz', action='store', type=int, default=600)
    parser.add_argument('--n_voters', action='store', type=int, default=5)
    parser.add_argument('--n_epochs', action='store', type=int, default=150)
    args = parser.parse_args()

    torch.manual_seed(0)

    # Set value of k
    k = 6
    
    #Data Files
    data_path = args.sequence_file
    GT_file = args.GT_file
    print(data_path)
    # Retrieve the numer and define the mutations
    # You can include the whatever callable transformation you want.
    
    n_mimics = args.n_mimics
    mutations = [] 
    
    for i in range(n_mimics):
        mutations.append(transition_transversion(1e-4, 0.5e-4))  
    
    # Building the data loaders.
    batch_size = args.batch_sz
    assert batch_size % n_mimics == 0, "batch_size must be a multiple of n_mimics"

    idx, dataloaders = create_dataloaders(data_path, mutations, k=k, 
                                          batch_sz=batch_size//n_mimics, 
                                          GT_file=GT_file)
    
    n_features = 2079
    n_clusters = args.n_clusters
    
    
    for i in range(args.n_voters):
        l = 3.5  # 2.8
        _lr = 8.e-5

        # Load and Initialize the Network
        print(f'Loading and Initalizing the Network {i+1} out of {args.n_voters}')
        net = myNet(n_features, n_clusters)
        net.apply(weights_init)
        net = net.cuda()
        print('Start training...')
        prediction, acc = eval_training(net, idx, dataloaders, args.n_epochs, batch_size, l=l, _lr=_lr)
        predictions.append(prediction)
        accuracies.append(acc)

    predictions = np.array(predictions)
    mode, counts = stats.mode(predictions, axis=0)
    # print(mode)
    print(accuracies)

    w = np.zeros((numClasses, numClasses), dtype=np.int64)
    for i in range(y_test.shape[0]):
        w[y_test[i], mode[0][i]] += 1

    print(w)
    print("accuracy: ", np.sum(np.diag(w) / np.sum(w)))

if __name__ == '__main__':
    main()
