################ I CHANGED THE LEARNING RATE  ---------------------------
#lr=5e-4

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
import torch.optim as optim

sys.path.append('src/')

from LossFunctions import IID_loss, info_nce_loss
from torch.utils.data import DataLoader
from PytorchUtils import NetLinear, myNet
from ResNet import ConvNet, ResNet18

from utils import SequenceDataset, create_dataloader, \
                  SummaryFasta

# Random Seeds for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

global dtype
global EPS

dtype = torch.FloatTensor
EPS = sys.float_info.epsilon

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.cuda.FloatTensor



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
    def __init__(self, args: dict):

        self.sequence_file = args['sequence_file']
        self.n_clusters =  args['n_clusters']
        self.GT_file = args['GT_file']


        self.k = args['k']
        
        if args['model_size'] == 'linear':
            self.n_features = 4**self.k
            self.net = NetLinear(self.n_features, args['n_clusters'])
            self.reduce = False
            
        elif args['model_size'] == 'small':
            d = {4: 135, 5: 511, 6: 2079}
            self.n_features = d[args['k']]
            self.net = myNet(self.n_features, args['n_clusters'])
            self.reduce = True
            
        elif args['model_size'] == 'full':
            self.net = ResNet18(1, args['n_clusters'])
        else:
            raise ValueError("Invalid Model Type")
        
        self.net.apply(weights_init)
        self.net.to(device)
        self.epoch = 0
        self.EPS = sys.float_info.epsilon
        
        self.n_mimics = args['n_mimics']
        self.batch_sz = args['batch_sz']
        self.optimizer = args['optimizer']
        self.l = args['lambda'] 
        self.lr = args['lr']
        self.weight = args['weight']
        self.schedule = args['scheduler']
        self.mutate = True

        if self.optimizer == 'RMSprop': 
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=0.01, momentum=0.9)
        elif self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not supported")

        print("Number of Trainable Parameters: ", sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        
        if self.schedule == 'Plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        elif self.optimizer == 'Triangle':
            self.schedule = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5, mode="triangular2")

        
        #self.writer = SummaryWriter()
    
        #print(self.net)
        #print("Number of Trainable Parameters: ", 
        #      sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        
    def build_dataloader(self):
        #Data Files
        data_path = self.sequence_file
        GT_file = self.GT_file
                
        #Define the mutations for the data augmentations --> Should we augment the original data too? 
        self.dataloader = create_dataloader(data_path, 
                                             self.n_mimics, 
                                             k=self.k, 
                                             batch_size=self.batch_sz, 
                                             GT_file=GT_file,
                                             reduce=self.reduce)

    def unsupervised_training_epoch(self):
        n_features = self.n_features
        batch_size = self.batch_sz
        k = self.k 
        self.net.train()
        running_loss = 0.0

        for i_batch, sample_batched in enumerate(self.dataloader):
            sample = sample_batched['true'].view(-1, 1, 2 ** k, 2 ** k).type(dtype)
            modified_sample = sample_batched['modified'].view(-1, 1, 2 ** k, 2 ** k).type(dtype)
            
            # zero the gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            z1, h1 = self.net(sample)
            z2, h2 = self.net(modified_sample)

            loss = IID_loss(z1, z2, lamb=self.l) #+ info_nce_loss(h1, h2, 1)
            loss.backward()
            self.optimizer.step()
            running_loss += loss
            
        running_loss /= i_batch
        
        if self.schedule == 'Plateau':
            self.scheduler.step(running_loss)
        elif self.optimizer == 'Triangle':
            self.scheduler.step()

        # if self.epoch % 30 == 0 and self.epoch != 0:
        #     with torch.no_grad():
        #         for param in self.net.parameters():
        #             param.add_(torch.randn(param.size()).type(dtype) * 0.09)

        self.epoch += 1
        #print(f'Epoch: {self.epoch} \t Loss: {running_loss}')
        return running_loss.item()
    
    def contrastive_training_epoch(self):
        n_features = self.n_features
        batch_size = self.batch_sz
        k = self.k 
        self.net.train()
        running_loss = 0.0

        for i_batch, sample_batched in enumerate(self.dataloader):
            sample = sample_batched['true'].view(-1, 1, self.n_features).type(dtype)
            modified_sample = sample_batched['modified'].view(-1, 1, self.n_features).type(dtype)
            
            # zero the gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            z1, h1 = self.net(sample)
            z2, h2 = self.net(modified_sample)

            loss = (1-self.weight)*info_nce_loss(h1, h2, 0.8) + (self.weight)*IID_loss(z1, z2, lamb=self.l)
            loss.backward()
            self.optimizer.step()

            running_loss += loss
            
        running_loss /= i_batch

        if self.schedule == 'Plateau':
            self.scheduler.step(running_loss)
        elif self.optimizer == 'Triangle':
            self.scheduler.step()
        self.epoch += 1

        return running_loss.item()

    def predict(self, data=None):
        
        n_features = self.n_features
        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file, reduce=self.reduce)
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
                kmers = test['kmer'].view(-1, 1, self.n_features).type(dtype)
                outputs, logits = self.net(kmers)
                probs,  predicted = torch.max(outputs, 1)

                #Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())
                latent.extend(logits.cpu().tolist())
                
        return np.array(y_pred), np.array(probabilities), np.array(latent) 

    def calculate_probs(self, data=None):
        
        n_features = self.n_features
        test_dataset = SequenceDataset(self.sequence_file, k=self.k, transform=None, GT_file=self.GT_file, reduce=self.reduce)
        test_dataloader = DataLoader(test_dataset, 
                             batch_size=self.batch_sz,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)
 
        probabilities = []
        with torch.no_grad():
            self.net.eval()
            for test in test_dataloader:
                #kmers = test['kmer'].view(-1, 1, 2** self.k, 2 ** self.k).type(dtype)
                kmers = test['kmer'].view(-1, 1, n_features).type(dtype)

                #calculate the prediction by running through the network
                outputs, logits = self.net(kmers)
                probabilities.extend(outputs.cpu().tolist())

        return np.array(probabilities)