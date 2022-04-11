import sys
import torch
import random
import math
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PytorchUtils import Dummy_Net, myDataset
from torch.utils.data import DataLoader
import time


def compute_joint(x_out, x_tf_out):
    # The function can be found originally in https://github.com/xu-ji/IIC.
    # It produces variable that requires grad (since args require grad)
    
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise

    return p_i_j

def IID_loss(x_out, x_tf_out, p_i_j,  lamb=1.0, EPS=sys.float_info.epsilon):
    # Impolementation of the IID loss function found in the paper:
    # "Invariant Information Clustering for Unsupervised Image 
    # Classification and Segmentation"
    # The function can be found originally in https://github.com/xu-ji/IIC
    _, k = x_out.size()
    #p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = x_out.mean(axis=0).view(k, 1).expand(k, k).clone()
    p_j = x_tf_out.mean(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j)
                      - lamb * torch.log(p_j)
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss

from LossFunctions import IID_loss, InstanceLoss, ClusterLoss
class IID_model():
    def __init__(self, X, y, n_features, n_clusters,verbose=False):
        self.net = Dummy_Net(n_features, n_clusters)
        
        self.epoch = 0
        self.u, self.s = 0, 0
        self.EPS = sys.float_info.epsilon
        self.X = X
        self.y = y
        
        self.net.cuda()
        
        if verbose:
            print(self.net)
            print("Number of Trainable Parameters: ", 
                  sum(p.numel() for p in self.net.parameters() if p.requires_grad))
        
    def build_dataloaders(self, mutate):
        batch_size = 510 #4500  #120   
        batch_sz =  170 #1500 #120 #40

        dtype = torch.cuda.FloatTensor
        train = myDataset(self.X, self.y, transform=None)

        self.index = list(range(len(train)))
        sampler = torch.utils.data.sampler.BatchSampler(self.index,batch_size=batch_sz,drop_last=True)
        train_dataloader = DataLoader(train, batch_sampler=sampler,num_workers=0)
        self.dataloaders = [train_dataloader]

        for d_i in range(3):
            train_mimics = myDataset(self.X, self.y, transform=mutate)
            train_mimics_dataloader = DataLoader(train_mimics,batch_sampler=sampler,num_workers=0)

            self.dataloaders.append(train_mimics_dataloader)

        num_train_batches = len(self.dataloaders[0])
        #print("Length of datasets vector %d" % len(self.dataloaders))
        #print("Number of batches per epoch: %d" % num_train_batches)
        sys.stdout.flush()
                
    def unsupervised_training_epoch(self):
        optimizer = optim.Adam(self.net.parameters(), lr=0.001) #0.001
        dtype = torch.cuda.FloatTensor
        batch_size = 510 #120    
        batch_sz = 170 #40
        l = 3
        
        n_mimics = 3
 
        iterators = (d for d in self.dataloaders)
        running_loss = 0.0
        random.shuffle(self.index)
        
        self.net.train()
        for i_batch, batch in enumerate(zip(*iterators)):

            #Batch is a tuple with the structure: [original, mimic_1, ... , mimic_n]
            #Each of the kmer tensor in the tuple is of size [mini_b_sz, 4**k]
            #So we need to resize the data into a "true batch" to feed into the network

            original_true_batch = torch.zeros(batch_size, 1, 26).type(dtype)
            mimics_true_batch = torch.zeros(batch_size, 1, 26).type(dtype)

            original = (batch[0]['features']).view(-1, 1, 26).type(dtype) #The first in the tuple
            mini_b_sz = original.shape[0]

            self.u = 0.1 * self.u + 0.9 * original.mean(axis=0)
            self.s = 0.1 * self.s + 0.9 * original.std(axis=0)

            for d_i in range(n_mimics):
                mimics = batch[d_i + 1]['features']
                #print(f'Mimics size{mimics.shape}, original size {original.shape}')
                assert mimics.shape[0] == mini_b_sz 

                start = d_i * mini_b_sz
                #original_true_batch[start: start + mini_b_sz, :,:] = original 
                #mimics_true_batch[start: start + mini_b_sz, :,:] = mimics.view(-1, 1, 26).type(dtype) 
                original_true_batch[start: start + mini_b_sz, :,:] = (original - self.u ) / (self.s+self.EPS) 
                mimics_true_batch[start: start + mini_b_sz, :,:] = (mimics.view(-1, 1, 26).type(dtype) - self.u) / (self.s+self.EPS)

            if (mini_b_sz != batch_size // n_mimics) and (self.epoch == 0):
                print(f'Last mini batch size: {mini_b_sz}')

            true_size = mini_b_sz * n_mimics    
            original_true_batch = original_true_batch[:true_size,:,:]
            mimics_true_batch = mimics_true_batch[:true_size,:,:]

            #zero the gradients
            self.net.zero_grad()
            #Forward + Backwards + Optimize

            z1 = self.net(original_true_batch)
            z2 = self.net(mimics_true_batch)
            
            loss = IID_loss(z1, z2, lamb=l)
            #loss = IID_loss(z1, z2, self.joint, lamb=l)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        #running_loss /= i_batch
        self.epoch += 1
        #print(f'Epoch: {self.epoch} \t Loss: {running_loss}')

        
    def supervised_train(self):
        train_dataset = myDataset(self.X, self.y, transform=None)
        optimizer = optim.Adam(self.net.parameters(), lr=0.01)
    
        train_dataloader = DataLoader(train_dataset, batch_size=100)
        
        criterion = torch.nn.CrossEntropyLoss()
        epochs = 50
        
        for epoch in range(epochs):  # This is the number of times we want to iterate over the full dataset
            running_loss = 0.0
            self.net.train()
            
            for i_batch, batch in enumerate(train_dataloader):

                kmers = batch['features'].view(-1, 1, 26).type(torch.FloatTensor)
                label_tensor = batch['labels'].type(torch.LongTensor)
                
                #zero the gradients
                self.net.zero_grad()

                #calculate the prediction by running through the network
                outputs = self.net(kmers)

                #The class with the highest energy is what we choose as prediction
                loss = criterion(outputs, label_tensor)
                loss.backward()
                optimizer.step()

                running_loss += loss

            #running_loss /= i_batch
            #print(f'Epoch: {epoch} \t Loss: {running_loss}')    
            
    def predict(self, data):
        test_dataset = myDataset(data, labels=np.zeros(data.shape[0]), transform=None)
    
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=120)
        
        y_pred = []
        probabilities = []

        with torch.no_grad():
            self.net.eval()
            for test in test_dataloader:

                kmers = test['features'].view(-1, 1, 26).type(torch.cuda.FloatTensor)

                #calculate the prediction by running through the network
                outputs = self.net((kmers  - self.u) / (self.s+self.EPS))
                
                #outputs = self.net(kmers)

                #The class with the highest energy is what we choose as prediction
                probs,  predicted = torch.max(outputs, 1)

                #Extend our list with predictions and groud truth
                y_pred.extend(predicted.cpu().tolist())
                probabilities.extend(probs.cpu().tolist())
                
        return np.array(y_pred), np.array(probabilities)
    

    