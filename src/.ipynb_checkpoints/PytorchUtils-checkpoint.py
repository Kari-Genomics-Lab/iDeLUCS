import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class myNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(myNet, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(n_input, 400),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(400, 128),
                nn.ReLU()
        )
        
        self.instance = nn.Linear(128, 128)  # Always check n_input here.
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128, n_output),  # Always check n_input here.
                nn.Softmax(dim=1)
        )
            
    def forward(self, x):
        x = x.view(-1, x.shape[2])
        x = self.layers(x)
        latent = self.instance(x)
        out = self.classifier(x)
        return out, latent
    
class NetLinear(nn.Module):
    def __init__(self, n_input, n_output):
        super(NetLinear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n_output),  # Always check n_input here.
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, x.shape[2])
        out = self.classifier(x)
        return out

    
class CAMINet(nn.Module):
    def __init__(self, n_tnfs, n_rpks, n_output):
        super(CAMINet, self).__init__()
        self.encode = nn.Sequential(
                nn.Linear(n_tnfs, 360),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(360, 200)
        )
        
        self.classifier = nn.Sequential(
                nn.Linear(200 + n_rpks, 400),
                nn.Dropout(p=0.5),
                nn.Linear(400, n_output)
        )
            
    def forward(self, x, z):
        x = x.view(-1, x.shape[2])
        x = self.layers(x)
        encoding = torch.cat((x, z), 1)
        out = self.classifier(x)
        return out, x    
    


class myDataset(Dataset):
    
    def __init__(self, data, labels=None, transform=None):
        
        if transform: 
            self.data=transform(data)           
        else:
            self.data = data
            
        self.labels = labels

        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return {'features': self.data[idx, :], 'labels': self.labels[idx]}
    
    
class Dummy_Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Dummy_Net, self).__init__()
        self.classifier = nn.Sequential(
                nn.Linear(n_input,64),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(64,128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, n_output),
                nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = x.view(-1, x.shape[2])
        return self.classifier(x)