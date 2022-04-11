import pickle
import argparse
import torch.optim as optim 
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from PytorchUtils import myDataset, myNet
import time
from collections import Counter



def test(net, x_test, y_test):
              
    net.eval()
    y_true = []
    predicted = []
    for i in range(x_test.shape[0]):
        sample = torch.from_numpy(x_test[i])
        label = y_test[i]
        sample = sample.view(1,1,x_test.shape[1]).type(torch.cuda.FloatTensor)
        with torch.no_grad():
          output = net(sample)

        top_n, top_i = output.topk(1) #Get Label from prediction
        predicted.append(top_i[0].item())
        y_true.append(label)
              
    n = max(y_true) + 1
    w = np.zeros((n, n), dtype=np.int32)

    for i in range(len(predicted)):
        w[y_true[i], predicted[i]] += 1
        
    return np.sum(np.diag(w))/np.sum(w)
    
def train(net, x, y , n_features):
    
    # creating the dataset.
    training_set =  myDataset(x, y)

    #Training parameters 
    batch_size = 512
    epochs = 200
    
    #Define the imbalanced weigths and the Loss Function
    c = Counter(y)
    _max = c.most_common(1)[0][1]
    w = torch.tensor([_max/x for x in c.values()]).type(torch.cuda.FloatTensor)
    criterion = nn.CrossEntropyLoss(weight = w)
    
    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        
        for i_batch, sample_batched in enumerate(dataloader):
            
            sample = sample_batched['features'].view(-1,1,n_features).type(torch.cuda.FloatTensor)
            label_tensor = sample_batched['labels'].type(torch.cuda.LongTensor)
            
            #print(label_tensor.shape, label_tensor)
            
            #zero the gradients
            optimizer.zero_grad()
            
            #forward + backward + optimize
            output = net(sample)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            
            running_loss += loss
        running_loss /= i_batch
        print(f'Epoch: {epoch} \t Loss: {running_loss} \t Accuracy: {test(net, x, y)}')
    
                
              
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store', type=str, default='None')
    args = parser.parse_args()

    TrainDataFile = args.data_path
    
    # Load Training Data.
    #x_pairs, x, y = pickle.load(open(TrainDataFile, "rb"))
    x, y = pickle.load(open(TrainDataFile, "rb"))
              
    n_features = x.shape[1]
    unique_labels = sorted(set(y))
    numClasses = len(unique_labels)

    print("The size of the training array is:", x.shape)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    

    # Building the network.
    numClasses = len(unique_labels)
    net = myNet(n_features, numClasses)

    net = net.cuda()
    
    print(net)
    print("Number of Trainable Parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # Training Process.
    train(net, x, y, n_features)

    # Testing Process.
    #test(net, x, y)

if __name__ == '__main__':
    main()
