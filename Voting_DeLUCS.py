#python Voting_DeLUCS.py FASTA.fas --GT_file=GT.tsv --n_epochs=30 --lambda=1.2 --k=6 --n_clusters=3 --n_mimics=8 --batch_sz=240

from src import models
from src.utils import ProcessFasta
import numpy as np
import pandas as pd
import csv
from src.utils import cluster_acc
import argparse
import torch
from scipy import stats

def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def run(args):

    model = models.IID_model(args)
    model.kmers, model.names, model.lengths, _ = ProcessFasta(args['sequence_file'], args['k'], GT_file=args['GT_file'])
    model.build_dataloaders()
    predictions = []
    accuracies = []
    
    df = pd.read_csv(args['GT_file'], sep='\t')
    y_true = df['cluster_id'].to_numpy()
    unique_labels = list(np.unique(y_true))
    numClasses = len(unique_labels)
    y = np.array(list(map(lambda x: unique_labels.index(x), y_true)))
    
    for i in range(5):
        model.net.apply(weights_init)
        model.epoch = 0
        for i in range(args['n_epochs']):
            loss = model.unsupervised_training_epoch()

            if args['noise'] != 0 and i != 0 and i % args['noise'] == 0:
                    with torch.no_grad():
                        for param in model.net.parameters():
                            param.add_(torch.randn(param.size()).type(torch.FloatTensor) * 0.09)  ## This may casue some troubles in CUDA


        length = len(model.names)
        y_pred, probabilities, latent = model.predict()
        y_pred = y_pred.astype(np.int32)


        # HUNGARIAN
        ind, acc = cluster_acc(y, y_pred)
        d = {}
        for i, j in ind:
            d[i] = j
            
        for i in range(length):
            y_pred[i] = d[y_pred[i]]
        
        predictions.append(y_pred)
        accuracies.append(acc)
        
    predictions = np.array(predictions)
    mode, counts = stats.mode(predictions, axis=0)
    print(accuracies)

    w = np.zeros((numClasses, numClasses), dtype=np.int64)
    for i in range(len(y_true)):
        w[y[i], mode[0][i]] += 1

    print(w)
    print("accuracy: ", np.sum(np.diag(w) / np.sum(w)))
            
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('sequence_file', action='store',type=str)
    parser.add_argument('--n_clusters', action='store',type=int,default=5)
    parser.add_argument('--n_epochs', action='store',type=int,default=100)
    parser.add_argument('--n_mimics', action='store',type=int,default=3)
    parser.add_argument('--batch_sz', action='store',type=int,default=240)
    parser.add_argument('--GT_file', action='store',type=str,default="")
    parser.add_argument('--k', action='store',type=int,default=6)
    parser.add_argument('--model_size', action='store',type=str,default="medium")
    parser.add_argument('--optimizer', action='store',type=str,default="adam")
    parser.add_argument('--lambda', action='store',type=float,default=2.8)
    parser.add_argument('--noise', action='store',type=int,default=0)

    args = vars(parser.parse_args())
    run(args)


if __name__ == '__main__':
    main()