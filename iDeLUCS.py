from src import models
from src.utils import ProcessFasta
import numpy as np
import pandas as pd
import csv
from src.utils import cluster_acc
import argparse
import torch

def run(args):

    model = models.IID_model(args)
    model.kmers, model.names, model.lengths, _ = ProcessFasta(args['sequence_file'], args['k'])
    model.build_dataloaders()
        
    for i in range(args['n_epochs']):
        probabilities = model.calculate_probs()
        loss = model.unsupervised_training_epoch()
        
        if args['noise'] != 0 and i != 0 and i % args['noise'] == 0:
                with torch.no_grad():
                    for param in model.net.parameters():
                        param.add_(torch.randn(param.size()) * 0.09)  ## This may casue some troubles in CUDA


    length = len(model.names)
    y_pred, probabilities, latent = model.predict()
    df = pd.read_csv(args['GT_file'], sep='\t')
    print("Cluster Distribution")
    print(df['cluster_id'].value_counts())
    y_true = df['cluster_id'].to_numpy()
    unique_labels = list(np.unique(y_true))
    numClasses = len(unique_labels)
    y = np.array(list(map(lambda x: unique_labels.index(x), y_true)))
    y_pred = y_pred.astype(np.int32)

    # HUNGARIAN
    ind, acc = cluster_acc(y, y_pred)
    d = {}
    for i, j in ind:
        d[i] = j
    w = np.zeros((numClasses, args['n_clusters']), dtype=np.int64)
    for i in range(length):
        #print(y[i], d[y_pred[i]])
        w[y[i], d[y_pred[i]]] += 1
    print(w)
    print(acc)

    

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('sequence_file', action='store',type=str)
    parser.add_argument('--n_clusters', action='store',type=int,default=5)
    parser.add_argument('--n_epochs', action='store',type=int,default=100)
    parser.add_argument('--n_mimics', action='store',type=int,default=3)
    parser.add_argument('--batch_sz', action='store',type=int,default=240)
    parser.add_argument('--GT_file', action='store',type=str,default=None)
    parser.add_argument('--k', action='store',type=int,default=6)
    parser.add_argument('--model_size', action='store',type=str,default="medium")
    parser.add_argument('--optimizer', action='store',type=str,default="adam")
    parser.add_argument('--lambda', action='store',type=float,default=2.8)
    parser.add_argument('--noise', action='store',type=int,default=0)

    args = vars(parser.parse_args())
    run(args)


if __name__ == '__main__':
    main()