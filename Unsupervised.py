#python Voting_DeLUCS.py FASTA.fas --GT_file=GT.tsv --n_epochs=30 --lambda=1.2 --k=6 --n_clusters=3 --n_mimics=8 --batch_sz=240

import sys
import pandas as pd
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import mixture
from sklearn.cluster import KMeans

from src.utils import SummaryFasta, compute_results, kmersFasta
import argparse
import time


def build_pipeline(numClasses, method):
    normalizers = []
    if method == 'GMM':
        normalizers = [('classifier', mixture.GaussianMixture(n_components=numClasses))]
    if method == 'k-means++':
        normalizers.append(('classifier', KMeans(n_clusters=numClasses, init='k-means++', random_state=321)))
    return Pipeline(normalizers)


def run(args):
    sys.stdout.write(f"\r........... Parsing Fasta File ................")
    sys.stdout.flush()
    names, lengths, GT, cluster_dis = SummaryFasta(args['sequence_file'],args['GT_file'])
    sys.stdout.write(f"\r........... Computing k-mers ................")
    sys.stdout.flush()
    _, x_train = kmersFasta(args['sequence_file'], k=args['k'])

    
    unique_labels = list(np.unique(GT))
    numClasses = len(unique_labels)
    y = np.array(list(map(lambda x: unique_labels.index(x), GT)))

    RESULTS = {}
    
    for i in range(10):
        sys.stdout.write(f"\r........... Fitting model {i+1}/10 ................")
        sys.stdout.flush()
        pipeline = build_pipeline(numClasses, args['method'])
        pipeline.fit(x_train)
        y_pred = pipeline.predict(x_train)
        #print("Cluster Accuracy")
        results, ind = compute_results(y_pred, x_train, y)
        for metric in results:
            if metric in RESULTS:
                RESULTS[metric] += results[metric]
            else:
                RESULTS[metric] = results[metric]
        
    for metric in  RESULTS:
        RESULTS[metric] /= 10
    
    print("")
    print(RESULTS) 
    
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('--sequence_file', action='store',type=str)
    parser.add_argument('--n_clusters', action='store',type=int,default=5)
    parser.add_argument('--GT_file', action='store',type=str,default=None)
    parser.add_argument('--k', action='store',type=int,default=6)
    parser.add_argument('--method', action='store',type=str,default='k-means++')

    args = vars(parser.parse_args())

    print("\nTraining Parameters:")
    for key in args:
        print(f'{key} \t -> {args[key]}')
    
    run(args)


if __name__ == '__main__':
    main()