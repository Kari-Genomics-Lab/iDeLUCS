import pandas as pd
import os
from idelucs import models
import numpy as np
import matplotlib.pyplot as plt

from idelucs.utils import SummaryFasta, plot_confusion_matrix, \
                      label_features, compute_results, generate_csv_mapping

import argparse
import torch
import sys
import time
from datetime import datetime
from resource import *
import csv
import tracemalloc

def save_results_in_file(dataset_name, model_name, model_parameters, results,file_name):
    # Create the file if it does not exitst and write first file with column names
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if file.tell() == 0:
            writer.writerow(['Dataset', 'Model', 'Parameters'] + list(results.keys()))

        # Write the new row
        row = [dataset_name, model_name, model_parameters] + list(results.values())
        writer.writerow(row)
        


def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif hasattr(m, 'reset_parameters'):
        m.reset_parameters()          

def run(args):
    # starting memory monitoring
    tracemalloc.start()
    
    
    now = time.asctime()
    time_stamp = now.split(' ')
    hour = now.split(' ')[3]
    time_stamp[3] = '-'.join(hour.split(':'))
    time_stamp = '_'.join(time_stamp[1:4])

    folder_name = os.getcwd()
    folder_name = f"{folder_name}/Results"

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    if not os.path.isdir(f'{folder_name}/{time_stamp}'):
        os.mkdir(f'{folder_name}/{time_stamp}')

    model = models.IID_model(args)
    model.names, model.lengths, model.GT, model.cluster_dis = SummaryFasta(model.sequence_file,
                                                                           model.GT_file)
    print(model.cluster_dis)
    stats = {"n_seq": len(model.lengths),
            "min_len": np.min(model.lengths),
            "max_len": np.max(model.lengths),
            "avg_len": np.mean(model.lengths)}

    print( f'No. Sequences: \t {stats["n_seq"]:,}')
    print(f'Min. Length: \t {stats["min_len"]:,}')
    print(f'Max. Length: \t {stats["max_len"]:,}')
    print(f'Avg. Length: \t {round(stats["avg_len"],2):,}')
    model.build_dataloader()
    predictions = []
    
    training_figure, display_training = plt.subplots(nrows=1, ncols=1)
    display_training.grid(True)
    display_training.set_title("Learning Curves")
    display_training.set_xlabel("Epoch")
    display_training.set_ylabel("Training Loss")

    start_time = datetime.now()
    for voter in range(args['n_voters']):
        sys.stdout.write(f"\r........... Training Model ({voter+1}/{ args['n_voters']})................")
        sys.stdout.flush()
        model.net.apply(weights_init)
        model.epoch = 0
        model_loss = []
        for i in range(args['n_epochs']):
            loss = model.contrastive_training_epoch()
            # loss = 10
            print(f"Epoch: {i}, Loss: {loss}, n_voter: {voter}")
            model_loss.append(loss)

        length = len(model.names)
        y_pred, probabilities, latent = model.predict()
        y_pred = y_pred.astype(np.int32)
        
        d = {}
        count = 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] in d:
                y_pred[i] = d[y_pred[i]]
            else:
                d[y_pred[i]] = count
                y_pred[i] = count
                count += 1
        predictions.append(y_pred)

        display_training.plot(model_loss, label=f'Model {voter+1}')
        display_training.axes.legend(loc=1)
        training_figure.savefig(f'{folder_name}/{time_stamp}/training_plots.jpg')
        
        print("Memory Usage:", tracemalloc.get_traced_memory())

    y_pred, probabilities = label_features(np.array(predictions), args['n_clusters'])
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("Training took:", time_taken)
    print("Memory Usage Summary:", tracemalloc.get_traced_memory())
    
    
    #--------------------- Computing and Saving the Results
    sys.stdout.write(f"\r........... Computing Results ................")
    sys.stdout.flush()
    
    if args['GT_file'] != None:
        unique_labels = list(np.unique(model.GT))
        numClasses = len(unique_labels)
        y = np.array(list(map(lambda x: unique_labels.index(x), model.GT)))
        results, ind = compute_results(y_pred, latent, y)

        d = {}
        for i, j in ind:
            d[i] = j
        w = np.zeros((numClasses, args['n_clusters']), dtype=np.int64)
        
        clustered = np.zeros_like(y, dtype=bool)
        for i in range(length):
            w[y[i], d[y_pred[i]]] += 1

            if y[i] == d[y_pred[i]]:
                clustered[i] = True

        fig, new_ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        if len(unique_labels) < 16:
            plot_confusion_matrix(w, unique_labels, ax=new_ax, normalize=False)
            fig.savefig(f'{folder_name}/{time_stamp}/contingency_matrix.jpg')
        else:
            # generating a csv mapping
            generate_csv_mapping(w, unique_labels)
    else:
        results, ind = compute_results(y_pred, latent)
        clustered = (probabilities >= 0.9)
       
    results["time taken"] = time_taken
    results["memory usage"] = tracemalloc.get_traced_memory()
    
    sys.stdout.write(f"\r........ Saving Results ..............")
    sys.stdout.flush()
    
    dataset_name = args['sequence_file'].split('/')[-1]
    save_results_in_file(dataset_name, "iDeLUCS", args, results, "ALL_RESULTS.tsv")

    names = np.array(model.names)
    data = np.concatenate((names[:,np.newaxis],
                              y_pred[:,np.newaxis],
                              probabilities[:,np.newaxis]), axis=1)

    df = pd.DataFrame(data, columns=['sequence_id','assignment','confidence_score'])
    df.to_csv(f'{folder_name}/{time_stamp}/assignments.tsv',sep='\t')

    df=pd.Series(results, name='Value')
    df.to_csv(f'{folder_name}/{time_stamp}/metrics.tsv',sep='\t')

    # ------------------------------- Computing and Saving the Representations
    
    sys.stdout.write(f"\r........... Computing Representations................")
    sys.stdout.flush()

    import umap
    embedding = umap.UMAP(random_state=42).fit_transform(latent)
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    ax.set_title("Representation of the Latent Space")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    ax.scatter(embedding[~clustered, 0],
                embedding[~clustered, 1],
                c = np.atleast_2d([0.5, 0.5, 0.5]),
                s=1,
                alpha=0.5)
    ax.scatter(embedding[clustered, 0],
               embedding[clustered, 1],
               c=y_pred[clustered],
               s=1,
               cmap='Spectral')
    plt.show()
    fig.savefig(f'{folder_name}/{time_stamp}/learned_representation.jpg')
    tracemalloc.stop()
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('--sequence_file', action='store',type=str)
    parser.add_argument('--n_clusters', action='store',type=int,default=5)
    parser.add_argument('--n_epochs', action='store',type=int,default=100)
    parser.add_argument('--n_mimics', action='store',type=int,default=3)
    parser.add_argument('--batch_sz', action='store',type=int,default=256)
    parser.add_argument('--GT_file', action='store',type=str,default=None)
    parser.add_argument('--k', action='store',type=int,default=6)
    parser.add_argument('--optimizer', action='store',type=str,default="RMSprop")
    parser.add_argument('--scheduler', action='store',type=str,default="None")
    parser.add_argument('--weight', action='store',type=float,default=0.25)
    parser.add_argument('--lambda', action='store',type=float,default=2.8)
    parser.add_argument('--lr', action='store',type=float,default=1e-3)
    parser.add_argument('--n_voters', action='store',type=int, default=5)
    parser.add_argument('--model_size', action='store',type=str,default="linear")

    args = vars(parser.parse_args())

    print("\nTraining Parameters:")
    for key in args:
        print(f'{key} \t -> {args[key]}')
    
    run(args)

if __name__ == '__main__':
    main()