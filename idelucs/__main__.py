#!/usr/bin/env python3

import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


from idelucs.utils import SummaryFasta, plot_confusion_matrix, \
                      label_features, compute_results

from idelucs import models

import argparse
import torch
import sys
import time

import csv
#-------------------------------
import hdbscan

def weights_init(m):
    """
    Kaiming initialization of the weights
    :param m: Layer
    :return:
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
        
def save_results_in_file(dataset_name, model_name, model_parameters, results, time, memory, file_name):
    # Create the file if it does not exitst and write first file with column names
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if file.tell() == 0:
            writer.writerow(['Dataset', 'Model', 'Parameters'] + list(results.keys()) + ['Time', 'Memory'])

        # Write the new row
        row = [dataset_name, model_name, model_parameters] + list(results.values()) + [time, memory]
        writer.writerow(row)


def run(args):

    start_time = time.time()
    now = time.asctime()
    time_stamp = now.split(' ')
    hour = now.split(' ')[3]
    time_stamp[3] = '-'.join(hour.split(':'))
    
    time_stamp = '_'.join(time_stamp[1:-1])
    
    folder_name = os.getcwd()
    folder_name = f'{folder_name}/Results'
    
    use_hdbscan = False

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    results_folder = f'{folder_name}/{(args["sequence_file"]).split("/")[-1].split(".")[0]}'
    
    if not os.path.isdir(f'{results_folder}'):
        os.mkdir(f'{results_folder}')
    
    os.mkdir(f'{results_folder}/{time_stamp}')

    if args['n_clusters'] == 0:
        args['n_clusters'], use_hdbscan = 200, True
        
    model = models.IID_model(args)
    model.names, model.lengths, model.GT, model.cluster_dis = SummaryFasta(model.sequence_file,
                                                                           model.GT_file)
    
    if use_hdbscan:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=len(model.names)//100 + 1, gen_min_span_tree=True, prediction_data=True)
    
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
    min_loss = np.inf
    
    training_figure, display_training = plt.subplots(nrows=1, ncols=1)
    display_training.grid(True)
    display_training.set_title("Learning Curves")
    display_training.set_xlabel("Epoch")
    display_training.set_ylabel("Training Loss")


    for voter in range(args['n_voters']):
        sys.stdout.write(f"\r........... Training Model ({voter+1}/{ args['n_voters']})................")
        sys.stdout.flush()
        model.net.apply(weights_init)
        model.epoch = 0
        model_loss = []
        model_min_loss=np.inf

        for i in range(args['n_epochs']):
            loss = model.contrastive_training_epoch()
            model_min_loss = min(model_min_loss, loss)
            model_loss.append(loss)

        length = len(model.names)
        y_pred, probabilities, latent = model.predict()

        if model_min_loss < min_loss:
            model_latent = latent
        
        if not use_hdbscan:
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
        training_figure.savefig(f'{results_folder}/{time_stamp}/training_plots.jpg')
                                                                

    if not use_hdbscan:
        y_pred, probabilities = label_features(np.array(predictions), args['n_clusters'])
        
    else:
        clusterer.fit(latent )
        y_pred = clusterer.labels_ + 1
        probabilities = clusterer.probabilities_
        args['n_clusters'] = np.max(y_pred) + 1
        

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
        
        if -1 in y_pred:
            d[-1] = 0
        
        w = np.zeros((numClasses, max(max(y_pred) + 1, max(y) + 1)), dtype=np.int64)
        clustered = np.zeros_like(y, dtype=bool)
        for i in range(y.shape[0]):
            w[y[i], d[y_pred[i]]] += 1

            if y[i] == d[y_pred[i]]:
                clustered[i] = True
         
        if args['n_clusters'] < 16:
            fig, new_ax = plt.subplots( nrows=1, ncols=1)  # create figure & 1 axis
            plot_confusion_matrix(w, unique_labels, ax=new_ax, normalize=False)
            fig.savefig(f'{results_folder}/{time_stamp}/contingency_matrix.jpg')
        else:
            # generating a csv mapping
            #generate_csv_mapping(w, unique_labels)
            w_df = pd.DataFrame(w)
            w_df.index = unique_labels
            w_df.to_csv(f'{results_folder}/{time_stamp}/contingency_matrix.tsv',sep='\t')

    else:
        results, ind = compute_results(y_pred, latent)
        #clustered = (probabilities >= 0.9)
        clustered = (probabilities >= 0.0)
    
    print(results)
    sys.stdout.write(f"\r........ Saving Results ..............")
    sys.stdout.flush()
    
    
    
    dataset_name = args['sequence_file'].split('/')[-1]
    del args['sequence_file']
    del args['GT_file']
    

    _time = (time.time() - start_time) #running_info.ru_utime + running_info.ru_stime
    hour = _time // 3600
    minutes = (_time  - (3600 * hour)) // 60
    seconds = _time - (hour * 3600) - (minutes * 60)
    memory = "" #(running_info.ru_maxrss/1e6)
    print("") #running_info)
    
    names = np.array(model.names)
    data = np.concatenate((names[:,np.newaxis],
                              y_pred[:,np.newaxis],
                              probabilities[:,np.newaxis]), axis=1)

    df = pd.DataFrame(data, columns=['sequence_id','assignment','confidence_score'])
    df.to_csv(f'{results_folder}/{time_stamp}/assignments.tsv',sep='\t')

    df=pd.Series(results, name='Value')
    df.to_csv(f'{results_folder}/{time_stamp}/metrics.tsv',sep='\t')
    
    save_results_in_file(dataset_name, "iDeLUCS", args, results, 
                         f'{hour}:{minutes}:{seconds}', 
                         memory, f'{os.getcwd()}/ALL_RESULTS.tsv')

    # ------------------------------- Computing and Saving the Representations
    if args['plot']:
        sys.stdout.write(f"\r........... Computing Representations................")
        sys.stdout.flush()

        import umap
        n_samples = len(model.names)
        mask = np.full(n_samples, True)
        MAX_SAMPLES = 1000
        if n_samples > MAX_SAMPLES :
            mask = np.full(n_samples, False)
            mask[:MAX_SAMPLES] = True
            np.random.shuffle(mask)

        embedding = umap.UMAP(random_state=42).fit_transform(latent)
        fig, ax = plt.subplots(nrows=1, ncols=1) 
        ax.set_title("Representation of the Latent Space")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        ax.scatter(embedding[mask & ~clustered, 0],
                   embedding[mask & ~clustered, 1],
                   c = np.atleast_2d([0.5, 0.5, 0.5]),
                   s=1,
                   alpha=0.5)
        ax.scatter(embedding[mask & clustered, 0],
                  embedding[mask & clustered, 1],
                  c=y[mask & clustered],  #y_pred[mask & clustered]
                  s=1,
                  cmap='Spectral')
        fig.savefig(f'{results_folder}/{time_stamp}/learned_representation.jpg', dpi=150)
    
def main():
    parser= argparse.ArgumentParser()
    parser.add_argument('--sequence_file', action='store',type=str)
    parser.add_argument('--n_clusters', action='store',type=int,default=0,
                        help='Expected or maximum number of clusters to find. \n'
                            'It should be equal or greater than n_true_clusters \n'
                            'when GT is provided. \n'
                            'NOTE: Use 0 for automatically finding fine-grained \n'
                            'clusters')
    parser.add_argument('--n_epochs', action='store',type=int,default=100,
                        help='Number of training epochs. An epoch is defined \n' 
                             'as a training iteration over all the training pairs.')
    parser.add_argument('--n_mimics', action='store',type=int,default=3, 
                        help='Number of data augmentations per sequence \n' 
                            'that will be considered during training.')
    parser.add_argument('--batch_sz', action='store',type=int,default=256)
    parser.add_argument('--GT_file', action='store',type=str,default=None)
    parser.add_argument('--k', action='store',type=int,default=6, help="k-mer length")
    parser.add_argument('--optimizer', action='store',type=str,default="RMSprop")
    parser.add_argument('--scheduler', action='store',type=str,default="None")

    parser.add_argument('--weight', action='store',type=float,default=0.25,
                        help='Relative importance of the contrastive objective on \n'
                            'the final loss. Use a higher value when low intra- \n'
                            'cluster distance is expected and a lower value when \n'
                            'high intra-cluster variability is expected')
    
    parser.add_argument('--lambda', action='store',type=float,default=2.8,
                        help='Hyperparameter to control cluster balance. \n'
                            'Use lambda: 1.2 when unbalanced clusters are expected \n'
                            'Use lambda: 2.8 when perfectly balanced clusters are expected \n')
    parser.add_argument('--lr', action='store',type=float,default=1e-3, help="Learning Rate")
    parser.add_argument('--n_voters', action='store',type=int, default=5, help="Number of Voters")
    parser.add_argument('--model_size', action='store',type=str,default="linear", help="Selection of 'conv', 'small', 'linear' or 'full'")
    parser.add_argument('--plot', action='store',type=bool, default=False, help="Set to True to plot the final output representation")
    
    args = vars(parser.parse_args())

    print("\nTraining Parameters:")
    for key in args:
        print(f'{key} \t -> {args[key]}')
    
    run(args)


if __name__ == '__main__':
    main()
