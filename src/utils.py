import sys
sys.path.append('src/')
import pyximport 
pyximport.install()

from kmers import kmer_counts, cgr

import random, itertools
from datetime import datetime
import os as os
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
import sklearn.metrics.cluster as metrics
from sklearn.cluster import KMeans
from collections import namedtuple

import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

def check_sequence(header, seq):
    """
    Adapted from VAMB: https://github.com/RasmussenLab/vamb
    
    Check that there're no invalid characters or bad format
    in the file. 
    
    Note: The GAPS ('-') that are introduced from alignment
    are considered valid characters. 
    """

    if len(header) > 0 and (header[0] in ('>', '#') or header[0].isspace()):
        raise ValueError('Bad character in sequence header')
    if '\t' in header:
        raise ValueError('tab included in header')

    basemask = bytearray.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV-',
                                   b'ACGTTTNNNNNNNNNNNNNNNNNNNNNN')

    masked = seq.translate(basemask, b' \t\n\r')
    stripped = masked.translate(None, b'ACGTN')
    if len(stripped) > 0:
        bad_character = chr(stripped[0])
        msg = "Invalid DNA byte in sequence {}: '{}'"
        raise ValueError(msg.format(header, bad_character))
    
class Random_N(object):
    """
    Mutate Genomic sequence using transitions only.
    :param seq: Original Genomic Sequence.
    :param threshold: probability of Transition.
    :return: Mutated Sequence.
    """

    def __init__(self, n_bp):
        self.n_bp = n_bp
        
    def __call__(self, seq):
    
        N = ord('N')

        index = np.random.randint(0, len(seq), self.n_bp)
        for i in index:
            seq[i] = N

    
class transition(object):
    """
    Mutate Genomic sequence using transitions only.
    :param seq: Original Genomic Sequence.
    :param threshold: probability of Transition.
    :return: Mutated Sequence.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, seq):
    
        A, C, G, T, N = ord('A'), ord('C'), ord('G'), ord('T'), ord('N')

        x = np.random.random(len(seq))
        index = np.where(x < self.threshold)[0]
        #print(index)
        mutations = {A:G, G:A, T:C, C:T, N:N}

        for i in index:
            #print(chr(seq[i]), '->', chr(mutations[seq[i]]))
            seq[i] = mutations.get(seq[i], N)




class transversion(object):
    """
    Mutate Genomic sequence (in binary) using transversions only.
    :param seq: Original Genomic Sequence.
    :param threshold: Probability of Transversion.
    :return: Mutated Sequence.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, seq):
        A, C, G, T, N = ord('A'), ord('C'), ord('G'), ord('T'), ord('N')

        x = np.random.random(len(seq))
        index = np.where(x < self.threshold)[0]
        #print(index)

        mutations = {A:[T,C],  G:[T,C], T:[A,G], C:[A,G], N:[N]}

        for i in index:
            #print(chr(seq[i]), '->', chr(random.choice(mutations[seq[i]])))
            seq[i] = random.choice(mutations.get(seq[i], [N]))            
            
class composition(object):
    """
    Mutate Genomic sequence (in binary) using both transitions 
    and transversions.
    :param seq: Original Genomic Sequence.
    :param threshold: Probability of Transversion.
    :return: Mutated Sequence.
    """
    def __init__(self, threshold_1, threshold_2, n):
        self.tf1 = transition(threshold_1)
        self.tf2 = transversion(threshold_2)
        self.tf3 = Random_N(n)
        
    def __call__(self, seq):

        self.tf1(seq)
        self.tf2(seq)
        self.tf3(seq)
    
def SummaryFasta(fname, GT_file=None):
    lines = list()
    seq_id = ""
    names, lengths = [], []
    ground_truth = None
    cluster_dis = None

    if GT_file: 
        ground_truth = []
        df = pd.read_csv(GT_file, sep='\t')
        GT_dict = dict(zip(df.sequence_id, df.cluster_id))
        cluster_dis = df['cluster_id'].value_counts().to_dict()

    for line in open(fname, "rb"):

        if line.startswith(b'#'):
            pass

        elif line.startswith(b'>'):
            if seq_id != "":
                seq = bytearray().join(lines)

                if (GT_file and not seq_id in GT_dict):
                    raise ValueError('Check GT for sequence {}'.format(seq_id))

                check_sequence(seq_id, seq)
                names.append(seq_id)  #seq_id
                lengths.append(len(seq))

                if GT_file:
                    ground_truth.append(GT_dict[seq_id])

                lines = []
                seq_id = line[1:-1].decode()  # Modify this according to your labels. 
            
            seq_id = line[1:-1].decode()
        
        else:
            lines += [line.strip()]
  
    if (GT_file and not seq_id in GT_dict):
        raise ValueError('Check GT for sequence {}'.format(seq_id))

    seq = bytearray().join(lines)
    check_sequence(seq_id, seq)
    names.append(seq_id[1:-1]) #seq_id[1:-1]
    lengths.append(len(seq))

    if GT_file:
        ground_truth.append(GT_dict[seq_id])

    return names, lengths, ground_truth, cluster_dis


def kmersFasta(fname, k=6, transform=None, reduce=False):
    lines = list()
    seq_id = ""
    names, kmers = [], []

    for line in open(fname, "rb"):
        if line.startswith(b'#'):
            pass

        elif line.startswith(b'>'):
            if seq_id != "":
                seq = bytearray().join(lines)
                names.append(seq_id)  #seq_id
                            
                if transform:
                    transform(seq)
                    
                counts = np.ones(4**k, dtype=np.int32)
                kmer_counts(seq, k, counts)
                #cgr(seq, k, counts)
                kmers.append(counts / np.sum(counts))
                
                lines = []
                seq_id = line[1:-1].decode()  # Modify this according to your labels.  
            seq_id = line[1:-1].decode()
        
        else:
            lines += [line.strip()]
  
    seq = bytearray().join(lines)
    names.append(seq_id[1:-1]) #seq_id[1:-1]
    if transform:
        transform(seq)
        
    counts = np.ones(4**k, dtype=np.int32)
    kmer_counts(seq, k, counts)
    #cgr(seq, k, counts)
    kmers.append(counts / np.sum(counts))
    
    if reduce:
        K_file = np.load(open(f'kernels/kernel{k}.npz','rb'))
        KERNEL = K_file['arr_0']
        return names, np.dot(np.array(kmers), KERNEL)
        
    return names, np.array(kmers)
 
import time 
def AugmentFasta(sequence_file, n_mimics, k=6, reduce=False):

    train_features = []
    start = time.time()
    TRANSFORM = composition(1e-3, 0.5e-3, 20)

    # Compute Features and save original data for testing.
    #sys.stdout.write(f'\r............computing augmentations (0/{n_mimics})................')
    #sys.stdout.flush()
    _, t_norm = kmersFasta(sequence_file, k=k, transform=TRANSFORM, reduce=reduce)
    t_norm.resize(t_norm.shape[0],1,t_norm.shape[1])
    
    
    #sys.stdout.write(f'\r............computing augmentations (1/{n_mimics})................')
    #sys.stdout.flush()
    _, t_mutated = kmersFasta(sequence_file, k=k, transform=TRANSFORM, reduce=reduce)
    t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
    train_features.extend(np.concatenate((t_norm, t_mutated), axis=1))
    
    #sys.stdout.write(f'\r............computing augmentations (2/{n_mimics})................')
    #sys.stdout.flush()
    _, t_mutated = kmersFasta(sequence_file, k=k, transform=TRANSFORM, reduce=reduce)
    t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
    train_features.extend(np.concatenate((t_norm, t_mutated), axis=1))
    
    for j in range(n_mimics-2):
        #sys.stdout.write(f'\r............computing augmentations ({3+j}/{n_mimics})................')
        #sys.stdout.flush()
        _, t_mutated = kmersFasta(sequence_file, k=k, transform=TRANSFORM, reduce=reduce)
        t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
        train_features.extend(np.concatenate((t_norm, t_mutated), axis=1)) 

    #sys.stdout.write(f'\r............computing augmentations ({j+1}/{n_mimics})................')
    #sys.stdout.flush()
    #_, t_mutated = kmersFasta(sequence_file, k=k, transform=Random_N(10))
    #t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
    #train_features.extend(np.concatenate((t_norm, t_mutated), axis=1))      

    x_train = np.array(train_features).astype('float32')
    x_test = np.reshape(t_norm, (-1, t_norm.shape[-1])).astype('float32')
    #print("\n Elapsed Time:", time.time()-start)

    # scaling the data.
    scaler = StandardScaler()
    scaler.fit(x_test)

    x_train_1 = scaler.transform(x_train[:, 0, :])
    x_train_2 = scaler.transform(x_train[:, 1, :])
    x_test = scaler.transform(x_test)

    x_train[:, 0, :] = x_train_1
    x_train[:, 1, :] = x_train_2
    
    return x_train

class AugmentedDataset(Dataset):
    """ 
    Dataset creation directly from fasta file.
    """

    def __init__(self, data):
        """
        : data: (numpy array) of size [:,2,:]
        """
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'true': self.data[idx, 0, :], 'modified': self.data[idx, 1, :]}  #<--- We can enforce the prediction of same vector
        return sample

class SequenceDataset(Dataset):
    """ Dataset creation directly from fasta file"""
    
    def __init__(self, fasta_file, k=6, transform=None, GT_file=None, reduce=False):
        """ Args:
            fasta_file (string): Path to te fasta file
            transform (callable, optional): Optional transform to be applied on a 
                                            sequence. Function computing the mimics 
        """
        self.names, self.lengths, self.GT, self.cluster_dis = SummaryFasta(fasta_file, GT_file)
        _, self.kmers = kmersFasta(fasta_file, k, transform, reduce=reduce)

        # scaling the data.
        scaler = StandardScaler()
        self.kmers = scaler.fit_transform(self.kmers)

        
    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.GT:           
            sample = {'kmer':self.kmers[idx,:],'name':self.names[idx], 'cluster_id':self.GT[idx]}
        else:
            sample = {'kmer':self.kmers[idx,:],'name':self.names[idx]}
            
        return sample

def create_dataloader(sequence_file, n_mimics, k=6, batch_size=512, GT_file=None, reduce=False):

    train_data = AugmentFasta(sequence_file, n_mimics, k=k, reduce=reduce)
    training_set = AugmentedDataset(train_data)
    return DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)


#-------------------- The following couple of functions are for plotting and saving the point cloud during training-----

def get_coord(probs, num_classes):
    # computes coordinate for 1 sample based on probability distribution over c
  
    coords_total = np.zeros(2, dtype=np.float32)
    probs_sum = probs.sum()

    fst_angle = 0.

    for c in range(num_classes):
        # compute x, y coordinates
        coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) + fst_angle
        coords[0] = np.sin(coords[0])
        coords[1] = np.cos(coords[1])
        coords_total += (probs[c] / probs_sum) * coords
    return coords_total
  
def PlotPolygon(predictions, num_classes, ax, title):
    hues = np.linspace(0.0, 1.0, num_classes + 1)[0:-1]  # ignore last one
    all_colours = [list((np.array(hsv_to_rgb(hue, 0.8, 0.8)) * 255.)
                   .astype(np.uint8)) for hue in hues]
    
    coordinates = []
    colors = []
    
    #add the points of perfect classification:
    for c in range(num_classes): 
         # compute x, y coordinates
         coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) 
         coords[0] = np.sin(coords[0])
         coords[1] = np.cos(coords[1])

         #colour = (np.array(all_colours[0][c])).astype(np.uint8)
         colour = (np.zeros(3).astype(np.uint8)) 
         coordinates.append(coords)
         colors.append(colour/255)

    for i in range(predictions.shape[0]):

        coord = get_coord(predictions[i, :], num_classes)
        render_c = np.argmax(predictions[i,:])
        #print(predictions[i, :], render_c)
        #render_c = y_true[i]
        colour = (np.array(all_colours[render_c])).astype(np.uint8)
        coordinates.append(coord)
        colors.append(colour/255)

    coordinates = np.array(coordinates)
    colors = np.array(colors)
    ax.scatter(coordinates[:,0], coordinates[:,1], c = colors, marker = '.', linewidths=0.5)

    ax.axis('equal')
    ax.axis('off')
    ax.set_title('Training Progress (%s)' %(title))
    # plt.savefig(path)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    :param y_true: true labels, numpy.array with shape `(n_samples,)`
    :param y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    :return:  accuracy, in [0,1]
    """

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return ind, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot_confusion_matrix(cm,
                          target_names,
                          pairs = None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False,
                          ax=None):
    if not isinstance(pairs, np.ndarray):
        accuracy = np.trace(cm)
    else:
        accuracy = 0
        for i, j in pairs:
            accuracy += cm[j][i]
    accuracy /= float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([''] * len(target_names))  # , rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.3f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        else:
            ax.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


ModeResult = namedtuple('ModeResult', ('mode', 'count'))
def mode_rand(a, axis):
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis+1:] + [axis])

    inds = np.ndindex(a_view.shape[:-1])
    modes = np.empty(a_view.shape[:-1], dtype=a.dtype)
    counts = np.zeros(a_view.shape[:-1], dtype=np.int32)

    for ind in inds:
        vals, cnts = np.unique(a_view[ind], return_counts=True)
        maxes = np.where(cnts == cnts.max())
        modes[ind], counts[ind] = vals[np.random.choice(maxes[0])], cnts.max()

    newshape = list(a.shape)
    newshape[axis] = 1
    return ModeResult(modes.reshape(newshape), counts.reshape(newshape))

def label_features(predictions, n_clusters):
    cluster = KMeans(n_clusters=n_clusters, init='k-means++')
    features = np.zeros((predictions.shape[1], predictions.shape[0]*n_clusters))

    for i in range(predictions.shape[0]):
        y_pred = predictions[i]
        for j in range(n_clusters):
            features[y_pred==j, i*n_clusters + j] = 1.

    _sums = np.sum(features, axis=0) / predictions.shape[1]
    y = cluster.fit_predict(features - _sums)
    return np.array(y)

def compute_results(y_pred, data, y_true=None):
    d = {}

    d['Davies-Boulding'] = metrics.davies_bouldin_score(data, y_pred)
    d['Silhouette-Score'] = metrics.silhouette_score(data, y_pred)

    if not y_true is None:
        d['NMI'] = metrics.adjusted_mutual_info_score(y_true, y_pred)
        d['ARI'] = metrics.adjusted_rand_score(y_true, y_pred)

        ind, acc = cluster_acc(y_true, y_pred)
        d['ACC'] = acc
    
        return d, ind

    return d, None