import sys
import pyximport 
pyximport.install()

from .kmers import kmer_counts

import random, itertools
import numpy as np
import pandas as pd
import sklearn.metrics.cluster as metrics
from sklearn.cluster import KMeans

import torch 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

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
    return masked
    
    
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
            seq[i] = mutations[mutations.get(seq[i], N)]

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
            seq[i] = random.choice(mutations.get(seq[i], [N]))            
            
class transition_transversion(object):
    """
    Mutate Genomic sequence (in binary) using both transitions 
    and transversions.
    :param seq: Original Genomic Sequence.
    :param threshold: Probability of Transversion.
    :return: Mutated Sequence.
    """
    def __init__(self, threshold_1, threshold_2):
        self.tf1 = transition(threshold_1)
        self.tf2 = transversion(threshold_2)
        
    def __call__(self, seq):

        self.tf1(seq)
        self.tf2(seq)
    
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

                seq = check_sequence(seq_id, seq)
                names.append(seq_id)
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
    seq = check_sequence(seq_id, seq)
    names.append(seq_id) 
    lengths.append(len(seq))

    if GT_file:
        ground_truth.append(GT_dict[seq_id])

    return names, lengths, ground_truth, cluster_dis


def reverse_complement(x, k):
    numbits = 2*k  
    mask = 0xAAAAAAAA
    x = ((x >> 1) & (mask>>1)) | ((x<< 1) & mask)
    x = (1 << numbits) - 1 - x
    rev = 0

    size = 2**numbits-1
    while(size > 0):
        rev <<= 1
        if x & 1 == 1:
            rev ^= 1
        x >>=1
        size >>= 1

    return rev

def kmer_rev_comp(kmer_counts, k):
    index=[]
    for kmer in range(4**k):
        revcomp = reverse_complement(kmer,k)

        # Only look at canonical kmers - this makes no difference
        if kmer <= revcomp:
            index.append(kmer)
            kmer_counts[kmer] += kmer_counts[revcomp]
            kmer_counts[kmer] *= 0.5
        #else:
        #    kmer_counts[kmer] = kmer_counts[revcomp]

    return kmer_counts[index]


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
                names.append(seq_id)  
                            
                if transform:
                    transform(seq)
                    
                counts = np.ones(4**k, dtype=np.int32)
                kmer_counts(seq, k, counts)
                #cgr(seq, k, counts)

                if reduce:
                    counts = kmer_rev_comp(counts,k)


                kmers.append(counts / np.sum(counts))
                
                lines = []
                seq_id = line[1:-1].decode()  # Modify this according to your labels.  
            seq_id = line[1:-1].decode()
        
        else:
            lines += [line.strip()]
  
    seq = bytearray().join(lines)
    names.append(seq_id)
    if transform:
        transform(seq)
        
    counts = np.ones(4**k, dtype=np.int32)
    kmer_counts(seq, k, counts)
    #cgr(seq, k, counts)
    if reduce:
        counts = kmer_rev_comp(counts,k)
    kmers.append(counts / np.sum(counts))
    
    #if reduce:
    #    K_file = np.load(open(f'kernels/kernel{k}.npz','rb'))
    #    KERNEL = K_file['arr_0']
    #    return names, np.dot(np.array(kmers), KERNEL)
        
    return names, np.array(kmers)

def cgrFasta(fname, k=6, transform=None):
    lines = list()
    seq_id = ""
    names, kmers = [], []

    for line in open(fname, "rb"):
        if line.startswith(b'#'):
            pass

        elif line.startswith(b'>'):
            if seq_id != "":
                seq = bytearray().join(lines)
                names.append(seq_id)  
                            
                if transform:
                    transform(seq)
                    
                counts = np.ones(4**k, dtype=np.int32)
                #kmer_counts(seq, k, counts)
                cgr(seq, k, counts)
                kmers.append(counts / np.sum(counts))
                
                lines = []
                seq_id = line[1:-1].decode()  # Modify this according to your labels.  
            seq_id = line[1:-1].decode()
        
        else:
            lines += [line.strip()]
  
    seq = bytearray().join(lines)
    names.append(seq_id)
    if transform:
        transform(seq)
        
    counts = np.ones(4**k, dtype=np.int32)
    #kmer_counts(seq, k, counts)
    cgr(seq, k, counts)
    kmers.append(counts / np.sum(counts))
    return names, np.array(kmers)

 
import time 
def AugmentFasta(sequence_file, n_mimics, k=6, reduce=False):

    train_features = []
    start = time.time()

    # Compute Features and save original data for testing.
    #sys.stdout.write(f'\r............computing augmentations (0/{n_mimics})................')
    #sys.stdout.flush()
    
    _, t_norm = kmersFasta(sequence_file, k=k, transform=transition_transversion(1e-2, 0.5e-2), reduce=reduce)
    t_norm.resize(t_norm.shape[0],1,t_norm.shape[1])
    
    
    #sys.stdout.write(f'\r............computing augmentations (1/{n_mimics})................')
    #sys.stdout.flush()
    _, t_mutated = kmersFasta(sequence_file, k=k, transform=transition(1e-2), reduce=reduce)
    t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
    train_features.extend(np.concatenate((t_norm, t_mutated), axis=1))
    
    #sys.stdout.write(f'\r............computing augmentations (2/{n_mimics})................')
    #sys.stdout.flush()
    _, t_mutated = kmersFasta(sequence_file, k=k, transform=transversion(0.5e-2), reduce=reduce)
    t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
    train_features.extend(np.concatenate((t_norm, t_mutated), axis=1))
    
    for j in range(n_mimics-2):
        #sys.stdout.write(f'\r............computing augmentations ({3+j}/{n_mimics})................')
        #sys.stdout.flush()
        _, t_mutated = kmersFasta(sequence_file, k=k, transform=Random_N(20), reduce=reduce)
        t_mutated.resize(t_mutated.shape[0], 1, t_mutated.shape[1])
        train_features.extend(np.concatenate((t_norm, t_mutated), axis=1)) 

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
    return DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0) #num_workers=4


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

def generate_csv_mapping(cm, target_names):
    result_dict = {}
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  
        result_dict[(target_names[i], j)] = cm[i, j]
    pandas_dict = {
        "Predicted": [],
        "Actual": [],
        "Count": []
    }
    for (target1, target2), count in result_dict.items():
        pandas_dict["Predicted"].append(target2)
        pandas_dict["Actual"].append(target1)
        pandas_dict["Count"].append(count)
    
    result_pd = pd.DataFrame(pandas_dict)
    result_pd.to_csv('confusion_matrix.csv')

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

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


from sklearn.metrics.pairwise import euclidean_distances
def label_features(predictions, n_clusters):
    """
    Clustering Ensemble 
    """
    cluster = KMeans(n_clusters=n_clusters, init='k-means++')
    features = np.zeros((predictions.shape[1], predictions.shape[0]*n_clusters))

    for i in range(predictions.shape[0]):
        y_pred = predictions[i]
        for j in range(n_clusters):
            features[y_pred==j, i*n_clusters + j] = 1.

    _sums = np.sum(features, axis=0) / predictions.shape[1]
        
    y = cluster.fit_predict(features - _sums)
    D = 1.0 / euclidean_distances(features - _sums, cluster.cluster_centers_, squared=True)
    D **= 1.0 / (2 - 1)
    D /= np.sum(D, axis=1)[:, np.newaxis]
    fuzzy_labels_ = D
    
    return np.array(y), fuzzy_labels_.max(axis=1)



def compute_results(y_pred, data, y_true=None):
    d = {}

    d['Davies-Boulding'] = metrics.davies_bouldin_score(data, y_pred)
    d['Silhouette-Score'] = metrics.silhouette_score(data, y_pred)

    if not y_true is None:
        d['NMI'] = metrics.adjusted_mutual_info_score(y_true, y_pred)
        d['ARI'] = metrics.adjusted_rand_score(y_true, y_pred)
        d['Homogeneity'] = metrics.homogeneity_score(y_true, y_pred)
        d['Completeness'] = metrics.completeness_score(y_true, y_pred)

        ind, acc = cluster_acc(y_true, y_pred)
        d['ACC'] = acc
    
        return d, ind

    return d, None



        
from itertools import permutations, product
import sys

def modified_Hungarian(test):
    rows, cols = linear_sum_assignment(test)
    new_cols = np.array(list(set(range(test.shape[1])).difference(cols)))

    if new_cols.shape[0] != 0:
        minval = sys.maxsize

        for i in permutations(new_cols, len(new_cols)):
            for j in product(rows, repeat=len(new_cols)):
                _sum = sum(test[j,i])
                if _sum < minval:
                    good_assignment = (j,i)

                minval = min(minval, _sum)

        full_rows = np.array(list(rows) + list(good_assignment[0]))
        full_cols = np.array(list(cols) + list(good_assignment[1]))
    else:
        full_rows, full_cols = rows, cols

    return full_rows, full_cols

def modified_cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy with more predicted clusters than true labels
    :param y_true: true labels, numpy.array with shape `(n_samples,)`
    :param y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    :return:  accuracy, in [0,1]
    """

    y_true = y_true.astype(np.int64)
    w = np.zeros((y_true.max() + 1, y_pred.max() + 1), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    
    #print(w)

    ind = modified_Hungarian(w.max() - w)
    ind = np.asarray(ind).T
    return ind, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size