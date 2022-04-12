import sys
sys.path.append('src/')
import pyximport 
pyximport.install()

from c_utils import  _kmercounts

import random, itertools
from datetime import datetime
import os as os
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class FastaEntry:
    """One single FASTA entry. Instantiate with string header and bytearray
    sequence."""

    basemask = bytearray.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV-',
                                   b'ACGTTTNNNNNNNNNNNNNNNNNNNNNN')
    __slots__ = ['header', 'sequence']

    def __init__(self, header, sequence):
        if len(header) > 0 and (header[0] in ('>', '#') or header[0].isspace()):
            raise ValueError('Header cannot begin with #, > or whitespace')
        if '\t' in header:
            raise ValueError('Header cannot contain a tab')

        masked = sequence.translate(self.basemask, b' \t\n\r')
        stripped = masked.translate(None, b'ACGTN')
        if len(stripped) > 0:
            bad_character = chr(stripped[0])
            msg = "Non-IUPAC DNA byte in sequence {}: '{}'"
            raise ValueError(msg.format(header, bad_character))

        self.header = header
        self.sequence = masked

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return '>{}\n{}'.format(self.header, self.sequence.decode())

    def format(self, width=60):
        sixtymers = range(0, len(self.sequence), width)
        spacedseq = '\n'.join([self.sequence[i: i+width].decode() for i in sixtymers])
        return '>{}\n{}'.format(self.header, spacedseq)

    def __getitem__(self, index):
        return self.sequence[index]

    def __repr__(self):
        return '<FastaEntry {}>'.format(self.header)

    def kmercounts(self, k):
        if k < 1 or k > 10:
            raise ValueError('k must be between 1 and 10 inclusive')

        counts = np.zeros(1 << (2*k), dtype=np.int32)
        _kmercounts(self.sequence, k, counts)
        return counts / np.linalg.norm(counts) #counts / np.sum(counts)
    
    
def byte_iterfasta(filehandle, comment=b'#'):
    """Yields FastaEntries from a binary opened fasta file.
    Usage:
    >>> with Reader('/dir/fasta.fna', 'rb') as filehandle:
    ...     entries = byte_iterfasta(filehandle) # a generator
    Inputs:
        filehandle: Any iterator of binary lines of a FASTA file
        comment: Ignore lines beginning with any whitespace + comment
    Output: Generator of FastaEntry-objects from file
    """

    # Make it work for persistent iterators, e.g. lists
    line_iterator = iter(filehandle)
    # Skip to first header
    try:
        for probeline in line_iterator:
            stripped = probeline.lstrip()
            if stripped.startswith(comment):
                pass

            elif probeline[0:1] == b'>':
                break

            else:
                raise ValueError('First non-comment line is not a Fasta header')

        else: # no break
            raise ValueError('Empty or outcommented file')

    except TypeError:
        errormsg = 'First line does not contain bytes. Are you reading file in binary mode?'
        raise TypeError(errormsg) from None

    header = probeline[1:-1].decode()
    buffer = list()

    # Iterate over lines
    for line in line_iterator:
        if line.startswith(comment):
            pass

        elif line.startswith(b'>'):
            yield FastaEntry(header, bytearray().join(buffer))
            buffer.clear()
            header = line[1:-1].decode()

        else:
            buffer.append(line)

    yield FastaEntry(header, bytearray().join(buffer))
    
    
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
            seq[i] = mutations[seq[i]]

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
            seq[i] = random.choice(mutations[seq[i]])
            
            
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

def add_noise(x_train):
    """
    Add artificial Gaussian noise to a training sample.
    :param x_train: ndarray with normalized kmers.
    :return: ndarray with gaussian noise.
    """
    n_features = x_train.shape[0]
    index = (np.random.random(n_features) < 0.25).astype('float32')
    noise = np.random.normal(0, 0.001, n_features)
    gaussian_train = x_train + noise * index
    return gaussian_train
        
def ProcessFasta(data_path, k, transform=None, GT_file = None):
    
    minlength = 1
    raw = []
    projected = []
    lengths = []
    contignames = list()
    ground_truth = None
     
    K_file = np.load(open(f'kernels/kernel{k}.npz','rb'))
    KERNEL = K_file['arr_0']

    
    if GT_file: 
        ground_truth = []
        df = pd.read_csv(GT_file, sep='\t')
        GT_dict = dict(zip(df.sequence_id, df.cluster_id))
    
    with open(data_path, 'rb') as fasta_file:
        entries = byte_iterfasta(fasta_file)

        for entry in entries:
            if (GT_file and not entry.header in GT_dict):
                raise ValueError('Check GT for sequence {}'.format(entry.header))

            if len(entry) < minlength:
                continue
                
            if GT_file:
                ground_truth.append(GT_dict[entry.header])
                
            if transform:
                transform(entry.sequence)
                raw.extend(add_noise(entry.kmercounts(k)))
            else:
                raw.extend(entry.kmercounts(k))


            lengths.append(len(entry))
            contignames.append(entry.header)
                  
    # Convert rest of contigs
    kmers = np.array(raw)

    # Don't use reshape since it creates a new array object with shared memory
    kmers.shape = (len(kmers)//4**k, 4**k)
    contignames = np.array(contignames)
    lengths = np.array(lengths)
    
    if ground_truth:
        groud_truth = np.array(ground_truth)

    return np.dot(kmers, KERNEL) , contignames, lengths, ground_truth
 
class SequenceDataset(Dataset):
    """ Dataset creation directly from fasta file"""
    
    def __init__(self, fasta_file, k=4, transform=None, GT_file=None):
        """ Args:
            fasta_file (string): Path to te fasta file
            transform (callable, optional): Optional transform to be applied on a 
                                            sequence. Function computing the mimics 
        """
        
        self.kmers, self.names, self.lengths, self.GT = ProcessFasta(fasta_file, k, transform, GT_file)
        
        
    def __len__(self):
        return self.lengths.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.GT:           
            sample = {'kmer':self.kmers[idx,:],'name':self.names[idx], 'cluster_id':self.GT[idx]}
        else:
            sample = {'kmer':self.kmers[idx,:],'name':self.names[idx]}
            
        return sample
    
    
def create_dataloaders(fasta_file,
                       mutations,
                       k=4,
                       batch_sz=350,
                       shuffle=False,
                       GT_file=None):

    train_kmers = SequenceDataset(fasta_file, k=k, transform=lambda x:x, GT_file=GT_file)
    
    index = list(range(len(train_kmers)))
    
    sampler = torch.utils.data.sampler.BatchSampler(index,
                                                    batch_size=batch_sz,
                                                    drop_last=True)
    
    train_dataloader = DataLoader(train_kmers, 
                                  batch_sampler=sampler,
                                  num_workers=0)

    if not shuffle:
        assert (isinstance(train_dataloader.sampler,
                       torch.utils.data.sampler.SequentialSampler))
    dataloaders = [train_dataloader]

    for d_i in range(len(mutations)):
        print(" Computing mimic sequences dataloader %d out of %d time %s" % 
              (d_i + 1, len(mutations), datetime.now()))
        sys.stdout.flush()

        train_mimics = SequenceDataset(fasta_file, k=k, transform=mutations[d_i], GT_file=GT_file)

        train_mimics_dataloader = DataLoader(train_mimics,
                                         batch_sampler=sampler,
                                         num_workers=0)

        if not shuffle:
            assert (isinstance(train_mimics_dataloader.sampler, torch.utils.data.sampler.SequentialSampler))
    
        assert (len(train_dataloader) == len(train_mimics_dataloader))
    
        dataloaders.append(train_mimics_dataloader)

    num_train_batches = len(dataloaders[0])
    print("Length of datasets vector %d" % len(dataloaders))
    print("Number of batches per epoch: %d" % num_train_batches)
    sys.stdout.flush()

    return index, dataloaders

from colorsys import hsv_to_rgb
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
    all_colours = [list((np.array(hsv_to_rgb(hue, 0.8, 0.8)) * 255.).astype(
    np.uint8)) for hue in hues]
    
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
    # print(w)
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return ind, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False,
                          ax=None):

    accuracy = np.trace(cm) / float(np.sum(cm))
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

from collections import namedtuple
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