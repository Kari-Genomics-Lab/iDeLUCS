import sys

sys.path.append('src/')
import pyximport 
pyximport.install()

from c_utils import  _kmercounts

import random
from datetime import datetime
import os as os
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment


class FastaEntry:
    """One single FASTA entry. Instantiate with string header and bytearray
    sequence."""

    basemask = bytearray.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV',
                                   b'ACGTTTNNNNNNNNNNNNNNNNNNNNN')
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
        return counts / np.sum(counts)
    
    
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
            if len(entry) < minlength or (GT_file and not entry.header in GT_dict) :
                continue
                
            if GT_file:
                ground_truth.append(GT_dict[entry.header])
                
            if transform: 
                transform(entry.sequence)
                
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
        
        self.kmers, self.names, self.lengths, self.GT =  ProcessFasta(fasta_file, k, transform, GT_file)
        
        
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

    train_kmers = SequenceDataset(fasta_file, k=k, transform=None, GT_file=GT_file)
    
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


class CAMIDataset(Dataset):
    """ Dataset creation directly from fasta file"""
    
    def __init__(self, fasta_file, abundance_file, k=4, transform=None, GT_file=None):
        """ Args:
            fasta_file (string): Path to te fasta file
            transform (callable, optional): Optional transform to be applied on a 
                                            sequence. Function computing the mimics 
        """
        
        self.kmers, self.names, self.lengths, self.GT =  ProcessFasta(fasta_file, k, transform, GT_file)
        
        with np.load(abundance_file) as data: 
            self.rpkms = data['arr_0']
        
        
    def __len__(self):
        return self.lengths.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.GT:           
            sample = {'kmer':self.kmers[idx,:],'name':self.names[idx], 'cluster_id':self.GT[idx], 'rpkm':self.rpkms[idx,:]}
        else:
            sample = {'kmer':self.kmers[idx,:],'rpkm':self.rpkms[idx,:], 'name':self.names[idx]}
            
        return sample

def create_CAMI_dataloaders(fasta_file, 
                            abundance_file,
                            mutations,
                            k=4,
                            batch_sz=350,
                            shuffle=False,
                            GT_file=None):
    
    print(" Computing original sequences dataloader time %s" % datetime.now())

    train_kmers = CAMIDataset(fasta_file, abundance_file, k=k, transform=None, GT_file=GT_file)
    
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

        train_mimics = CAMIDataset(fasta_file, abundance_file, k=k, transform=mutations[d_i], GT_file=GT_file)

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
