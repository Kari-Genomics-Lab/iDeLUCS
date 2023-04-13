import sys
import torch

import torch.nn as nn
import torch.nn.functional as F


global label_dtype
global dtype

dtype = torch.FloatTensor
label_dtype = torch.LongTensor

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.cuda.FloatTensor
    label_dtype = torch.cuda.LongTensor

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """
    Implementation of the IID loss function found in the paper:
    "Invariant Information Clustering for Unsupervised Image 
    Classification and Segmentation"
    The function can be found originally in https://github.com/xu-ji/IIC
    """
    
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k).clone()
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k).clone()  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j)
                      - lamb * torch.log(p_j)
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def compute_joint(x_out, x_tf_out):
    """
    Produces variable that requires grad (since args require grad)
    The function can be found originally in https://github.com/xu-ji/IIC
    """
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def info_nce_loss(z1,z2, temperature):
    """Info-NCE los function.
        Introduced by "A Simple Framework for Contrastive 
        Learning of Visual Representations" by T. Chen,
        S. Kornblith, M. Norouzi, and Geoffrey Hinton 
        (https://arxiv.org/abs/2002.05709)
    """
    N = z1.shape[0]
    criterion = nn.CrossEntropyLoss()
    features = torch.cat((z1, z2), 0).float()
    labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0]).type(label_dtype)

    logits = logits / temperature
    loss = criterion(logits, labels)
    return loss
