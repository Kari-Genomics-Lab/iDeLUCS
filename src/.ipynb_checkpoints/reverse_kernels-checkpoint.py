from os.path import abspath, dirname, join
import numpy as np
import itertools
from scipy.linalg import null_space
import sys

def reverse_complement(nuc):
    table = str.maketrans("ACGT", "TGCA")
    return nuc[::-1].translate(table)

def all_kmers(k):
    for i in itertools.product("ACGT", repeat=k):
        yield(''.join(i))

def create_projection_kernel(k):
    indexof = {kmer:i for i,kmer in enumerate(all_kmers(k))}
    linear_equations = list()

    # Constraint one: Frequencies sum to one (or in this scaled case, zero)
    linear_equations.append([1]*(4**k))

    # Constaint two: Frequencies are same as that of reverse complement
    for kmer in all_kmers(k):
        revcomp = reverse_complement(kmer)

        # Only look at canonical kmers - this makes no difference
        if kmer >= revcomp:
            continue

        line = [0]*(4**k)
        line[indexof[kmer]] = 1
        line[indexof[revcomp]] = -1
        linear_equations.append(line)

    linear_equations = np.array(linear_equations)
    kernel = null_space(linear_equations).astype(np.float32)
    print(kernel.shape)
    #assert kernel.shape == (4**k,((4**k)/2)-1)
    return kernel

def create_rc_kernel(k):
    indexof = {kmer:i for i,kmer in enumerate(all_kmers(k))}
    rc_matrix = np.zeros((4**k, 4**k), dtype=np.float32)
    for col, kmer in enumerate(all_kmers(k)):
        revcomp = reverse_complement(kmer)
        rc_matrix[indexof[kmer], col] += 0.5
        rc_matrix[indexof[revcomp], col] += 0.5

    return rc_matrix

def create_dual_kernel():
    return np.dot(create_rc_kernel(), create_projection_kernel())


k = int(sys.argv[1])

dual_kernel = create_rc_kernel(k)
print(dual_kernel.shape)

projection_kernel=create_projection_kernel(k)
print(projection_kernel.shape)

path = join('kernels', f'kernel{k}.npz')
np.savez_compressed(path, np.dot(dual_kernel, projection_kernel))
