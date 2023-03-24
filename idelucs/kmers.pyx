#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
cpdef void kmer_counts(unsigned char[::1] seq, int k, int[::1] counts):

    """
    Adapted from VAMB: https://github.com/RasmussenLab/vamb

    seq is expected to be np.uint8 of bytevalues of the contig.
    Only values  64, 67, 71, 84 (A, C, G, T) are accepted, all 
    others are skipped. The counts vector is expected to be an 
    array of 4^k 32-bit integers.
    """

    cdef unsigned int k_mer = 0
    cdef int bp, bp_code, i
    cdef int countdown = k-1
    cdef int contiglength = len(seq)
    cdef int size = (1 << (2 * k)) - 1

    cdef unsigned int* encoding = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    
    

    for i in range(contiglength):
        bp = seq[i]
        bp_code = encoding[bp]

        if bp_code == 4:
            countdown = k

        k_mer = ((k_mer << 2) | bp_code) & size

        if countdown == 0:
            counts[k_mer] += 1
        else:
            countdown -= 1

    
cpdef void cgr(unsigned char[::1] seq, int k, int[::1] CGR):
    """
    seq is expected to be np.uint8 of bytevalues of the contig.
    Only values  64, 67, 71, 84 (A, C, G, T) are accepted, all 
    others are skipped. The CGR vector  is expected to be an 
    array of 4^k 32-bit integers that when reshaped, should 
    show the fCGR
    """

    cdef unsigned int cgr_i = 0
    cdef unsigned int cgr_j = 0

    cdef int bp, i_bit, j_bit
    cdef int n_bp = 0
    cdef int contiglength = len(seq)
    cdef int size = (1 << k) - 1

    cdef unsigned int* i_encoding = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 1, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    cdef unsigned int* j_encoding = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 0, 2, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for i in range(contiglength):
        bp = seq[i]
        i_bit = i_encoding[bp]
        j_bit = j_encoding[bp]

        if i_bit < 2 :
            cgr_i = (i_bit << n_bp | cgr_i) #Move previous kmer, append new one and use the mask to fix kmer-length
            cgr_j = (j_bit << n_bp | cgr_j)  
            n_bp += 1        
        else:
            n_bp = 0  #Restart whenever there's an N
            cgr_i = 0
            cgr_j = 0
        

        if n_bp == k:
            CGR[(cgr_i << k) + cgr_j] += 1
            cgr_i >>= 1 
            cgr_j >>= 1 
            n_bp -= 1  #The other 

