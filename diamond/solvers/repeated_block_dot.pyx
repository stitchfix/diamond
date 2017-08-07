import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def repeated_block_dot(np.ndarray[double, ndim=2] block, 
    		       long block_shape, 
                       np.ndarray[double, ndim=1] v):

    cdef long p = len(v)
    cdef long num_blocks = p/block_shape
    cdef long i, j, k, ind, block_offset
    cdef np.ndarray[double, ndim=1] results = np.zeros(p)

    for k in range(num_blocks):
        for i in range(block_shape):
            block_offset = block_shape * k
            ind = block_offset + i
            for j in range(block_shape):
                results[ind] += v[block_offset + j] * block[i, j]

    return results
