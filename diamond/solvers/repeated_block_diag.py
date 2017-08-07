import numpy as np
from scipy import sparse

import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)

from diamond.solvers.repeated_block_dot import repeated_block_dot


class RepeatedBlockDiagonal(object):

    def __init__(self, block, num_blocks):

        (k1, k2) = block.shape
        assert k1 == k2, "Block must be square"

        self._block = 1.0 * block
        self._block_shape = k1
        self._num_blocks = num_blocks

    def dot(self, x):
        """ Perform dot product between RepeatedBlockDiagonal objects

        Args:
            x : RepeatedBlockDiagonal.

        Returns:
            repeated_block_dot with dot product of the 2 matrices
        """

        return repeated_block_dot(self._block, self._block_shape, x)

    @property
    def sparse_matrix(self):
        """ Create a block diagonal sparse matrix, with a single repeated block

        This method essentially replicates the functionality of scipy.linalg.block_diag.
        We rewrote it for this special case to improve the memory footprint - scipy's version
        is general purpose, allowing different blocks along the diagonal. The implementation of
        it is too memory intensive and quickly causes OOM errors with resonably sized
        design matrices. In our case, we only have one block we are replicating along the
        diagonal, so we don't need to iteratively build up the overall block diagonal matrix.

        :return:
            csr block diagonal matrix
        """
        # L = number of repated blocks (i.e. num_levels)
        L = self._num_blocks
        K = self._block_shape

        _block_rep = np.tile(np.ravel(self._block.flatten()), L)

        i = np.repeat(np.arange(L*K), K)
        j = np.tile(np.tile(np.arange(K),K), L) + np.repeat(np.arange(0, L*K, K), K*K)

        return sparse.csr_matrix((_block_rep, (i,j)), shape=(L*K,L*K))


# TODO: Make real unit tests
def test_repeated_block():

    np.random.seed(100)

    k = 5
    m = 10

    block = np.random.standard_normal((k,k))
    vec = np.random.standard_normal(k*m)

    r = RepeatedBlockDiagonal(block, m)

    np.testing.assert_array_almost_equal(r.dot(vec),
                                         r.sparse_matrix.dot(vec))
