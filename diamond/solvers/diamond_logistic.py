""" Diamond solver for logistic regression """
import logging
import time
import numpy as np
from diamond.solvers.utils import dot
from diamond.solvers.utils import l2_logistic_fixed_hessian

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class FixedHessianSolverMulti(object):
    """ This class wraps the fit method"""

    def __init__(self):
        pass

    @staticmethod
    def fit(Y, designs, H_invs, sparse_inv_covs, **kwargs):
        """ Fit the model. Outer iterations loop over main effects and each grouping factor

        Args:
            Y : array_like. Vector of binary responses in {0, 1}
            designs : dict. Design matrices for main effects and grouping factors
            H_invs: dict. dictionary of inverse Hessian matrix for each grouping factor
            sparse_inv_covs : dict. dictionary of sparse regularization matrices,\
            one for each grouping factor

        Keyword Args:
            min_its : int. Minimum number of outer iterations
            max_its : int. Maximum number of outer iterations
            tol : float. If parameters change by less than `tol`, convergence has been achieved.
            inner_tol : float. Tolerance for inner loops
            initial_offset : array_like. Offset vector. Defaults to 0
            fit_order : list. Order in which to fit main and random effects
            permute_fit_order : boolean. Change the fit order at each iteration
            verbose : boolean. Display updates at every iteration

        Returns:
            dict: estimated intercepts, main effects, and random effects
        """

        min_its = kwargs.get('min_its', 20)
        max_its = kwargs.get('max_its', 5000)
        tol = kwargs.get('tol', 1E-5)
        inner_tol = kwargs.get('inner_tol', 1E-2)
        fit_order = kwargs.get('fit_order', None)
        permute_fit_order = kwargs.get('permute_fit_order', False)
        initial_offset = kwargs.get('offset', 0.)
        verbose = kwargs.get('verbose', False)
        if not verbose:
            LOGGER.setLevel(logging.WARNING)

        # Cycle through fitting different groupings

        start_time = time.time()
        effects = {k: np.zeros(designs[k].shape[1]) for k in designs.keys()}
        old_effects = {k: np.zeros(designs[k].shape[1]) for k in designs.keys()}

        if fit_order is None:
            fit_order = designs.keys()

        for i in range(max_its):
            # periodically recompute the offset
            if i % 10 == 0:
                offset = initial_offset
                for g in designs.keys():
                    offset += dot(designs[g], effects[g])

            change = 0.0
            for grouping in fit_order:
                if i > 0:
                    g_change = np.linalg.norm(effects[grouping] - old_effects[grouping]) / \
                               np.linalg.norm(effects[grouping])
                else:
                    g_change = np.inf
                # if g_change < tol and i >= min_its:
                #     # no need to continue converging this group
                #     # cutoff
                #     continue

                old_effects[grouping] = 1.0 * effects[grouping]
                offset += -dot(designs[grouping], effects[grouping])

                # fit group effects
                effects[grouping] = 1.0 * l2_logistic_fixed_hessian(designs[grouping],
                                                                    Y,
                                                                    H_invs[grouping],
                                                                    sparse_inv_covs[grouping],
                                                                    offset=offset,
                                                                    beta=effects[grouping],
                                                                    tol=inner_tol)

                offset += dot(designs[grouping], effects[grouping])

                change = max(change,
                             np.linalg.norm(effects[grouping] - old_effects[grouping]) /
                             np.linalg.norm(effects[grouping]))

            xbeta = 0
            penalty = 0
            for g in designs.keys():
                xbeta += dot(designs[g], effects[g])
                if g != 'main':
                    penalty += dot(effects[g], dot(sparse_inv_covs[g], effects[g]))
            loss = -1 * np.sum(dot(Y, xbeta) - np.log(1 + np.exp(-xbeta)))
            obj = loss + penalty
            LOGGER.info("loss: %f penalty: %f seconds elapsed %d",
                        loss, penalty, time.time() - start_time)
            LOGGER.info("iteration: %d relative coef change: %f obj: %f",
                        i, change, obj)

            if permute_fit_order:
                # change cycle order for next iteration
                np.random.shuffle(fit_order)

            if change < tol and i >= min_its:
                LOGGER.info("total seconds elapsed: %0.0f", time.time() - start_time)
                LOGGER.info("reached convergence after %d steps. relative coef change: %f",
                            i, change)
                break

        return effects, obj
