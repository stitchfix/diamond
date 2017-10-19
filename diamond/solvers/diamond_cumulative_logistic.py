""" Diamond solver for cumulative logistic regression """
import logging
import time
import numpy as np
from diamond.solvers.utils import dot
from diamond.solvers.utils import l2_clogistic_fixef, l2_clogistic_ranef

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class FixedHessianSolverCumulative(object):
    """This class wraps the fit method"""

    def __init__(self):
        pass

    @staticmethod
    def fit(Y, main_design, inter_designs,
            H_inter_LUs, penalty_matrices, effects, **kwargs):
        """ Fit the model. Outer iterations loop over intercepts, \
            main effects, and each grouping factor

        Args:
            Y : array_like. Response matrix.
            main_design : array_like. Design matrix for main effects
            inter_designs : dict of array_like: design matrices for random effects
            H_inter_LUs : dict of LU-decompositions of approximate Hessian matrices,\
            one for each grouping factor
            penalty_matrices : dict of sparse matrices for regularization
            effects: dict. Current value of parameter estimates

        Keyword Args:
            min_its : int. Minimum number of outer iterations
            max_its : int. Maximum number of outer iterations
            tol : float. If parameters change by less than `tol`, convergence has been achieved.
            inner_tol : float. Tolerance for inner loops
            offset : array_like. Offset vector. Defaults to 0
            verbose : boolean. Display updates at every iteration

        Returns:
            dict of estimated intercepts, main effects, and random effects
        """

        min_its = kwargs.get('min_its', 10)
        max_its = kwargs.get('max_its', 100)
        tol = kwargs.get('tol', 1e-5)
        inner_tol = kwargs.get('tol', 1e-2)
        offset = kwargs.get('offset', np.zeros(Y.shape[0]))
        verbose = kwargs.get('verbose', False)
        if not verbose:
            LOGGER.setLevel(logging.WARNING)

        start_time = time.time()
        old_effects = {k: effects[k] for k in inter_designs}

        for i in range(max_its):
            change = 0.0
            # first, do the fixed effects
            LOGGER.info("Starting to fit fixed effects")
            if main_design is not None:
                offset += -dot(main_design, effects['main'])
            alpha, main_effects = l2_clogistic_fixef(X=main_design,
                                                     Y=Y,
                                                     offset=offset,
                                                     alpha=effects['intercepts'],
                                                     beta=effects['main'],
                                                     tol=inner_tol,
                                                     min_its=1)
            if main_design is not None:
                offset += dot(main_design, main_effects)
            for grouping in H_inter_LUs:
                LOGGER.info("Starting to fit %s", grouping)
                if i > 0:
                    g_change = np.linalg.norm(effects[grouping] - old_effects[grouping]) / \
                               np.linalg.norm(effects[grouping])
                else:
                    g_change = np.inf
                if g_change < inner_tol and i >= min_its:
                    LOGGER.info("Group %s has already converged, skipping it", grouping)
                    continue
                old_effects[grouping] = 1.0 * effects[grouping]
                offset += -dot(inter_designs[grouping], effects[grouping])
                effects[grouping] = 1.0 * l2_clogistic_ranef(X=inter_designs[grouping],
                                                             Y=Y,
                                                             LU=H_inter_LUs[grouping],
                                                             penalty_matrix=penalty_matrices[grouping],
                                                             alpha=alpha,
                                                             beta=effects[grouping],
                                                             offset=offset,
                                                             tol=inner_tol,
                                                             min_its=4)
                offset += dot(inter_designs[grouping], effects[grouping])
                change_g = np.linalg.norm(effects[grouping] - old_effects[grouping]) / np.linalg.norm(effects[grouping])
                change = max(change, change_g)

            LOGGER.info("seconds elapsed: %0.0f", (time.time() - start_time))
            LOGGER.info("iteration: %d relative coef change: %f", i, change)
            if change < tol and i >= min_its:
                LOGGER.info("total seconds elapsed: %0.0f", time.time() - start_time)
                LOGGER.info("reached convergence after %d steps. relative coef change: %f",
                            i, change)
                break
        # put everything in 1 dictionary and return it
        effects['intercepts'] = alpha
        effects['main'] = main_effects
        return effects
