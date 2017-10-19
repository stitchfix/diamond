""" Test gradient and hessian functions
for cumulative logistic regression
by using finite differences
https://en.wikipedia.org/wiki/Finite_difference
"""
from diamond.solvers.utils import l2_clogistic_gradient, l2_clogistic_llh, l2_clogistic_hessian, \
    _l2_clogistic_hessian_slope
from diamond.glms.cumulative_logistic import CumulativeLogisticRegression
import unittest
import pandas as pd
import numpy as np

class TestFiniteDifference(unittest.TestCase):
    def setUp(self):
        wine = pd.read_csv("examples/ordinal/wine.csv")
        # convert categorical to numeric
        wine.temp = 1.0 * (wine.temp == 'warm')
        wine.contact = 1.0 * (wine.contact == "yes")
        clmm_variance = 1.2798
        df_variance = pd.DataFrame({"group": "judge", "var1": "intercept", "vcov": clmm_variance,
                                    "var2": np.nan, "sdcor": np.sqrt(clmm_variance)}, index=[0])
        ff = "rating ~ temp + contact + (1|judge)"
        self.solver = CumulativeLogisticRegression(wine, df_variance)
        self.results = self.solver.fit(ff, tol=1e-8)

    def test_gradient_fixef(self, h=1e-6):
        " Use finite differences to test gradient of main effects "

        grad = l2_clogistic_gradient(
            X=self.solver.main_design,
            Y=self.solver.response,
            intercept=True,
            beta=self.results["main"]["main_value"].values,
            alpha=self.results["intercepts"],
            # need to adjust for random effects
            offset=self.solver.grouping_designs["judge"].dot(self.results["judge"]["intercept"].values)
        )

        for i in range(len(grad)):
            delta = np.zeros(len(grad))
            delta[i] = h

            grad_fwd = \
                l2_clogistic_llh(X=self.solver.main_design,
                                 Y=self.solver.response,
                                 alpha=self.results["intercepts"] + delta[:4],
                                 beta=self.results["main"]["main_value"].values + delta[4:],
                                 penalty_matrix=np.zeros((2, 2)),
                                 offset=self.solver.grouping_designs["judge"].dot(
                                     self.results["judge"]["intercept"].values)
                                 )
            grad_bwd = \
                l2_clogistic_llh(X=self.solver.main_design,
                                 Y=self.solver.response,
                                 alpha=self.results["intercepts"] - delta[:4],
                                 beta=self.results["main"]["main_value"].values - delta[4:],
                                 penalty_matrix=np.zeros((2, 2)),
                                 offset=self.solver.grouping_designs["judge"].dot(
                                     self.results["judge"]["intercept"].values)
                                 )
            grad_fd = (grad_fwd - grad_bwd) / (2 * h)
            self.assertAlmostEqual(grad[i], grad_fd, places=6)

    def test_gradient_ranef(self, h=1e-6):
        " Use finite differences to test gradient of random effects "

        grad = \
            l2_clogistic_gradient(X=self.solver.grouping_designs["judge"],
                                  Y=self.solver.response,
                                  alpha=self.results["intercepts"],
                                  beta=self.results["judge"]["intercept"].values,
                                  penalty_matrix=self.solver.sparse_inv_covs["judge"].sparse_matrix,
                                  # adjust for main effects
                                  offset=self.solver.main_design.dot(self.results["main"]["main_value"].values))
        # this function includes derivatives of the intercepts, which we checked above
        # so restrict to just the random effects
        grad = grad[4:]

        for i in range(len(grad)):
            delta = np.zeros(len(grad))
            delta[i] = h

            grad_fwd = \
                l2_clogistic_llh(X=self.solver.grouping_designs["judge"],
                                 Y=self.solver.response,
                                 alpha=self.results["intercepts"],
                                 beta=self.results["judge"]["intercept"].values + delta,
                                 penalty_matrix=self.solver.sparse_inv_covs["judge"].sparse_matrix,
                                 offset=self.solver.main_design.dot(self.results["main"]["main_value"].values)
                                 )
            grad_bwd = \
                l2_clogistic_llh(X=self.solver.grouping_designs["judge"],
                                 Y=self.solver.response,
                                 alpha=self.results["intercepts"],
                                 beta=self.results["judge"]["intercept"].values - delta,
                                 penalty_matrix=self.solver.sparse_inv_covs["judge"].sparse_matrix,
                                 offset=self.solver.main_design.dot(self.results["main"]["main_value"].values)
                                 )
            grad_fd = (grad_fwd - grad_bwd) / (2 * h)
            self.assertAlmostEqual(grad[i], grad_fd, places=6)

    def test_hessian_fixef(self, h=1e-4):
        " test hessian for intercepts and main effects "
        PI = np.zeros((2, 2))
        np.random.seed(413698)
        alpha = sorted(np.random.normal(size=4))
        beta = np.random.normal(size=2)

        hess = l2_clogistic_hessian(X=self.solver.main_design,
                                    Y=self.solver.response,
                                    intercept=True,
                                    alpha=alpha,
                                    beta=beta,
                                    offset=None,
                                    penalty_matrix=PI)

        def eval_nllh(alpha_beta):
            alpha = alpha_beta[:4]
            beta = alpha_beta[4:]
            llh = l2_clogistic_llh(X=self.solver.main_design,
                                   Y=self.solver.response,
                                   alpha=alpha,
                                   beta=beta,
                                   offset=None,
                                   penalty_matrix=PI)
            return -1 * llh

        for i in range(6):
            delta = np.zeros(6)
            delta[i] = h
            for j in range(6):
                gamma = np.zeros(6)
                gamma[j] = h
                # finite difference formula from
                # https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
                alpha_beta = np.concatenate([alpha, beta])
                f1 = eval_nllh(alpha_beta + delta + gamma)
                f2 = eval_nllh(alpha_beta + delta - gamma)
                f3 = eval_nllh(alpha_beta - delta + gamma)
                f4 = eval_nllh(alpha_beta - delta - gamma)

                fd_approx = (f1 - f2 - f3 + f4) / (4 * h ** 2)
                self.assertAlmostEqual(hess[i, j], fd_approx, places=3)

    def test_hessian_ranef(self, h=1e-4):
        " test hessian for random effects "
        np.random.seed(325798)
        alpha = sorted(np.random.normal(size=4))
        beta = np.random.normal(size=9)

        hess = l2_clogistic_hessian(X=self.solver.grouping_designs["judge"],
                                    Y=self.solver.response,
                                    intercept=False,
                                    alpha=alpha,
                                    beta=beta,
                                    offset=None,  # don't need the offsets for a unit test)
                                    penalty_matrix=self.solver.sparse_inv_covs["judge"].sparse_matrix)

        def eval_nllh(beta):
            llh = l2_clogistic_llh(X=self.solver.grouping_designs["judge"],
                                   Y=self.solver.response,
                                   alpha=alpha,
                                   beta=beta,
                                   offset=None,
                                   penalty_matrix=self.solver.sparse_inv_covs["judge"].sparse_matrix)
            return -1 * llh

        for i in range(len(beta)):
            delta = np.zeros(len(beta))
            delta[i] = h
            for j in range(len(beta)):
                gamma = np.zeros(len(beta))
                gamma[j] = h

                f1 = eval_nllh(beta + delta + gamma)
                f2 = eval_nllh(beta + delta - gamma)
                f3 = eval_nllh(beta - delta + gamma)
                f4 = eval_nllh(beta - delta - gamma)

                fd_approx = (f1 - f2 - f3 + f4) / (4 * h ** 2)
                self.assertAlmostEqual(hess[i, j], fd_approx, places=5)
