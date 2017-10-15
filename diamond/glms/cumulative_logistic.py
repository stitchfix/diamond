import logging
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from diamond.glms.glm import GLM
from diamond.solvers.diamond_cumulative_logistic import \
        FixedHessianSolverCumulative

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class CumulativeLogisticRegression(GLM):
    """
    Cumulative logistic regression model
    with arbitrary crossed random effects and known covariances
    """

    def __init__(self, train_df, priors_df, copy=False, test_df=None):
        super(CumulativeLogisticRegression, self).__init__(train_df,
                                                           priors_df,
                                                           copy,
                                                           test_df)
        self.solver = FixedHessianSolverCumulative()
        self.H_main = None
        self.H_inter_LUs = {}
        self.J = None
        self.response = None  # n x J matrix of response counts

    def initialize(self, formula, **kwargs):
        r""" Get ready to fit the model by parsing the formula,
            checking priors, and creating design, penalty, and Hessian matrices

        Args:
            formula (string): R-style formula expressing the model to fit.
                eg. :math:`y \sim 1 + x + (1 + x | group)`
        Keyword Args:
            kwargs: additional arguments to pass to solver.fit method
        """
        super(CumulativeLogisticRegression, self).initialize(formula, **kwargs)
        # ordinal uses separate solvers for main/interaction effects
        # consequently, need to store main design matrix as its own object
        self.grouping_designs.pop('main', None)

    def fit(self, formula, **kwargs):
        r""" Fit the model specified by formula and training data

        Args:
            formula (string): R-style formula expressing the model to fit.
                eg. :math:`y \sim 1 + x + (1 + x | group)`
        Keyword Args:
            intercepts : array-like, optional. Initial values for intercepts. \
                Must be monotonically increasing and \
                have length == number of response levels minus one
            main : array-like, optional. Initial values for main effects. \
                Must have length == number of main effects specified in formula
            kwargs : additional arguments passed to solver.fit
        Returns:
            dict of parameter estimates with keys "intercepts", "main",
                and one key for each grouping factor
        """
        self.initialize(formula, **kwargs)

        # set initial parameters
        default_intercepts = np.linspace(-1.0, 1.0, self.J - 1)
        self.effects['intercepts'] = kwargs.get('intercepts',
                                                default_intercepts)
        self.effects['main'] = kwargs.get('main', np.zeros(self.num_main))
        for g in self.grouping_designs:
            self.effects[g] = np.zeros(self.grouping_designs[g].shape[1])
        self.effects = self.solver.fit(Y=self.response,
                                       main_design=self.main_design,
                                       inter_designs=self.grouping_designs,
                                       H_inter_LUs=self.H_inter_LUs,
                                       penalty_matrices=self.sparse_inv_covs,
                                       effects=self.effects,
                                       **kwargs)
        self._create_output_dict()
        return self.results_dict

    def _create_response_matrix(self):
        LOGGER.info("Creating response matrix.")
        df = pd.DataFrame({
            'index': self.train_df.index,
            'y': self.train_df[self.response]})
        Y = pd.pivot_table(df,
                           index='index',
                           columns=['y'],
                           aggfunc=len,
                           fill_value=0).as_matrix()
        self.response = Y
        self.J = self.response.shape[1]
        LOGGER.info("Created response matrix with shape (%d, %d)",
                    self.response.shape[0], self.response.shape[1])

    def _create_hessians(self):
        """ Create bounds on the Hessian matrices
        Args:
            None
        Returns:
            None
        """
        LOGGER.info("creating Hessians")
        for g in self.groupings.keys():
            X = self.grouping_designs[g]
            H = 0.5 * X.transpose().dot(X) + \
                self.sparse_inv_covs[g].sparse_matrix
            self.H_inter_LUs[g] = sparse.linalg.splu(H.tocsc())

    def _create_output_dict(self):
        """ Extract coefficients from fitted model
        Args:
            None
        Returns:
            dictionary with keys "main", "intercepts",
                and one key for each grouping factor. \
            Values of the dictionary are dataframes
            """
        LOGGER.info("extracting coefficients")
        self.results_dict['intercepts'] = self.effects['intercepts']
        self.results_dict['main'] = pd.DataFrame({
            'variable': self.main_effects,
            'main_value': self.effects['main']})
        self._create_output_dict_inter()

    def predict(self, new_df):
        """ Use the estimated model to make predictions. \
        New levels of grouping factors are given fixed effects,
            with zero random effects

        Args:
            new_df (DataFrame):  data to make predictions on
        Returns:
            n x J matrix, where n is the number of rows \
            of new_df and J is the number \
            of possible response values. The (i, j) entry of \
           this matrix is the probability that observation i \
            realizes response level j.
        """
        eta = super(CumulativeLogisticRegression, self).predict(new_df)
        intercepts = self.effects['intercepts']
        J = self.J
        preds = np.zeros((len(eta), J))
        preds[:, 0] = expit(intercepts[0] + eta)
        preds[:, J - 1] = 1.0 - expit(intercepts[J - 2] + eta)
        for j in range(1, J - 1):
            preds[:, j] = expit(intercepts[j] + eta) - \
                expit(intercepts[j - 1] + eta)
        return preds
