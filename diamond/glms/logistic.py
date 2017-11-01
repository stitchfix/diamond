import logging
import time
import numpy as np
import pandas as pd
from diamond.glms.glm import GLM
from diamond.solvers.diamond_logistic import FixedHessianSolverMulti
from diamond.solvers.utils import custom_block_diag
from scipy.special import expit

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class LogisticRegression(GLM):
    """
    Logistic regression model
    with arbitrary crossed random effects and known covariances
    """

    def __init__(self, train_df, priors_df, copy=False, test_df=None):
        super(LogisticRegression, self).__init__(train_df,
                                                 priors_df,
                                                 copy,
                                                 test_df)
        self.solver = FixedHessianSolverMulti()
        self.H_main_inv = None
        self.H_invs = {}

    def _create_response_matrix(self):
        """ Store vector of binary responses in an array """
        self.response = self.train_df[self.response]

    def _create_hessians(self):
        """ Create bounds on the Hessian matrices
        Args:
            None
        Returns:
            None
        """
        LOGGER.info("creating Hessians")
        H_main = np.array(self.main_design.T.dot(self.main_design).todense())
        H_main *= 0.25
        # do a pseudoinverse for the main effects because they are
        # unregularized and could be constant columns
        self.H_main_inv = np.linalg.pinv(H_main)

        for g in self.groupings.keys():
            LOGGER.info("creating H_inter for %s", g)
            inter_design = self.grouping_designs[g]
            H_inter = 0.25 * inter_design.T.dot(inter_design)

            q = len(self.level_maps[g])
            inv_blocks = []
            block_length = H_inter.shape[0] // q

            for k in range(q):
                if k % 100000 == 0:
                    LOGGER.info("time elapsed: %.1f",
                                time.time() - self.start_time)
                    LOGGER.info("blocks inverted: %i of %i", k, q)
                block = H_inter[(k * block_length):((k + 1) * block_length),
                                (k * block_length):((k + 1) * block_length)]
                block = np.array(block.todense())
                iblk = np.linalg.inv(block + self.sparse_inv_covs[g]._block)
                inv_blocks.append(iblk)

            LOGGER.info("creating H_invs")

            self.H_invs[g] = custom_block_diag(inv_blocks)

        self.H_invs['main'] = self.H_main_inv

    def fit(self, formula, **kwargs):
        r""" Fit the model specified by formula and training data

        Args:
            formula (string): R-style formula expressing the model to fit.
                eg. :math:`y \sim 1 + x + (1 + x | group)`
        Keyword Args:
            kwargs : additional arguments passed to solver
        Returns:
            dict of estimated parameters
        """
        self.initialize(formula, **kwargs)
        self.effects, self.obj_fun = self.solver.fit(self.response,
                                                     self.grouping_designs,
                                                     self.H_invs,
                                                     self.sparse_inv_covs,
                                                     **kwargs)
        self._create_output_dict()
        return self.results_dict

    def refit(self, **kwargs):
        r""" Update a fitted model. Any kwargs not supplied will be reused
            from original call to self.fit()

        Args:
            None
        Keyword Args:
            kwargs: fitting parameters
        Returns:
            None
        """
        kwargs = {x: kwargs.get(x, self.fit_kwargs.get(x, None))
                  for x in set(kwargs.keys()).union(self.fit_kwargs.keys())}

        self.effects, self.obj_fun = self.solver.fit(self.response,
                                                     self.grouping_designs,
                                                     self.H_invs,
                                                     self.sparse_inv_covs,
                                                     **kwargs)
        self._create_output_dict()
        return self.results_dict

    def _create_output_dict(self):
        """ Extract coefficients
        Args:
            None
        Returns:
            Dictionary of estimated coefficients. Keys are "fixed_effects"
                and one key for each grouping factor
        """
        LOGGER.info("extracting coefficients")
        main_coefs = pd.DataFrame({'variable': self.main_effects,
                                   'value': self.effects["main"]})

        self.results_dict['fixed_effects'] = main_coefs
        self._create_output_dict_inter()

    def _merge_effects(self):
        """ Add together estimated fixed and random coefficients
        Args:
            None
        Returns:
            None
        """
        LOGGER.info("extracting coefficients")
        main_coefs = pd.DataFrame({'variable': self.main_effects,
                                   'value': self.effects["main"]})

        groupings_coefs = {}
        for g in self.groupings.keys():
            coefs = pd.DataFrame({'inter_value': self.effects[g]})

            idx = g + '_idx'
            coefs[idx] = np.arange(len(coefs)) / len(self.groupings[g])
            coefs['inter_idx'] = np.mod(np.arange(len(coefs)),
                                        len(self.groupings[g]))

            coefs = coefs.merge(self.inter_maps[g], on='inter_idx').\
                merge(self.level_maps[g], on=idx)
            groupings_coefs[g] = coefs

        # start merging all of the coefficients together
        merged_coefs = main_coefs
        for g in groupings_coefs:
            merged_coefs = merged_coefs.merge(groupings_coefs[g],
                                              on='variable',
                                              how='right',
                                              suffixes=('_1', '_2'))
            try:
                merged_coefs['value'] = merged_coefs['value'] + \
                        merged_coefs['inter_value']
            except KeyError:
                merged_coefs['value'] = merged_coefs['value'] + \
                        merged_coefs['inter_value_2']

        # do some wrangling to get the pivot to work correctly
        cols = self.grouping_factors
        cols.extend(['variable', 'value'])
        merged_coefs = merged_coefs[cols]
        index = range(np.product(self.num_levels.values())) * \
            merged_coefs.variable.nunique()
        merged_coefs['index'] = index

        pivoted_merged_coefs = merged_coefs.pivot(index='index',
                                                  columns='variable',
                                                  values='value').reset_index()
        del pivoted_merged_coefs.index.name
        pivoted_merged_coefs.columns.name = None
        self.results = pivoted_merged_coefs.merge(merged_coefs,
                                                  on='index',
                                                  how='left').drop(
            ['variable', 'value', 'index'], axis=1)

        # Add in columns that are main effects but not interaction effects
        inter_cols = self.groupings.values()[0]
        main_cols = main_coefs['variable'].unique()
        vars_to_add = list(set(main_cols) - set(inter_cols))
        for v in vars_to_add:
            LOGGER.info("%s is a main effect but not an interaction effect." +
                        "Adding it to the results now", v)
            self.results[v] = main_coefs.value[main_coefs['variable'] == v]
            self.results[v] = self.results[v].astype(float)

    def predict(self, new_df):
        """ Use estimated coefficients to make predictions on new data

        Args:
            new_df (DataFrame). DataFrame to make predictions on.
        Returns:
            array-like. Predictions on the response scale, i.e. probabilities
        """
        return expit(super(LogisticRegression, self).predict(new_df))
