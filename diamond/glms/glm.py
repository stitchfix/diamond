import abc
import logging
import re
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from diamond.solvers.repeated_block_diag import RepeatedBlockDiagonal
from scipy import sparse
from future.utils import iteritems

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class GLM(object):
    """
    Binary or cumulative logistic regression model with arbitrary
    crossed random effects and known covariance
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, train_df, priors_df, copy=False, test_df=None):
        r""" Initialize a diamond model

        Args:
            train_df (DataFrame): DataFrame to estimate the model parameters
            priors_df (DataFrame): Covariance matrix to use for regularization.
                Format is | group | var1 | var2 | vcov |
                where group represents the grouping factor, var1 and var2
                    specify the row and column of the covariance matrix,
                and vcov is a scalar for that entry of the covariance matrix.
                    Note that if var2 is NULL, vcov is
                    interpreted as the diagonal element of the covariance
                    matrix for var1
            copy (boolean): Make a copy of train_df. If False, new columns
                will be created and the index will be reset.
            test_df (DataFrame): This is used to make predictions.
        Returns:
            Object (GLM)
        """
        self.solver = None  # solver will be set by child classes

        self.train_df = train_df.copy(deep=True) if copy else train_df
        self.test_df = test_df
        self.priors_df = priors_df
        self.variances = None

        # set by self.parse_formula
        self.response = None
        self.main_effects = []
        self.groupings = defaultdict(list)
        self.grouping_factors = []
        self.group_levels = {}
        self.num_levels = {}
        self.total_num_interactions = 0
        self.num_main = 0

        # set by self.create_penalty_matrix
        self.sparse_inv_covs = {}
        self.fit_order = []

        # set by self.create_design_matrix
        self.level_maps = {}
        self.main_map = None
        self.inter_maps = {}
        self.main_design = None
        self.grouping_designs = {}

        # set by self.fit
        self.fit_kwargs = None
        self.effects = {}
        self.results = None
        self.results_dict = {}
        self.start_time = None
        self.obj_fun = None

    def initialize(self, formula, **kwargs):
        r""" Get ready to fit the model by parsing the formula,
        checking priors, and creating design, penalty, and Hessian matrices

        Args:
            formula (string): R-style formula expressing the model to fit.
                eg. :math:`y \sim 1 + x + (1 + x | group)`
        Keyword Args:
            kwargs: additional arguments to pass to fit method
        """
        self.fit_kwargs = kwargs
        self.start_time = time.time()
        self._parse_formula(formula)
        self._check_priors_df()

        self._create_design_matrix()
        self._create_response_matrix()
        self._create_penalty_matrix()
        self._create_hessians()

    @abc.abstractmethod
    def _create_response_matrix(self):
        """ Must be implemented by subclasses """
        raise NotImplementedError

    @abc.abstractmethod
    def _create_hessians(self):
        """ Must be implemented by subclasses """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, formula, **kwargs):
        """ Must be implemented by subclasses """
        raise NotImplementedError

    @abc.abstractmethod
    def _create_output_dict(self):
        """ Must be implemented by subclasses """
        raise NotImplementedError

    def _check_priors_df(self):
        """Run simple validations on priors data frame

        This method runs a number of sanity checks on the priors data frame:
            - ensure that the expected columns are present
            - ensure that all rows in priors_df are consistent with the formula
            - ensure that all random effect variables at least have a variance

        Args:
            None

        Returns:
            None
        """
        _groupings = {g: f + [np.nan] for g, f in iteritems(self.groupings)}

        allowed_covs = {
            g: [[v, w] for i, v in enumerate(f)
                for j, w in enumerate(f)
                if j > i and not isinstance(v, float)]
            for g, f in iteritems(_groupings)
            }

        required_covs = {
            g: [[v, np.nan] for i, v in enumerate(f)
                if not isinstance(v, float)] for g, f in iteritems(_groupings)
            }

        example_priors_data = {
            "group": [g for g, f in iteritems(allowed_covs) for _ in f],
            "var1": [j[0] for _, f in iteritems(allowed_covs) for j in f],
            "var2": [j[1] for _, f in iteritems(allowed_covs) for j in f],
            "vcov": [0.1 for _, f in iteritems(allowed_covs) for _ in f]
        }
        example_priors_df = pd.DataFrame.from_dict(example_priors_data)

        expected_cols = set(["group", "var1", "var2", "vcov"])
        actual_cols = set(self.priors_df.columns)
        if len(expected_cols - actual_cols) != 0:
            raise AssertionError("""
                priors_df is expected to have the following columns:
                | group | var1 | var2 | vcov |
            """)

        # check that all rows in self.priors_df are valid
        for row in self.priors_df.iterrows():
            group = row[1].group
            var1 = row[1].var1
            var2 = row[1].var2

            # we need to standardize nan values
            if str(var2) in ['nan', 'None', 'null']:
                var2 = np.nan

            var_pair = [var1, var2]

            num_matches = sum([np.array_equal(var_pair, p)
                               for p in allowed_covs[group]])
            if num_matches != 1:
                raise AssertionError("""There is a row in your priors_df which is not expected.
                Unexpected row: {}
                A valid priors_df looks something like:
                {}

                """.format([group] + var_pair, example_priors_df))

            try:
                required_covs[group].remove(var_pair)
            except ValueError:
                # thrown when var_pair not in required_covs[group]
                pass

        # loop through the required_covs and make sure none are remaining
        remaining_required_covs = sum([len(f) for _, f in iteritems(required_covs)])
        if remaining_required_covs > 0:
            raise AssertionError("""Priors_df is missing some required rows.
            If you want an unregularized random effect, include it as a fixed
            effect instead.

            The missing rows are: {}

            """.format(required_covs))

    @staticmethod
    def _check_formula(formula):
        """
        Check that the formula contains all necessary ingredients,
            such as a response and at least one group
        Args:
            formula (string): R-style formula expressing the model to fit.
                eg. "y ~ 1 + x + (1 + x | group)"
        Returns:
            None
        """
        valid_formula_str = """
        A valid formula looks like:
            response ~ 1 + feature1 + feature2 + ... +
            (1 + feature1 + feature2 + ... | doctor_id)
        """
        if "~" not in formula:
            msg = "Formula missing '~'. You need a response. {}"
            raise AssertionError(msg.format(valid_formula_str))
        if "|" not in formula:
            msg = "Formula missing '|'. You need at least 1 group. {}"
            raise AssertionError(msg.format(valid_formula_str))

    def _parse_formula(self, formula):
        """
        Args:
            formula (string): R-style formula expressing the model to fit.
                eg. "y ~ 1 + x + (1 + x | group)"
        Returns:
            None
        """
        # strip all newlines, tabs, and spaces
        formula = re.sub(r'[\n\t ]', '', formula)

        # split the response from the formula terms
        self.response = formula.split("~")[0]
        terms = formula.split("~")[1:][0]

        interactions = re.findall(r'\(([A-Za-z0-9_|\+]+)\)', terms)
        # parse the interactions. these are terms like (1|doctor_id)
        for i in interactions:
            # remove interactions from terms list
            terms = terms.replace("(%s)" % i, "")
            i_terms = i.split('|')[0]
            i_group = i.split('|')[1:][0]
            for i_term in i_terms.split("+"):
                if i_term == "1":
                    self.groupings[i_group].append("intercept")
                elif i_term != '':
                    self.groupings[i_group].append(i_term)
            self.group_levels[i_group] = self.train_df[i_group].unique()
            self.grouping_factors.append(i_group)
        for g in self.grouping_factors:
            self.num_levels[g] = self.train_df[g].nunique()
            self.total_num_interactions += self.num_levels[g]
        # parse the main terms
        for m in terms.split("+"):
            if m == "1":
                self.main_effects.append("intercept")
            elif m != '':
                self.main_effects.append(m)
        self.num_main = len(self.main_effects)

    def _create_penalty_matrix(self):
        """
        Take the provided covariance matrices and transform it
            into an L2 penalty matrix
        Args:
            None
        Returns:
            None
        """
        self.variances = self.priors_df
        self.variances.ix[self.variances['var1'] == '(Intercept)', 'var1'] = 'intercept'
        LOGGER.info("creating covariance matrix")

        # if "group" is a column in the variances, rename it to "grp"
        self.variances.rename(columns={'grp': 'group'}, inplace=True)

        inv_covs = {}
        for g in self.groupings.keys():
            n = len(self.groupings[g])
            cov_mat = np.zeros((n, n))
            var_grp = self.variances[self.variances.group == g]
            if len(var_grp) > 0:  # if no priors, then leave cov_mat as zeros
                for row in var_grp[['var1', 'var2', 'vcov']].iterrows():
                    if str(row[1]['var2']) in ['nan', 'None', 'null']:
                        i = self.groupings[g].index(row[1]['var1'])
                        cov_mat[i, i] = row[1]['vcov']
                    else:
                        i = self.groupings[g].index(row[1]['var1'])
                        j = self.groupings[g].index(row[1]['var2'])
                        cov_mat[i, j] = row[1]['vcov']
                        cov_mat[j, i] = row[1]['vcov']
                inv_covs[g] = np.linalg.inv(cov_mat)
                self.sparse_inv_covs[g] = \
                    RepeatedBlockDiagonal(inv_covs[g], self.num_levels[g])
        self.sparse_inv_covs['main'] = None

    def _create_main_design(self, **kwargs):
        r"""
        Create design matrix for main effects
        Keyword Args:
            * *df* (``DataFrame``). specify a new dataframe to create
                design matrix from
        Returns:
            array_like: design matrix in sparse CSR format

        """
        df = kwargs.get('df', self.train_df)
        df.reset_index(drop=True, inplace=True)
        df['row_index'] = df.index
        df['intercept'] = 1.0  # assume intercept is always included

        id_cols = ['row_index']

        melted_df = pd.melt(df[id_cols + self.main_effects], id_cols)
        melted_df = melted_df.merge(self.main_map, on='variable')
        melted_df['col_index'] = melted_df['main_idx']
        row = melted_df.row_index
        col = melted_df.col_index
        data = melted_df.value
        return sparse.coo_matrix((data, (row, col)),
                                 shape=(max(row) + 1, max(col) + 1)).tocsr()

    def _create_inter_design(self, g, **kwargs):
        r"""
        Create random effects design matrix for grouping factor g
        This is straightforward when you create the matrix using the training
            DataFrame
        But a new DataFrame can have new levels of g which did not exist in
            training DF
        For these levels, the random coefficients are set to zero
        But as a practical matter, it's easier to zero out the values of the
            predictors
        here than it is to modify the fitted coefficient vector
        Args:
            g (string): grouping factor to create design matrix
        Keyword Args:
            * *df* (``DataFrame``). specify a new dataframe to create
                design matrix from
        Returns:
            array_like : design matrix in sparse CSR format
        """
        idx = g + '_idx'

        df = kwargs.get('df', self.train_df)
        df.reset_index(drop=True, inplace=True)
        df['row_index'] = df.index
        if 'intercept' in self.groupings[g]:
            df['intercept'] = 1.0

        id_cols = [g, 'row_index']

        # level_maps has levels of g and an index for each level
        melted_inter = pd.melt(df[id_cols + self.groupings[g]], id_cols).merge(
            self.level_maps[g], how='left', on=g).merge(
            self.inter_maps[g], how='inner', on='variable')
        # inter_maps has variables in formula for this grouping factor,
        # plus indexes
        # because of the above left join, some idx values are NULL
        # but we need to keep the row indexes
        nrows = max(melted_inter.row_index) + 1
        # now drop the null column index
        melted_inter.dropna(inplace=True)

        melted_inter.sort_values(by=[g, idx], inplace=True)
        melted_inter['col_index'] = melted_inter['inter_idx'] + \
            melted_inter[idx] * len(self.groupings[g])

        row = melted_inter.row_index
        col = melted_inter.col_index
        data = melted_inter.value

        if g in self.grouping_designs.keys():
            # this means training matrix was already created for this group
            # reuse the same shape: indices are lined up,
            # everything else will be 0 b/c of sparse matrix
            ncols = self.grouping_designs[g].shape[1]
        else:
            ncols = max(col) + 1

        return sparse.coo_matrix((data, (row, col)),
                                 shape=(nrows, ncols)).tocsr()

    def _create_design_matrix(self):
        """
        Create row index and dummy intercept column
        Args:
            None
        Returns:
            None
        """
        self.train_df['row_index'] = self.train_df.index

        # Create some dataframes to help with indexing
        for g in self.groupings.keys():
            idx = g + '_idx'
            self.level_maps[g] = pd.DataFrame({
                g: np.unique(self.train_df[g]),
                idx: np.arange(self.num_levels[g])})
            self.inter_maps[g] = pd.DataFrame({
                'variable': self.groupings[g],
                'inter_idx': np.arange(len(self.groupings[g]))})
        self.main_map = pd.DataFrame({
            'variable': self.main_effects,
            'main_idx': np.arange(len(self.main_effects))})

        LOGGER.info("creating main design matrix")
        # Compute indices (row_index, col_index) into sparse design matrix
        self.main_design = self._create_main_design()

        for g in self.groupings.keys():
            LOGGER.info("creating %s design matrix", g)
            self.grouping_designs[g] = self._create_inter_design(g)

        self.grouping_designs['main'] = self.main_design

    def _create_output_dict_inter(self):
        """ Format the estimated coefficients into a nice dictionary
        Args:
            None

        Returns:
            None
        """
        groupings_coefs = {}
        for g in self.groupings.keys():
            coefs = pd.DataFrame({'inter_value': self.effects[g]})
            idx = g + '_idx'
            # repeat each index self.groupings[g] times
            # e.g. [0, 0, 1, 1, ...]
            coefs[idx] = np.arange(len(coefs)) // len(self.groupings[g])
            coefs['inter_idx'] = np.mod(np.arange(len(coefs)),
                                        len(self.groupings[g]))
            coefs = coefs.merge(self.inter_maps[g], on='inter_idx').\
                merge(self.level_maps[g], on=idx)
            groupings_coefs[g] = coefs
            # make the individual entries in self.results_dict
            _df = coefs.pivot(index=g,
                              columns='variable',
                              values='inter_value').reset_index()
            del _df.index.name
            _df.columns.name = None
            self.results_dict[g] = _df

    def predict(self, new_df):
        r""" Obtain predictions from a fitted diamond object.
        New levels of grouping factors are given fixed effects,
        with zero random effects

        Args:
            new_df (DataFrame):  data to make predictions on

        Returns:
            array_like: predictions, whose length is equal to the \
            number of rows of the supplied DataFrame
        """
        if self.num_main > 0:
            X = self._create_main_design(df=new_df)
            predictions = X.dot(self.effects['main'])
        else:
            predictions = np.zeros(len(new_df.index))
        for g in self.groupings.keys():
            Z = self._create_inter_design(g, df=new_df)
            predictions += Z.dot(self.effects[g])
        return predictions
