"""
Diamond GLM tests
"""
import unittest

import numpy as np
import pandas as pd
from diamond.glms.logistic import LogisticRegression


class TestGLM(unittest.TestCase):
    def setUp(self):
        data = {"response": [0, 1, 1],
                "var_a": [21, 32, 10],
                "cyl": [4, 6, 4]}
        df = pd.DataFrame(data, index=[0, 1, 2])

        priors_data = {
            "grp": ["cyl", "cyl", "cyl"],
            "var1": ["intercept", "intercept", "var_a"],
            "var2": [np.NaN, "var_a", np.NaN],
            "vcov": [0.123, -1.42, 0.998]
        }
        priors_df = pd.DataFrame(priors_data, index=[0, 1, 2])

        self.formula = "response ~ 1 + var_a + (1 + var_a | cyl)"

        self.model = LogisticRegression(train_df=df,
                                        priors_df=priors_df,
                                        test_df=None)

    def test_parse_formula(self):
        self.model._parse_formula(self.formula)

        self.assertEqual(self.model.num_main, 2)
        self.assertEqual(self.model.response, "response")
        self.assertListEqual(self.model.main_effects, ["intercept", "var_a"])
        self.assertEqual(self.model.total_num_interactions,
                         self.model.train_df.cyl.nunique())
        self.assertListEqual(self.model.grouping_factors, ["cyl"])
        self.assertListEqual(list(self.model.group_levels.keys()),
                             ["cyl"])
        self.assertListEqual(list(self.model.group_levels["cyl"]),
                             [4, 6])

    def test_create_penalty_matrix(self):
        self.model._parse_formula(self.formula)
        self.model._create_penalty_matrix()

        expected_inv_cov_block = np.linalg.inv([[0.123, -1.42],
                                               [-1.42, 0.998]])
        actual_inv_cov_block = self.model.sparse_inv_covs["cyl"]._block

        self.assertListEqual(sorted(list(self.model.sparse_inv_covs.keys())),
                             ["cyl", "main"])
        self.assertEqual(self.model.sparse_inv_covs["cyl"]._num_blocks, 2)
        self.assertEqual(self.model.sparse_inv_covs["cyl"]._block_shape, 2)
        self.assertTrue((expected_inv_cov_block == actual_inv_cov_block).all())

    def test_create_main_design(self):
        self.model._parse_formula(self.formula)
        self.model._create_design_matrix()

        expected_design = [[1, float(row[1].var_a)]
                           for row in self.model.train_df.iterrows()]
        actual_design = self.model._create_main_design()

        self.assertEqual(actual_design.shape,
                         (len(self.model.train_df),
                          self.model.train_df.cyl.nunique()))
        self.assertTrue((expected_design == actual_design.todense()).all())

    def test_create_inter_design(self):
        self.model._parse_formula(self.formula)
        self.model._create_design_matrix()

        expected_design = [[1, float(row[1].var_a), 0, 0] if
                           row[1].cyl == 4 else [0, 0, 1, float(row[1].var_a)]
                           for row in self.model.train_df.iterrows()]
        actual_design = self.model._create_inter_design(g="cyl")

        # shape is (num variables * num levels, num observations)
        self.assertEqual(actual_design.shape,
                         (len(self.model.train_df),
                             2 * self.model.train_df.cyl.nunique()))
        self.assertTrue((expected_design == actual_design.todense()).all())


if __name__ == '__main__':
    unittest.main()
