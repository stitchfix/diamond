import unittest
import numpy as np
import pandas as pd
from diamond.glms.logistic import LogisticRegression
import os
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class TestLogistic(unittest.TestCase):
    def test_setUp(self, tol=0.02):
        # assumes working directory is diamond/
        folder = "diamond/integration_tests/logistic"

        simulated_data_loc = "%s/simulated_logistic_df.csv" % folder
        estimated_covariance_loc = "%s/simulated_logistic_covariance.csv" % folder
        resources_exist = os.path.exists(simulated_data_loc) and os.path.exists(estimated_covariance_loc)
        if not resources_exist:
            logging.info("Simulating data and estimating covariances in R")
            os.system("/usr/local/bin/Rscript %s/logistic_generate_and_fit.R" % folder)
        logging.info("Reading in training data and R::lme4-estimated covariance matrix")
        df_train = pd.read_csv(simulated_data_loc)
        df_estimated_covariance = pd.read_csv(estimated_covariance_loc)

        self.model = LogisticRegression(train_df=df_train,
                                        priors_df=df_estimated_covariance,
                                        copy=True,
                                        test_df=None)
        logging.info("Fitting model in diamond")
        self.formula = "y ~ 1 + x + (1 + x | level)"
        results = self.model.fit(self.formula, tol=1e-4, verbose=True)

        # the format of the coefficient vector is:
        # fixed effects, then [random intercept, random slope] for each level
        beta_hat = np.append(results["fixed_effects"].value.values,
                             pd.melt(results["level"], "level").sort_values(["level", "variable"]).value.values)

        beta_true = pd.read_csv("%s/simulated_logistic_true_parameters.csv" % folder)["x"].values
        rel_error = np.mean((beta_hat - beta_true) ** 2) / np.mean(abs(beta_true))
        if rel_error > tol:
            logging.warn("relative error = %f > tolerance = %f" % (rel_error, tol))
        else:
            logging.info("relative error = %f < tolerance = %f" % (rel_error, tol))
        # make sure the coefficients are very close
        self.assertTrue(rel_error < tol)


if __name__ == '__main__':
    unittest.main()
