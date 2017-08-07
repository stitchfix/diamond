import unittest
import numpy as np
import pandas as pd
from diamond.glms.cumulative_logistic import CumulativeLogisticRegression
import os
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class TestCumulativeLogistic(unittest.TestCase):
    def test_setUp(self, tol=0.02):
        # assumes working directory is diamond/
        folder = "diamond/integration_tests/clogistic"
        simulated_data_loc = "%s/simulated_clogistic_df.csv" % folder
        estimated_covariance_loc = "%s/simulated_clogistic_covariance.csv" % folder
        resources_exist = os.path.exists(simulated_data_loc) and os.path.exists(estimated_covariance_loc)
        if not resources_exist:
            logging.info("Simulating data and estimating covariances in R")
            os.system("/usr/local/bin/Rscript %s/clogistic_generate_and_fit.R" % folder)
        logging.info("Reading in training data and R::ordinal-estimated covariance matrix")

        df_train = pd.read_csv(simulated_data_loc)
        df_estimated_covariance = pd.read_csv(estimated_covariance_loc)

        self.formula = "y ~ x + (1 + x | level)"

        self.model = CumulativeLogisticRegression(train_df=df_train,
                                                  priors_df=df_estimated_covariance,
                                                  copy=True,
                                                  test_df=None)
        logging.info("Fitting model in diamond")
        results = self.model.fit(self.formula, tol=1e-3, max_its=5, verbose=True)

        # the format of the coefficient vector is:
        # fixed effects, then [random intercept, random slope] for each level
        beta_hat = np.append(results["main"]["main_value"].values,
                             pd.melt(results["level"], "level").sort_values(["level", "variable"]).value.values)

        # drop the 0 value at the head of beta_true
        # this is a placeholder, which reflects that there is no fixed intercept in this model
        beta_true = pd.read_csv("%s/simulated_clogistic_true_parameters.csv" % folder)["x"].values[1:]
        rel_error = np.mean((beta_hat - beta_true) ** 2) / np.mean(abs(beta_true))
        if rel_error > tol:
            logging.warn("relative error of coefs = %f > tolerance = %f" % (rel_error, tol))
        else:
            logging.info("relative error of coefs = %f < tolerance = %f" % (rel_error, tol))
        # make sure the coefficients are very close
        self.assertTrue(rel_error < tol)

        # check intercepts, too
        alpha_true = pd.read_csv("%s/simulated_clogistic_true_intercepts.csv" % folder).ix[1:3, "x"].values
        alpha_hat = results["intercepts"]
        rel_error_alpha = np.mean((alpha_hat - alpha_true) ** 2) / np.mean(abs(alpha_true))

        if rel_error_alpha > tol:
            logging.warn("relative error of intercepts = %f > tolerance = %f" % (rel_error_alpha, tol))
        else:
            logging.info("relative error of intercepts = %f < tolerance = %f" % (rel_error_alpha, tol))
        self.assertTrue(rel_error_alpha < tol)


if __name__ == '__main__':
    unittest.main()
