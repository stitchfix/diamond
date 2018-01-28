# Integration test
# simulate data according to
# logistic regression model GLMM
# fit priors using lme4
# learn parameters using diamond
# measure discrepancy between true parameters and diamond estimates
library(lme4)
library(logging); basicConfig()
# library(MASS)
# library(MCMCpack)
inv_logit <- function(x) 1 / (1 + exp(-x))


set.seed(893)

# sample cov matrix from inv wishart distribution
SIGMA <- MCMCpack::riwish(3, matrix(c(1,.3,.3,1),2,2))
# covariance matrix must be symmetric...
stopifnot(all(abs(t(SIGMA) - SIGMA) < 1e-8))
# ...and positive definite
stopifnot(det(SIGMA) > 0)

loginfo("Creating random coefficients")
ngroups <- 400
ranef <- MASS::mvrnorm(n=ngroups, mu=c(0, 0), Sigma=SIGMA)
fixef <- c(-1, 1)

# how many obs in each group
n <- round(rnorm(n=ngroups, mean=1000, sd=100))
n <- pmax(1, n)

loginfo("Creating 1 random feature")
X <- matrix(0, nrow=sum(n), ncol=length(fixef) * (ngroups + 1))
X[, 1] <- 1 # intercept
X[, 2] <- rnorm(n=nrow(X))

loginfo("Creating design matrix")
for (i in 1:ngroups) {
    row_start <- 1 + ifelse(i==1, 0, sum(n[1:(i-1)]))
    row_end <- row_start + n[i] - 1
    X[row_start:row_end, (2*i + 1) :(2*i + 2)] <- X[row_start:row_end, 1:2]
}

loginfo("With these, simulating binary responses")
vec_beta <- c(fixef, c(rbind(ranef[, 1], ranef[, 2])))
p <- inv_logit(X %*% vec_beta)
y <- rbinom(n=nrow(X), size=1, prob=p)

df <- data.frame("x"=X[, 2], "y"=y, "level"=rep(1:ngroups, n))
loginfo("Fitting model in lme4")
m <- lme4::glmer(y ~ 1 + x + (1 + x | level),
           data=df,
           family=binomial)
df_priors <- as.data.frame(VarCorr(m))
df_priors[['group']] <- df_priors[['grp']]
df_priors[c('sdcor', 'grp')] <- NULL
for (x in paste0("var", 1:2)) {
    df_priors[[x]] <- gsub("(Intercept)", "intercept", df_priors[[x]], fixed=TRUE)
}
# make sure columns are in the right order
df_priors <- df_priors[, c("group", "var1", "var2", "vcov")]

# assumes working directory is diamond/
write.csv(df, "diamond/integration_tests/logistic/simulated_logistic_df.csv", row.names=FALSE)
write.csv(vec_beta, "diamond/integration_tests/logistic/simulated_logistic_true_parameters.csv", row.names=FALSE)
write.csv(df_priors, "diamond/integration_tests/logistic/simulated_logistic_covariance.csv", row.names=FALSE)
