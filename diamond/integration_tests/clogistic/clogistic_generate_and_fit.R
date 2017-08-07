# Integration test
# simulate data according to
# cumulative logit  model
# Agresti, 2nd Ed, p 275, eqn (7.5) 
# fit priors using ordinal
# learn parameters using diamond
# measure discrepancy between true parameters and diamond estimates
library(ordinal)
library(dplyr)
library(logging); basicConfig()


set.seed(636)

# sample cov matrix from inv wishart distribution
SIGMA <- MCMCpack::riwish(3, matrix(c(1,.3,.3,1),2,2))
# covariance matrix must be symmetric...
stopifnot(all(abs(t(SIGMA) - SIGMA) < 1e-8))
# ...and positive definite
stopifnot(det(SIGMA) > 0)

loginfo("Creating random coefficients")
ngroups <- 400
ranef <- MASS::mvrnorm(n=ngroups, mu=c(0, 0), Sigma=SIGMA)
fixef <- rnorm(1)

# how many obs in each group
n <- round(rnorm(n=ngroups, mean=1000, sd=300))
n <- pmax(1, n)

loginfo("Creating 1 random feature")
# each group has a random intercept and slope
# and there is an overall fixed-effect slope
X <- matrix(0, nrow=sum(n), ncol= 2 * (ngroups + 1))
X[, 1] <- 1  # for random intercepts
X[, 2] <- rnorm(n=nrow(X))  # for random slopes

loginfo("Creating design matrix")
for (i in 1:ngroups) {
    row_start <- 1 + ifelse(i==1, 0, sum(n[1:(i-1)]))
    row_end <- row_start + n[i] - 1 
    X[row_start:row_end, (2*i + 1) :(2*i + 2)] <- X[row_start:row_end, 1:2]
}

loginfo("With these, simulating binary responses")
# there is fixed intercept in this model
vec_beta <- c(0, fixef, c(rbind(ranef[, 1], ranef[, 2])))
# latent variable interpretation
error <- rlogis(nrow(X))
y_star <- -1 * X %*% vec_beta + error

# K groups => K-1 intercepts
K <- 4
intercepts <- seq(-2.5, 2.5, length.out=K-1)
intercepts <- c(-Inf, intercepts, +Inf)
y <- cut(y_star, intercepts, labels=1:K)
# make sure this worked as intended:
# lapply(split(y_star, y), range)

df <- data.frame("x"=X[, 2], "y"=y, "level"=rep(1:ngroups, n))
loginfo("Fitting model using `ordinal` package")
m <- ordinal::clmm(y ~ x + (1 + x | level), data=df)

df_priors <- VarCorr(m)[[1]]
colnames(df_priors) <- gsub("(Intercept)", "intercept", colnames(df_priors), fixed=TRUE)
rownames(df_priors) <- gsub("(Intercept)", "intercept", rownames(df_priors), fixed=TRUE)

df_priors <- reshape2::melt(df_priors) %>%
             mutate(group="level") %>%
             rename_("vcov"="value")
colnames(df_priors) <- tolower(colnames(df_priors))
df_priors[df_priors[["var1"]] == df_priors[["var2"]], "var2"] <- NA
df_priors <- df_priors[-2, c("group", "var1", "var2", "vcov")]

# assumes working directory is diamond/
write.csv(df, "diamond/integration_tests/clogistic/simulated_clogistic_df.csv", row.names=FALSE)
write.csv(vec_beta, "diamond/integration_tests/clogistic/simulated_clogistic_true_parameters.csv", row.names=FALSE)
write.csv(intercepts, "diamond/integration_tests/clogistic/simulated_clogistic_true_intercepts.csv", row.names=FALSE)
write.csv(df_priors, "diamond/integration_tests/clogistic/simulated_clogistic_covariance.csv", row.names=FALSE)               
