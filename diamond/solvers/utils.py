""" Helper functions for diamond solvers.
Mostly linear algebra/gradients/Hessians.
Includes logistic and cumulative logistic functions
"""
from scipy import sparse
from scipy.special import expit
from scipy.sparse.linalg import spsolve
import numpy as np

from diamond.solvers.repeated_block_diag import RepeatedBlockDiagonal


def dot(x, y):
    """A generic dot product for sparse and dense vectors

    Args:
        x: array_like, possibly sparse
        y: array_like, possibly sparse
    Returns:
        dot product of x and y
    """
    if sparse.issparse(x):
        return np.array(x.dot(y))
    elif isinstance(x, RepeatedBlockDiagonal):
        return np.array(x.dot(y))
    return np.dot(x, y)


def solve(A, b):
    """
    A generic linear solver for sparse and dense matrices

    Args:
        A : array_like, possibly sparse
        b : array_like
    Returns:
        x such that Ax = b
    """
    if sparse.issparse(A):
        return spsolve(A, b)
    else:
        return np.linalg.solve(A, b)


def l2_logistic_fixed_hessian(X,
                              Y,
                              fixed_hess_inv,
                              penaltysq=None,
                              min_its=2,
                              max_its=200,
                              beta=None,
                              offset=None,
                              tol=1e-3):
    """Fit a l2-regularized logistic regression with a fixed-Hessian Newtonish method.

    Args:
        X : array_like. Design matrix
        Y : array_like. Vector of responses with values in {0, 1}
        fixed_hess_inv : sparse penalty matrix
        penalty_sq : array_like. Square penalty matrix
        min_its : int. Minimum number of iterations
        max_its : int. Maximum number of iterations
        beta : array_like. Vector of initial parameters, with length == number of columns of `X`
        offset: array_like. Added to X * beta. Defaults to 0
        tol : float. Convergence tolerance on relative change in `beta`
    Returns:
        array_like. Estimated parameters
    """

    if beta is None:
        beta = np.zeros(X.shape[1])

    if offset is None:
        offset = np.zeros(X.shape[0])

    if penaltysq is None:
        penaltysq = np.zeros((X.shape[1], X.shape[1]))

    for i in range(max_its):

        old_beta = 1.0 * beta

        # find the gradient
        p = 1.0 / (1 + np.exp(-(dot(X, beta) + offset)))
        grad = dot(X.T, Y - p) - dot(penaltysq, beta)

        # find a step direction (with the inverse of the fixed Hessian)
        step = dot(fixed_hess_inv, grad)

        # find a true Newton step in the chosen direction
        newton_num = dot(grad, step)
        newton_den = np.sum(dot(penaltysq, step) ** 2) + dot(p * (1 - p), dot(X, step) ** 2)
        step *= (newton_num / newton_den)

        # just take the step already!
        beta += step

        change = np.linalg.norm(beta - old_beta) / np.linalg.norm(beta)
        if change < tol and i > min_its:
            break
    return beta


def custom_block_diag(blocks):
    """ create csr sparse block diagonal matrix from identically-sized blocks

    Blocks don't need to be identical, but they do need to be the same shape.
    """
    L = len(blocks)
    K = blocks[0].shape[0]

    _data = [x.flatten() for x in blocks]
    m = np.arange(_data[0].shape[0])

    flat_data = np.zeros(L * len(m))
    for n in range(L):
        flat_data[m + n * len(m)] = _data[n][m]

    # now make the block diagonal array
    i = np.repeat(np.arange(L * K), K)
    j = np.tile(np.tile(np.arange(K), K), L) + np.repeat(np.arange(0, L * K, K), K * K)

    return sparse.csr_matrix((flat_data, (i, j)), shape=(L * K, L * K))


def l2_clogistic_llh(X, Y, alpha, beta, penalty_matrix, offset):
    """ Penalized log likelihood function for proportional odds cumulative logit model

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        alpha : array_like. intercepts.\
        must have shape == one less than the number of columns of `Y`
        beta : array_like. parameters.\
        must have shape == number of columns of X
        penalty_matrix : array_like. Regularization matrix
        offset : array_like, optional. Defaults to 0
    Returns:
        scalar : penalized loglikelihood
    """
    offset = 0.0 if offset is None else offset
    obj = 0.0
    J = Y.shape[1]
    Xb = dot(X, beta) + offset
    for j in range(J):
        if j == 0:
            obj += dot(np.log(expit(alpha[j] + Xb)), Y[:, j])
        elif j == J - 1:
            obj += dot(np.log(1 - expit(alpha[j - 1] + Xb)), Y[:, j])
        else:
            obj += dot(np.log(expit(alpha[j] + Xb) - expit(alpha[j - 1] + Xb)), Y[:, j])
    obj -= 0.5 * dot(beta, dot(penalty_matrix, beta))
    return -np.inf if np.isnan(obj) else obj


def l2_clogistic_gradient(X, Y, intercept=True, **kwargs):
    """ Gradient of cumulative logit model

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        intercept : boolean. If intercept=False, just calculate the gradient for the coefficients
        offset : array_like, optional. Defaults to 0
        alpha : array_like, optional. intercepts.\
        must have shape == one less than the number of columns of `Y`
        beta : array_like, optional. parameters.\
        must have shape == number of columns of X
        penalty_matrix : array_like, optional. Regularization matrix

    Returns:
        dense array of the GRADIENT of the NEGATIVE, penalized loglikelihood function
    """
    if X is not None:
        # X can be None if there are no fixed effects - i.e. just fit the intercept
        p = X.shape[1]
        beta = kwargs.get('beta', np.zeros(X.shape[1]))
        offset = kwargs.get('offset', np.zeros(X.shape[0]))
    else:
        p = 0
        beta = None
        offset = None
    J = Y.shape[1]
    alpha = kwargs.get('alpha', np.linspace(-1.0, 1.0, J - 1))
    penalty_matrix = kwargs.get('penalty_matrix', sparse.csr_matrix((p, p)))

    IL = _l2_clogistic_gradient_IL(X=X,
                                   alpha=alpha,
                                   beta=beta,
                                   offset=offset,
                                   n=Y.shape[0])
    if X is None:
        grad_beta = None
    else:
        grad_beta = _l2_clogistic_gradient_slope(X=X,
                                                 Y=Y,
                                                 IL=IL,
                                                 beta=beta,
                                                 penalty_matrix=penalty_matrix)
    if intercept:
        grad_alpha = _l2_clogistic_gradient_intercept(IL, Y, alpha)
        # MINIMIZE NEGATIVE LOGLIKELIHOOD
        if X is not None:
            return -1.0 * np.concatenate([grad_alpha, grad_beta])
        else:
            return -1.0 * grad_alpha
    else:
        return -1.0 * grad_beta


def _l2_clogistic_gradient_IL(X, alpha, beta, offset=None, **kwargs):
    """ Helper function for calculating the cumulative logistic gradient. \
        The inverse logit of alpha[j + X*beta] is \
        ubiquitous in gradient and Hessian calculations \
        so it's more efficient to calculate it once and \
        pass it around as a parameter than to recompute it every time

    Args:
        X : array_like. design matrix
        alpha : array_like. intercepts. must have shape == one less than the number of columns of `Y`
        beta : array_like. parameters. must have shape == number of columns of X
        offset : array_like, optional. Defaults to 0
        n : int, optional.\
        You must specify the number of rows if there are no main effects
    Returns:
        array_like. n x J-1 matrix where entry i,j is the inverse logit of (alpha[j] + X[i, :] * beta)
    """
    J = len(alpha) + 1
    if X is None:
        n = kwargs.get("n")
    else:
        n = X.shape[0]
    if X is None or beta is None:
        Xb = 0.
    else:
        Xb = dot(X, beta) + (0 if offset is None else offset)
    IL = np.zeros((n, J - 1))
    for j in range(J - 1):
        IL[:, j] = expit(alpha[j] + Xb)
    return IL


def _l2_clogistic_gradient_intercept(IL, Y, alpha):
    """ Gradient of penalized loglikelihood with respect to the intercept parameters

    Args:
        IL : array_like. See _l2_clogistic_gradient_IL
        Y : array_like. response matrix
        alpha : array_like. intercepts. must have shape == one less than the number of columns of `Y`
    Returns:
        array_like : length J-1
    """
    exp_int = np.exp(alpha)
    grad_alpha = np.zeros(len(alpha))
    J = len(alpha) + 1
    for j in range(J - 1):  # intercepts
        # there are J levels, and J-1 intercepts
        # indexed from 0 to J-2
        if j == 0:
            grad_alpha[j] = dot(Y[:, j], 1 - IL[:, j]) -\
                            dot(Y[:, j + 1], exp_int[j] / (exp_int[j + 1] - exp_int[j]) + IL[:, j])
        elif j < J - 2:
            grad_alpha[j] = dot(Y[:, j], exp_int[j] / (exp_int[j] - exp_int[j - 1]) - IL[:, j]) - \
                            dot(Y[:, j + 1], exp_int[j] / (exp_int[j + 1] - exp_int[j]) + IL[:, j])
        else:  # j == J-2. the last intercept
            grad_alpha[j] = dot(Y[:, j], exp_int[j] / (exp_int[j] - exp_int[j - 1]) - IL[:, j]) - \
                            dot(Y[:, j + 1], IL[:, j])
    return grad_alpha


def _l2_clogistic_gradient_slope(X, Y, IL, beta, penalty_matrix):
    """ Gradient of penalized loglikelihood with respect to the slope parameters

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        IL : array_like. See _l2_clogistic_gradient_IL
        beta : array_like. parameters. must have shape == number of columns of X
        penalty_matrix : array_like. penalty matrix
    Returns:
        array_like : same length as `beta`.
    """
    grad_beta = np.zeros(len(beta))
    J = Y.shape[1]
    XT = X.transpose()  # CSC format
    for j in range(J):  # coefficients
        if j == 0:
            grad_beta = dot(XT, Y[:, j] * (1.0 - IL[:, j]))
        elif j < J - 1:
            grad_beta += dot(XT, Y[:, j] * (1.0 - IL[:, j] - IL[:, j - 1]))
        else:  # j == J-1. this is the highest level of response
            grad_beta -= dot(XT, Y[:, j] * IL[:, j - 1])
    grad_beta -= dot(penalty_matrix, beta)
    return grad_beta


def l2_clogistic_hessian(X, Y, intercept=True, **kwargs):
    """ Hessian matrix of proportional odds cumulative logit model

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        intercept : boolean. If intercept=False,\
            just calculate the gradient for the coefficients
        alpha : array_like, optional.intercepts.\
            must have shape == one less than the number of columns of `Y`
        beta : array_like, optional. parameters.\
            must have shape == number of columns of X
        offset : array_like, optional. Defaults to 0
        penalty_matrix : array_like, optional.
    Returns:
        Square matrix of dimension == number of columns of `X`
    """
    if X is not None:
        # this is the case if there are no fixed effects, i.e. just an intercept
        p = X.shape[1]
        beta = kwargs.get('beta', np.zeros(X.shape[1]))
        offset = kwargs.get('offset', np.zeros(X.shape[0]))
    else:
        p = 0
        beta = None
        offset = None

    J = Y.shape[1]

    # default arguments
    alpha = kwargs.get('alpha', np.linspace(-1.0, 1.0, J - 1))
    penalty_matrix = kwargs.get('penalty_matrix', sparse.csr_matrix((p, p)))

    ILL = _l2_clogistic_hessian_ILL(X, alpha, beta, offset=offset, n=Y.shape[0])
    if X is None:
        hess_beta = None
    else:
        hess_beta = _l2_clogistic_hessian_slope(X=X,
                                                Y=Y,
                                                ILL=ILL,
                                                penalty_matrix=penalty_matrix,
                                                value=True)

    # MINIMIZE NEGATIVE LOGLIKELIHOOD
    if intercept:
        hess_alpha = -1.0 * _l2_clogistic_hessian_intercept(X, Y, ILL, alpha)
        if X is None:
            return hess_alpha
        if sparse.issparse(hess_beta):
            hess_beta = hess_beta.todense()
        hess_alpha[J - 1:, J - 1:] = hess_beta
        return hess_alpha
    else:
        return hess_beta


def _l2_clogistic_hessian_ILL(X, alpha, beta, **kwargs):
    """ this is the derivative of the `IL` term

    Args:
        X : array_like. design matrix
        alpha : array_like. intercepts.\
        must have shape == one less than the number of columns of `Y`
        beta : array_like. parameters. \
        must have shape == number of columns of X
        offset : array_like, optional. Defaults to 0
        n : int, optional. You must specify the number of rows if there are no main effects
    Returns:
        array_like. n x J-1 matrix containing terms for Hessian calculation
    """
    offset = kwargs.get('offset')
    if X is None:
        n = kwargs.get('n')
    else:
        n = X.shape[0]
    if X is None or beta is None:
        Xb = np.zeros(n)
    else:
        Xb = dot(X, beta) + (0 if offset is None else offset)
    ILL = np.zeros((X.shape[0], len(alpha)))
    for j in range(len(alpha)):
        ILL[:, j] = np.exp(alpha[j] + Xb) * (1.0 + np.exp(alpha[j] + Xb)) ** -2.0
    return ILL


def _l2_clogistic_hessian_intercept(X, Y, ILL, alpha):
    """ second derivatives of intercepts

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        ILL : array_like. See _l2_clogistic_hessian_ILL
        alpha : array_like. intercepts.\
        must have shape == one less than the number of columns of `Y`
    Returns:
        array_like. Matrix of second derivatives of main effects and intercepts
    """
    exp_int = np.exp(alpha)
    J = len(alpha) + 1
    p = X.shape[1] if X is not None else 0
    hess_alpha = np.zeros((p + J - 1, p + J - 1))

    # important to initialize hess to 0. this covers the non-adjacent intercepts
    for j in range(J - 1):
        if j == 0:
            hess_alpha[j, j] = -dot(Y[:, j], ILL[:, j]) -\
                               np.dot(Y[:, j + 1], (exp_int[j] * exp_int[j + 1]) / (exp_int[j + 1] - exp_int[j]) ** 2.0 + ILL[:, j])
        elif j < J - 2:
            hess_alpha[j, j] = -dot(Y[:, j], (exp_int[j] * exp_int[j - 1]) / (exp_int[j] - exp_int[j - 1]) ** 2.0 + ILL[:, j]) - \
                               dot(Y[:, j + 1], (exp_int[j] * exp_int[j + 1]) / (exp_int[j + 1] - exp_int[j]) ** 2.0 + ILL[:, j])
        else:  # j == J-2
            hess_alpha[j, j] = -dot(Y[:, j], (exp_int[j] * exp_int[j - 1]) / (exp_int[j] - exp_int[j - 1]) ** 2.0 + ILL[:, j]) - \
                               dot(Y[:, j + 1], ILL[:, j])

    for j in range(J - 2):
        # non-adjacent intercepts have zero mixed partial derivatives
        hess_alpha[j, j + 1] = sum(Y[:, j + 1]) * exp_int[j] * exp_int[j + 1] / (exp_int[j + 1] - exp_int[j]) ** 2.0
        hess_alpha[j + 1, j] = hess_alpha[j, j + 1]  # symmetric

    if X is not None:  # \partial \alpha_j, \partial \beta
        for j in range(J - 1):
            hess_alpha[j, (J - 1):] = -dot(X.transpose(), ILL[:, j] * (Y[:, j] + Y[:, j + 1]))
            hess_alpha[(J - 1):, j] = hess_alpha[j, (J - 1):]  # symmetric

    return hess_alpha


def _l2_clogistic_hessian_slope(X, Y, ILL, penalty_matrix, value=True):
    """ Calculate the matrix of partial second derivatives of the slopes

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        ILL : array_like. inverse logit of something. see helper function
        penalty_matrix : array_like. penalty matrix
        value : boolean. ILL: if True, return the actual Hessian. this is SUPER slow! \
        if False, return the diagonal weight matrix needed to calculate the Hessian downstream
    Returns:
        either the true Hessian if value=True, or a sparse diagonal matrix for downstream use
    """
    n = Y.shape[0]
    J = ILL.shape[1] + 1

    D = Y[:, 0] * ILL[:, 0]  # initialization
    for j in range(1, J):
        if j < J - 1:
            D += Y[:, j] * (ILL[:, j] + ILL[:, j - 1])
        else:
            D += Y[:, j] * ILL[:, j - 1]

    # make an n x n diagonal matrix whose diagonal is D, off-diagonal is 0
    DD = sparse.csr_matrix((D, (range(n), range(n))), shape=(n, n))
    if value:
        return X.transpose().dot(DD.dot(X)) + penalty_matrix  # SLOW
    else:
        return DD  # faster to push matrix mult downstream


def l2_clogistic_ranef(X, Y, alpha, beta, penalty_matrix, offset, LU, **kwargs):
    """ Use a fixed inverse hessian to determine step direction \
    this inverse hessian is represented by the LU decomposition of the approximate Hessian

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        alpha : array_like. intercepts.\
        must have shape == one less than the number of columns of `Y`
        beta : array_like. parameters.\
        must have shape == number of columns of X
        penalty_matrix : array_like
        offset : array_like, optional. Defaults to 0
        LU : LU decomposition of approximate Hessian matrix
        min_its : int, optional. Minimum number of iterations to take
        max_its : int, optional. Maximum number of iterations to take
        tol : float. Convergence tolerance on relative change in `beta`

    Returns:
        Updated parameter vector `beta`
    """

    min_its = kwargs.get('min_its', 2)
    max_its = kwargs.get('max_its', 200)
    tol = kwargs.get('tol', 1e-3)

    for i in range(max_its):

        old_beta = beta.copy()
        grad = l2_clogistic_gradient(X, Y, intercept=False, alpha=alpha, beta=beta,
                                     penalty_matrix=penalty_matrix, offset=offset)
        step = LU.solve(grad)

        ILL = _l2_clogistic_hessian_ILL(X, alpha, beta, offset=offset, n=Y.shape[0])
        W = _l2_clogistic_hessian_slope(X, Y, ILL, penalty_matrix, value=False)

        Xu = dot(X, step)
        upstairs = dot(step, grad)
        downstairs = dot(Xu, dot(W, Xu)) + dot(step, dot(penalty_matrix, step))
        step *= np.sum(upstairs / downstairs)

        beta -= step
        change = np.linalg.norm(beta - old_beta) / np.linalg.norm(beta)
        if change < tol and i >= min_its:
            break

    return beta


def l2_clogistic_fixef(X, Y, **kwargs):
    """ Take a true, second-order Newton step for fixed effects

    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        alpha : array_like. intercepts.\
        must have shape == one less than the number of columns of `Y`
        beta : array_like. parameters. \
        must have shape == number of columns of X
        min_its : int, optional. Minimum number of iterations
        max_its : int, optional. Maximum number of iterations
        tol : float, optional. Convergence tolerance on relative change in `alpha` and `beta`
        penalty_matrix : array_like, optional.

    Returns:
        Updated intercepts, updated main effects, both array-like
    """
    if X is not None:
        # X can be none if there are no fixed effects - i.e. just fit the intercept
        p = X.shape[1]
        beta = kwargs.get('beta', np.zeros(X.shape[1]))
        offset = kwargs.get('offset', np.zeros(X.shape[0]))
    else:
        p = 0
        beta = None
        offset = None

    J = Y.shape[1]
    min_its = kwargs.get('min_its', 2)
    max_its = kwargs.get('max_its', 200)
    tol = kwargs.get('tol', 1e-3)
    alpha = kwargs.get('alpha', np.linspace(-1, 1, J - 1))
    penalty_matrix = kwargs.get('penalty_matrix', sparse.csr_matrix((p, p)))

    for i in range(max_its):
        old_alpha = 1.0 * alpha
        old_beta = 1.0 * beta if beta is not None else None

        grad = l2_clogistic_gradient(X=X,
                                     Y=Y,
                                     alpha=alpha,
                                     beta=beta,
                                     penalty_matrix=penalty_matrix,
                                     offset=offset)
        H = l2_clogistic_hessian(X=X,
                                 Y=Y,
                                 alpha=alpha,
                                 beta=beta,
                                 penalty_matrix=penalty_matrix,
                                 offset=offset)

        step = solve(H, grad)
        upstairs = dot(step, grad)
        downstairs = dot(step, dot(H, step))  # cf ranef: penalty matrix is 0 for fixed effects
        # sometimes step size is a 1x1 matrix. sum converts to scalar
        step_size = np.sum(upstairs / downstairs)
        step *= step_size

        alpha -= step[0:J - 1]

        if beta is None:
            change = np.linalg.norm(alpha - old_alpha) / np.linalg.norm(alpha)
        else:
            beta -= step[J - 1:]
            change = np.linalg.norm(np.concatenate([alpha, beta]) -
                                    np.concatenate([old_alpha, old_beta])) / \
                     np.linalg.norm(np.concatenate([alpha, beta]))
        if change < tol and i >= min_its:
            break

    return alpha, beta
