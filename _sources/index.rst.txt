.. diamond documentation master file, created by
   sphinx-quickstart on Wed Mar 15 16:50:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Diamond
===================================

Diamond utilizes iterative, quasi-Newton 2nd-order solvers to estimate certain generalized linear models (glms) with known covariance structure. A common use is fitting mixed effects models, with their covariance already being known by another means (e.g. after fitting in R using lme4). These 2nd-order iterative solvers are considerably faster than a full-blown solution, assuming that the covariances are known. Currently, Diamond does not solve for the covariance structure. This must be input a-priori. In addition, only logistic and ordinal logistic response variables are currently implemented. Mathematically, Diamond solves the following problem:

If :math:`l` is the loglikelihood of the data given parameters
:math:`\vec\beta`, diamond minimizes
:math:`-l + \frac{1}{2} \vec\beta^T \Pi \vec\beta`
for a given matrix :math:`\Pi`

For logistic regression, :math:`l = \sum\limits_{i=1}^n y_i log(p_i) + (1-y_i) log(1-p_i)`
where :math:`p_i = \sigma^{-1}(\vec\beta^T x_i)`

For ordinal logistic regression,
:math:`l = l(\textbf{y}|\vec \alpha, \vec \beta ) = \sum\limits_{i=1}^n \sum\limits_{j=1}^J y_{ij} \left [ \sigma^{-1}(\alpha_j + \vec \beta \cdot \vec x_i) - \sigma^{-1}(\alpha_{j-1} + \vec \beta \cdot \vec x_i) \right ]`

where
   - :math:`\sigma^{-1}(x) = \frac{1}{1 + e^{-x}}` is the inverse logit function
   - :math:`\vec\alpha` are the intercepts
   - :math:`\vec x_i` the features of observation :math:`i`
   - :math:`y_{ij}=1` if and only if observation :math:`i` realizes response level :math:`j`
   - :math:`y_{ij}=0` in all other cases

Optimization Details
====================

The key idea is that we can _almost_ separate the optimization problem across levels fo the random effects. But of course the main effects are common to the model for each level, so we can't really separate the problem. This does, however, suggest a strategy that alternates between two stages:

  1. solve for main effects
  2. solve for random effect coefficients with the main effects from (1) treated as constants

Because the whole problem is convex this alternating iteration will converge.

The advantage of this approach is that the sub-problem in (2) has a block diagonal Hessian matrix and so we can afford to invert it. This will let us use a newton-like method that will converge faster than a 1st order approach.

Diamond was designed to tackle large problems with many random effect levels. One of the biggest bottlenecks with a standard Newton iterative approach is the need to invert the Hessian at every step. Even with ours being block-diagonal, this can be a considerably slow operation. To speed up convergence, we employ a fixed-Hessian approach, where we compute the Hessian once at the beginning, then treat it as fixed throughout all iterations.

In principle, one could also separate the problem in (2) across levels and solve them in parallel, potentially making it faster still / sidestepping the need for tons of memory.

The reason the fixed Hessian might be useful is related to the convexity of the problem. One helpful way of thinking about it is to think of gradient descent, Newton's method and the fixed Hessian as all being example of majorization-minimization algorithms.

The basic idea is that instead of minimizing our function f (which we don't know how to do analytically) we form a local approximation to f with a simple form and minimize that. Then we iterate between forming new local approximations and minimizing them.

The choice of the local approximation gives different methods:

  - gradient descent: a "diagonal" quadratic approximation **INSERT EQUATION HERE**
  - newton-raphson: a 2nd order taylor approximation (with hessian) **INSERT EQUATION HERE**
  - fixed-Hessian: a quadratic approximation like the taylor series, but with a fixed Hessian, chosen to bound the 2nd order taylor series. The fixed Hessian matrix must be "larger" than the true Hessian (the difference must be positive definite). 

  ![fixed Hessian formula](http://www.sciweavers.org/tex2img.php?eq=%20%5Cwidetilde%7BH%7D%3D-%20%5Cfrac%7B1%7D%7B4%7DXX%5ET%20-%20%5Clambda%20%5Cem%7BI%7D%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \widetilde{H}=- \frac{1}{4}XX^T - \lambda \em{I}  " width="137" height="43" /> "Fixed Hessian Approximation")

Intuitively, the best quadratic approximation is the taylor series. But using a non-diagonal bound on the Hessian may be a much better approximation than the diagonal bound that gives us gradient descent.

Step Size
=========

When using the fixed-Hessian method, we can often select better step sizes by using a conjugate gradient approach.

To select the modified step size:

  - Take the step direction (u) from the fixed hessian approach
  - Calculate the restriction of f(x) in the direction of u, and (using the chain rule) find the exact gradient and hessian for the restriction
  - Use the gradient/hessian to find a newton step size (a newton step for the restricted function along u)

Happily, we don't need to invert the true hessian to compute the step size, and so we can do this efficiently at each iteration

The idea is to use the step direction from the fixed-Hessian method, but then to take advantage of this property to do a better job selecting the step size than we would by using the fixed hessian alone. Minka claims this is significantly better in practice.


More Information
================

For a more detailed discussion, see [http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf](http://research.microsoft.com/en-us/um/people/minka/papers/logreg/minka-logreg.pdf) section 5: Fixed-Hessian Newton method

Conjugate Gradient Method: [https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

Contents:

.. toctree::
   :maxdepth: 2

   diamond.glms
   diamond.solvers



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
