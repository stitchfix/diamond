Diamond
=======

O Diamond, Diamond, thou little knowest the mischief thou hast done.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: diamond_fire.jpg?raw=true
   :alt: Damn You Diamond!

   Damn You Diamond

`(Diamond was Newton's mischievous
dog) <https://en.wikipedia.org/wiki/Diamond_(dog)>`__

What is Diamond?
----------------

Diamond utilizes iterative, quasi-Newton 2nd-order solvers for certain
kinds of generalized linear models (GLMs) with arbitrary but known
L2-regularization. A common use is fitting mixed-effects models, with
their covariance already being known by another means (e.g. lme4). These
2nd-order iterative solvers are considerably faster than a full-blown
solution.

Limitations
-----------

-  The random-effects covariances must be input a-priori. Unlike `R's
   lme4 <https://cran.r-project.org/web/packages/lme4/lme4.pdf>`__ or
   `Julia's MixedModels <https://github.com/dmbates/MixedModels.jl>`__,
   Diamond does not estimate the covariance of random effects terms.
-  Diamond only supports the following models

   -  logistic regression
   -  ordinal logistic regression using proportional odds, as defined in
      Section 7.2.1 of Categorical Data Analysis, 2nd Ed., by Alan
      Agresti

-  Currently, only formulae with crossed, independent random effects are
   supported. Using the mtcars dataset as an example, these look like
   ``mpg ~ 1 + hp + (1 + hp | cyl) + (1 | gear)``. I.e. no hierarchical
   terms

Installation
------------

You must have `docker <https://docs.docker.com/engine/installation/>`__
installed. Then, run
``docker run -ti --rm -p 8888:8888 tsweetser/diamond``

Copy-paste the URL, including the token, into your browser. Then, check
out the Jupyter notebook examples!

Troubleshooting installation
----------------------------

-  You may need to restart docker if you've been running jupyter
   notebooks locally on port 8888.

Documentation
-------------

See `documentation <http://stitchfix.github.io/diamond/>`__ for more
details on the details of Diamond and how to use it

Contributing to Diamond
-----------------------

We always welcome contributions. See
`CONTRIBUTING.md <CONTRIBUTING.md>`__

Running Tests
-------------

You will need R to run the integration tests. From the root directory,
run ``pip install nose`` then ``nosetests``.

Development Status
------------------

Diamond is an evolving project. Please file issues if you would like to
use Diamond in new ways.

License
-------

See `LICENSE.txt <LICENSE.txt>`__
