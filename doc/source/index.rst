.. diamond documentation master file, created by
   sphinx-quickstart on Wed Mar 15 16:50:05 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Diamond
===================================

Diamond utilizes iterative, quasi-Newton 2nd-order solvers to estimate certain generalized linear models (glms) with known covariance structure. A common use is fitting mixed effects models, with their covariance already being known by another means (e.g. after fitting in R using lme4). These 2nd-order iterative solvers are considerably faster than a full-blown solution, assuming that the covariances are known. Currently, Diamond does not solve for the covariance structure. This must be input a-priori. In addition, only logistic and ordinal logistic response variables are currently implemented. See the `Readme <http://github.com/stitchfix/diamond>`_ for more details, `this blog post <http://multithreaded.stitchfix.com/blog/2017/08/07/diamond1/>`_ for the math behind Diamond, and `this blog post <http://multithreaded.stitchfix.com/blog/2017/08/07/diamond2/>`_ for more on mixed-effects models and the specifics of Diamond.




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
