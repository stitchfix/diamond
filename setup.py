from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension(
    name='diamond.solvers.repeated_block_dot',
    sources=['diamond/solvers/repeated_block_dot.pyx'],
    include_dirs=[numpy.get_include()]
    )
]

setup(
    name='diamond',
    version='0.1',
    author=['Aaron Bradley', 'Timothy Sweetser'],
    author_email=['abradley@stitchfix.com', 'tsweetser@stitchfix.com'],
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'cython'],
    license='MIT License',
    description='GLMMs with known variances in python with Newton-like solver',
    ext_modules=cythonize(extensions)
)
