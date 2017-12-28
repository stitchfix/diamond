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
    name='sf-diamond',
    version='0.2.2',
    author=['Aaron Bradley', 'Timothy Sweetser'],
    url='http://github.com/stitchfix/diamond',
    author_email=['abradley@stitchfix.com', 'tsweetser@stitchfix.com'],
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'future',
                      'pandas',
                      'cython'],
    license='LICENSE.txt',
    description='GLMMs with known variances in python with Newton-like solver',
    long_description=open('README.rst').read(),
    ext_modules=cythonize(extensions)
)
