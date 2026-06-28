from distutils.core import setup
from Cython.Build import cythonize

setup(name="useful_functions", ext_modules=cythonize('useful_functions.pyx'),)
