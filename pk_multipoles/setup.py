import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "useful_functions",
        ["useful_functions.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
