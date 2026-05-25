from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("src/**/*.pyx", annotate=False),
    include_dirs=[numpy.get_include()]
)
