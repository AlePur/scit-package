from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    # Explicitly find and map packages relative to 'src'
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    ext_modules=cythonize("src/**/*.pyx", annotate=False),
    include_dirs=[numpy.get_include()]
)