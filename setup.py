from distutils.core import setup, Extension
from Cython.Build import cythonize
import sysconfig
import numpy

# Get the actual Python include path
#python_include = sysconfig.get_path('include')
#numpy_include = "/tmp/np-include"

setup(
    ext_modules=cythonize("src/**/*.pyx", annotate=False),
    include_dirs=[numpy.get_include()]
    #include_dirs=[numpy_include],
    #include_dirs=[
    #       "/home/aleksander/Skrivbord/pyproject/venv/bin/python",
    #       "/home/aleksander/Skrivbord/pyproject/venv/lib/python3.12/site-packages/numpy/_core/include"
    #],
    #extra_compile_args=["-O3", "-march=native", "-fopenmp"],
    #extra_link_args=["-fopenmp"],
)
