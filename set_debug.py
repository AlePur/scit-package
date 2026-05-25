# temp_setup.py
from setuptools import setup, Extension
import os

# Define the *exact* path you confirmed contains the 'numpy' sub-directory
numpy_include_path = "/home/aleksander/Skrivbord/pyproject/venv/lib/python3.12/site-packages/numpy/_core/include"

# Create a dummy C file for testing if you don't have your full one ready
# echo '#include "numpy/arrayobject.h"' > test_module.c
# Then add more basic C code if needed to make it a valid file

test_module = Extension(
    'test_module',
    sources=['src/advanced/_median.c'], # <-- IMPORTANT: Replace with your actual C/C++ source file name
                               # If you made a dummy, use 'test_module.c'
    include_dirs=[numpy_include_path],
    # Add other arguments you might have, e.g., library_dirs, libraries etc.
)

setup(
    name='TestBuild',
    version='0.1.0',
    ext_modules=[test_module],
)