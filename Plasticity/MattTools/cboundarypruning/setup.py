#!/usr/bin/env python
"""
setup.py file for SWIG kimservice
"""

from numpy.distutils.core import setup, Extension
import os

boundary_pruning_module = Extension('_boundarypruning',
    sources=['boundary_pruning_wrapper.i','boundary_pruning.cpp'],
    swig_opts=['-c++'],
    include_dirs=['.'],
    libraries=['rt', 'm']
    )

setup (name = 'boundarypruning',
    version = '0.0.1a',
    author      = "Woosong Choi, Matt Bierbaum",
    description = """CPP boundary pruning method""",
    ext_modules = [boundary_pruning_module],
    py_modules = ['boundarypruningutil', 'boundarypruningvtk'] 
    )
