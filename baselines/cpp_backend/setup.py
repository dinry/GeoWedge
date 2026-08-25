# -*- coding: utf-8 -*-
"""Build script for the C++ baseline core extension.

Usage:
    cd baselines/cpp_backend
    python setup.py build_ext --inplace
"""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if os.uname().sysname == "Darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.1")

ext_modules = [
    Pybind11Extension(
        "baseline_cpp_core",
        ["baseline_cpp_core.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native", "-DNDEBUG"],
    ),
]

setup(
    name="baseline_cpp_core",
    version="0.1.0",
    description="C++17 core routines for streaming package-query baselines.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
