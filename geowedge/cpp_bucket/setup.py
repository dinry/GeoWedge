# -*- coding: utf-8 -*-
"""Build script for the wedge_bucket_cpp pybind11 extension.

Usage:
    cd cpp_bucket
    python setup.py build_ext --inplace

After a successful build, `wedge_bucket_cpp.cpython-*.so` (or `.pyd` on
Windows) will appear in this folder and can be imported from Python.
"""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if os.uname().sysname == "Darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.1")

ext_modules = [
    Pybind11Extension(
        "wedge_bucket_cpp",
        ["wedge_bucket_ext.cpp"],
        cxx_std=17,
        extra_compile_args=[
            "-O3",              # full optimization
            "-march=native",    # use host CPU's full instruction set
            "-ffast-math",      # allow FP reassociation (~5% extra)
            "-DNDEBUG",         # strip asserts
        ],
    ),
]

setup(
    name="wedge_bucket_cpp",
    version="0.1.0",
    description="C++17 wedge log-bucket search — pybind11 extension for "
                "streaming AML feasibility detection.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
