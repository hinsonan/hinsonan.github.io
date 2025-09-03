# pybind11 example

This is a small program demonstrating how to build and call a C++ program from python.

## Quick Start

`docker compose build`

`docker compose up`

## How the Build Process Happens

The setup calls the compiler and builds a algorithms_cpp module

```python
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "algorithms_cpp",                          # Module name in Python
        ["src/bindings.cpp", "src/algorithms.cpp"], # C++ source files
        include_dirs=["src"],                       # Header locations
        cxx_std=11,                                 # C++ standard
        extra_compile_args=["-O3", "-march=native"], # Optimization flags
    ),
]

setup(
    name="algorithms_comparison",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
```

This builds the shared objects and links them to a shared library. Now `algorithms_cpp` can be imported into python
