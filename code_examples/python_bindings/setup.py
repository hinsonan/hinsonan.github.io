from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "algorithms_cpp",
        ["src/bindings.cpp", "src/algorithms.cpp"],
        include_dirs=["src"],
        cxx_std=11,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="algorithms_comparison",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.6",
)