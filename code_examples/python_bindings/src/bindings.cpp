#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "algorithms.h"

namespace py = pybind11;

PYBIND11_MODULE(algorithms_cpp, m) {
    m.doc() = "High-performance algorithms in C++";
    
    m.def("find_primes", &find_primes,
          "Find all prime numbers up to limit using Sieve of Eratosthenes",
          py::arg("limit"));
    
    m.def("matrix_multiply", &matrix_multiply,
          "Multiply two square matrices",
          py::arg("a"), py::arg("b"), py::arg("n"));
    
    m.def("fibonacci", &fibonacci,
          "Calculate Fibonacci number with memoization",
          py::arg("n"));
    }
    