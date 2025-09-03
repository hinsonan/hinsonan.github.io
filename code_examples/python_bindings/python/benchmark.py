#!/usr/bin/env python3
"""
Performance benchmark comparing pure Python implementations with C++ bindings.
Demonstrates the performance benefits of using pybind11 for computationally intensive tasks.
"""

import time
import random
import sys
import os
from typing import Tuple, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import C++ module
try:
    import algorithms_cpp
except ImportError as e:
    print(f"ERROR: C++ module 'algorithms_cpp' not found!")
    print(f"Please build the module first: python setup.py build_ext --inplace")
    sys.exit(1)

# Import Python implementations
from algorithms import find_primes, matrix_multiply, fibonacci


def time_function(func, *args, **kwargs) -> Tuple[float, Any]:
    """
    Time a function execution.
    
    Returns:
        Tuple of (execution_time_in_seconds, function_result)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def format_comparison(py_time: float, cpp_time: float) -> str:
    """
    Format the performance comparison between Python and C++.
    
    Args:
        py_time: Python execution time in seconds
        cpp_time: C++ execution time in seconds
    
    Returns:
        Formatted speedup string
    """
    if cpp_time > 0:
        speedup = py_time / cpp_time
        return f"{speedup:.1f}x"
    return "∞"


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{title}")
    print("-" * len(title))


def run_benchmark(name: str, py_func, cpp_func, *args, **kwargs) -> Tuple[float, float]:
    """
    Run a single benchmark comparing Python and C++ implementations.
    
    Args:
        name: Benchmark name for display
        py_func: Python function to benchmark
        cpp_func: C++ function to benchmark
        *args, **kwargs: Arguments to pass to both functions
    
    Returns:
        Tuple of (python_time, cpp_time) in seconds
    """
    print_header(name)
    
    # Run Python version
    py_time, py_result = time_function(py_func, *args, **kwargs)
    print(f"Python:  {py_time:8.4f} seconds", end="")
    
    # Run C++ version
    cpp_time, cpp_result = time_function(cpp_func, *args, **kwargs)
    
    # Display results
    speedup = format_comparison(py_time, cpp_time)
    print(f"  │  C++: {cpp_time:8.4f} seconds  │  Speedup: {speedup:>6}")
    
    # Verify results match (if they're comparable)
    if hasattr(py_result, '__len__') and hasattr(cpp_result, '__len__'):
        if len(py_result) != len(cpp_result):
            print(f"⚠️  Warning: Result mismatch! Python: {len(py_result)}, C++: {len(cpp_result)}")
    
    return py_time, cpp_time


def main():
    """Run all benchmarks and display summary."""
    
    print("=" * 70)
    print(" " * 15 + "PYTHON vs C++ PERFORMANCE BENCHMARK")
    print(" " * 20 + "Using pybind11 bindings")
    print("=" * 70)
    
    results: List[Tuple[str, float, float]] = []
    
    # Benchmark 1: Prime Numbers
    py_time, cpp_time = run_benchmark(
        "Prime Numbers (up to 100,000)",
        find_primes, algorithms_cpp.find_primes,
        100000
    )
    results.append(("Prime Sieve", py_time, cpp_time))
    
    # Benchmark 2: Matrix Multiplication
    n = 100
    matrix_a = [random.random() for _ in range(n * n)]
    matrix_b = [random.random() for _ in range(n * n)]
    
    py_time, cpp_time = run_benchmark(
        f"Matrix Multiplication ({n}x{n})",
        matrix_multiply, algorithms_cpp.matrix_multiply,
        matrix_a, matrix_b, n
    )
    results.append(("Matrix Multiply", py_time, cpp_time))
    
    # Benchmark 3: Fibonacci
    fib_n = 35
    py_time, cpp_time = run_benchmark(
        f"Fibonacci (n={fib_n})",
        fibonacci, algorithms_cpp.fibonacci,
        fib_n
    )
    results.append(("Fibonacci", py_time, cpp_time))
    
    # Display summary
    print("\n" + "=" * 70)
    print(" " * 30 + "SUMMARY")
    print("=" * 70)
    
    # Table header
    print(f"\n{'Algorithm':<20} {'Python (s)':>12} {'C++ (s)':>12} {'Speedup':>10}")
    print("-" * 56)
    
    # Individual results
    total_py = 0.0
    total_cpp = 0.0
    
    for name, py_t, cpp_t in results:
        speedup = format_comparison(py_t, cpp_t)
        print(f"{name:<20} {py_t:12.4f} {cpp_t:12.4f} {speedup:>10}")
        total_py += py_t
        total_cpp += cpp_t
    
    # Total row
    print("-" * 56)
    total_speedup = format_comparison(total_py, total_cpp)
    print(f"{'TOTAL':<20} {total_py:12.4f} {total_cpp:12.4f} {total_speedup:>10}")
    
    # Final message
    print("\n" + "=" * 70)
    avg_speedup = total_py / total_cpp if total_cpp > 0 else float('inf')
    print(f"🚀 C++ with pybind11 is on average {avg_speedup:.1f}x faster than pure Python!")
    print("=" * 70)


if __name__ == "__main__":
    main()