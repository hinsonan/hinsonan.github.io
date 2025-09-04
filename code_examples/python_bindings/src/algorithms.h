#pragma once
#include <vector>
#include <complex>
#include <string>

// Prime number sieve
std::vector<int> find_primes(int limit);

// Matrix multiplication
std::vector<double> matrix_multiply(const std::vector<double>& a,
                                   const std::vector<double>& b,
                                   int n);

// Fibonacci with memoization
long long fibonacci(int n);
