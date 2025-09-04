def find_primes(limit):
    """Sieve of Eratosthenes in pure Python"""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]

def matrix_multiply(a, b, n):
    """Matrix multiplication in pure Python"""
    c = [0.0] * (n * n)
    
    for i in range(n):
        for j in range(n):
            sum_val = 0.0
            for k in range(n):
                sum_val += a[i * n + k] * b[k * n + j]
            c[i * n + j] = sum_val
    
    return c

def fibonacci(n):
    """Fibonacci with memoization in pure Python"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)