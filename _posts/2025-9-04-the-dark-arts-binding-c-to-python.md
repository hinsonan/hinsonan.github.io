---
layout: post
title: "Mastering the Dark Arts: Binding C to Python"
date: 2025-09-04
categories: ML
---

So you want to dabble in the dark arts and gain the sweet performance boost of `C/C++` while still using `python`. I've got some news for you, buddy: this road is paved with dead corpses and you should turn back now. The only problem is your company dropped you off here and that one crucial program written in python is too slow for your needs. No matter how much you pray to the JIT compiler and spam `numba` decorators your program is slower than your dead tortoise.

No one on your team even knows what garbage collection is. They have heard rumors of the GIL, but they still think it's something fish use to breathe. The senior ML dev you approached for help? They're suddenly "sick" indefinitely after you dared suggest there's life beyond Jupyter notebooks.

Time to don the black robes and learn the dark arts. This python program is about to perform at a rate man cannot comprehend

# I'm all Bound Up

Bindings are a bridge between C and python. The python interpreter can call compiled C/C++ code. This is important since C++ programs can operate at much higher speeds than interpreting an entire python program. Most of all the heavy duty computations in python are just calling C executables. Numpy and Pytorch all have a large portion of their operations in C/C++.

The reason that python can call C/C++ is because the interpreter is written in C and python has an API for the ctypes in python. When you are making a binding you are doing the following:

* Creating a shared library that has your C/C++ program
* Wrapping the code and translating between python types and C types
* Exposing certain functions you want to call in python

## Options for casting the Dark Arts

### [Manual python/C API](https://docs.python.org/3/c-api/intro.html#)

This is a pretty rough option. It requires you to know a lot about the python and C ecosystem. I have used this in the past for primitive functions and operations with simple data types. I don't think this is the best choice if you need more complex programs done in a shorter development timeline.

Here is an example of how to sum a list using this API

```c
long
sum_list(PyObject *list)
{
    Py_ssize_t i, n;
    long total = 0, value;
    PyObject *item;

    n = PyList_Size(list);
    if (n < 0)
        return -1; /* Not a list */
    for (i = 0; i < n; i++) {
        item = PyList_GetItem(list, i); /* Can't fail */
        if (!PyLong_Check(item)) continue; /* Skip non-integers */
        value = PyLong_AsLong(item);
        if (value == -1 && PyErr_Occurred())
            /* Integer too big to fit in a C long, bail out */
            return -1;
        total += value;
    }
    return total;
}
```

Even for a simple sum function, this has many checks to ensure things are the right size. In more complicated examples, you end up managing a lot of references and memory allocation that don't feel as natural as normal C/C++. You also have to manage and convert all your python types to C types. Let's not even mention the GIL. There is a lot of room for error here.

### `ctypes`

[ctypes](https://docs.python.org/3/library/ctypes.html) provides C compatible data types and lets you call `.so` files or `.dll` files. Let's run through an example.

```c
#include <stdio.h>

void myprint(void);

void myprint()
{
    printf("hello world\n");
}
```

Compile the code

`gcc -shared -Wl,-soname,testlib -o testlib.so -fPIC testlib.c`

Write the python code

```python
import ctypes

testlib = ctypes.CDLL('/full/path/to/testlib.so')
testlib.myprint()
```

This seems like a good choice, except if you ever use a C++ feature, then `ctypes` is not compatible. This is meant for C types obviously. The other issue is loading the dll. Depending on the OS, you may need to load different DLLs based on what the user is using for their OS. This line won't work on every machine `testlib = ctypes.CDLL('/full/path/to/testlib.so')`. The question of memory is still up for grabs since it could be the compiled C program or python that needs to manage it.

### `CFFI` (C Foreign Function Interface for Python)

[CFFI](https://cffi.readthedocs.io/en/stable/) is another interface that allows you to call C, and once again, it is not compatible with C++. You are going to have to write some nasty wrappers. The build process also grows quickly.

Here is a basic example

```c
from cffi import FFI
ffi = FFI()
ffi.cdef("""
    typedef struct {
        unsigned char r, g, b;
    } pixel_t;
""")
image = ffi.new("pixel_t[]", 800*600)

f = open('data', 'rb')     # binary mode -- important
f.readinto(ffi.buffer(image))
f.close()

image[100].r = 255
image[100].g = 192
image[100].b = 128

f = open('data', 'wb')
f.write(ffi.buffer(image))
f.close()
```

Let's be honest when you first see this you have no clue what is going on. The `cdef` is a string... This is the simplest example and it only gets harder from here. I have also seen some people complain about padding when you compile so the ways that the data gets shifted could cause issues for you. Due to the lack of C++ support, build complexity, and memory management issues, this is not a great choice. It is probably a bit easier and portable than `ctypes` since you don't have to define all the ctypes manually but that still does not gain you much. CFFI can also check your binding types at compile time and save you some runtime crashes.

### SWIG

This method is pretty interesting but the way it's designed is mainly for C++ programmers who want to add some scripting to their project. It is not geared toward python programmers who want to call C libs. [SWIG](https://www.swig.org/exec.html) has some basic tutorials that are almost too simple and many people have complaints about using it for our purpose of bridging C and python. You can find this example on their [tutorial](https://www.swig.org/tutorial.html)

Build an interface file

```c
/* example.i */
%module example
%{
/* Put header files here or function declarations like below */
extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
%}

extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
```

Build a tcl module

```
unix % swig -tcl example.i
unix % gcc -fPIC -c example.c example_wrap.c \
       -I/usr/include/tcl
unix % gcc -shared example.o example_wrap.o -o example.so
unix % tclsh
% load ./example.so example
% puts $My_variable
3.0
% fact 5
120
% my_mod 7 3
1
% get_time
Sun Feb 11 23:01:07 2018
```

this generates the following code...Good luck if you ever needed to debug this

```c

/*
 * File : example_wrap.c
 * Thu Apr  4 13:11:45 1996
 *
 * This file was automatically generated by :
 * Simplified Wrapper and Interface Generator (SWIG)
 * 
 * Copyright (c) 1995,1996
 * The Regents of the University of California and
 * The University of Utah
 *
 */

/* Implementation : TCL */

#define INCLUDE_TCL    
#define INCLUDE_TK     
#include INCLUDE_TCL
#include 
#include 
static void _swig_make_hex(char *_c, void *_ptr, char *type) {
static char _hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
			'a', 'b', 'c', 'd', 'e', 'f' };
  unsigned long _p,_s;
  char     _result[128], *_r;
  _r = _result;
  _p = (unsigned long) _ptr;
  if (_p > 0) {
     while (_p > 0) {
        _s = _p & 0xf;
        *(_r++) = _hex[_s];
        _p = _p >> 4;
     } 
     *_r = '_';
     while (_r >= _result) *(_c++) = *(_r--);
  } else {
     strcpy(_c,"NULL");
  }
  if (_ptr) strcpy(_c,type);
}
static char *_swig_get_hex(char *_c, void **ptr, char *_t) {
  unsigned long _p;
  char *_tt;
  _p = 0;
  if (*_c == '_') { 
     _c++;
     while (*_c) {
       if ((*_c >= '0') && (*_c <= '9')) _p = (_p << 4) + (*_c - '0');
       else if ((*_c >= 'a') && (*_c <= 'f')) _p = (_p << 4) + ((*_c - 'a') + 10);
       else break; 
       _c++;
     }
     if (_p == 0) {
         return (char *) _c;
     }
     _tt = _c;
     if (_t) {
        if (strcmp(_c,_t)) return _tt;
     }
     *ptr = (void *) _p;
      return (char *) 0;
   } else {
      if (strcmp(_c,"NULL") == 0) {
         *ptr = (void *) 0;
         return (char *) 0;
      }
      else
         return _c;
  }
}

#define SWIG_init    Wrap_Init




/* A TCL_AppInit() function that lets you build a new copy
 * of tclsh.
 *
 * The macro WG_init contains the name of the initialization
 * function in the wrapper file.
 */

#ifndef SWIG_RcFileName
char *SWIG_RcFileName = "~/.myapprc";
#endif

#if TCL_MAJOR_VERSION == 7 && TCL_MINOR_VERSION >= 4
int main(int argc, char **argv) {

  Tcl_Main(argc, argv, Tcl_AppInit);
  return(0);

}
#else
extern int main();
#endif

int Tcl_AppInit(Tcl_Interp *interp){
  int SWIG_init(Tcl_Interp *);  /* Forward reference */

  if (Tcl_Init(interp) == TCL_ERROR) 
    return TCL_ERROR;

  /* Now initialize our functions */

  if (SWIG_init(interp) == TCL_ERROR)
    return TCL_ERROR;

  tcl_RcFileName = SWIG_RcFileName;
  return TCL_OK;
}

extern double   My_variable;
extern int  fact(int  );
extern int  mod(int  ,int  );
extern char * get_time();
int _wrap_tcl_fact(ClientData clientData, Tcl_Interp *interp, int argc, char *argv[]) {
	 int  _result;
	 int _arg0;

	 if (argc != 2) {
		 Tcl_SetResult(interp, "Wrong # args  int  : fact n ",TCL_STATIC);
		 return TCL_ERROR;
	}
	 _arg0 = (int ) atol(argv[1]);
	 _result = fact(_arg0);
	 sprintf(interp->result,"%ld", (long) _result);
	 return TCL_OK;
}
int _wrap_tcl_mod(ClientData clientData, Tcl_Interp *interp, int argc, char *argv[]) {
	 int  _result;
	 int _arg0;
	 int _arg1;

	 if (argc != 3) {
		 Tcl_SetResult(interp, "Wrong # args  int  : mod x y ",TCL_STATIC);
		 return TCL_ERROR;
	}
	 _arg0 = (int ) atol(argv[1]);
	 _arg1 = (int ) atol(argv[2]);
	 _result = mod(_arg0,_arg1);
	 sprintf(interp->result,"%ld", (long) _result);
	 return TCL_OK;
}
int _wrap_tcl_get_time(ClientData clientData, Tcl_Interp *interp, int argc, char *argv[]) {
	 char * _result;

	 if (argc != 1) {
		 Tcl_SetResult(interp, "Wrong # args  char * : get_time ",TCL_STATIC);
		 return TCL_ERROR;
	}
	 _result = get_time();
	 Tcl_SetResult(interp, _result, TCL_VOLATILE);
	 return TCL_OK;
}
int Wrap_Init(Tcl_Interp *interp) {
	 if (interp == 0) 
		 return TCL_ERROR;
	 Tcl_LinkVar(interp, "My_variable", (char *) &My_variable, TCL_LINK_DOUBLE);
	 Tcl_CreateCommand(interp, "fact", _wrap_tcl_fact, (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
	 Tcl_CreateCommand(interp, "mod", _wrap_tcl_mod, (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
	 Tcl_CreateCommand(interp, "get_time", _wrap_tcl_get_time, (ClientData) NULL, (Tcl_CmdDeleteProc *) NULL);
	 return TCL_OK;
}
```

Now we can finally build the python module

```
unix % swig -python example.i
unix % gcc -c example.c example_wrap.c \
       -I/usr/local/include/python2.7
unix % ld -shared example.o example_wrap.o -o _example.so 
We can now use the python module as follows :
>>> import example
>>> example.fact(5)
120
>>> example.my_mod(7,3)
1
>>> example.get_time()
'Sun Feb 11 23:01:07 2018'
```

Use the module

```python
>>> import example
>>> example.fact(5)
120
>>> example.my_mod(7,3)
1
>>> example.get_time()
'Sun Feb 11 23:01:07 2018'
```

This simple example doesn't demonstrate the issues well. Granted this method was not designed for us so its ok if it's not what we wanted.

An annoying issue with larger libraries is specifying the interface file

```c
// pair.h.  A pair like the STL
namespace std {
   template<class T1, class T2> struct pair {
       T1 first;
       T2 second;
       pair() : first(T1()), second(T2()) { };
       pair(const T1 &f, const T2 &s) : first(f), second(s) { }
   };
}

// pair.i - SWIG interface
%module pair
%{
#include "pair.h"
%}

// Ignore the default constructor
%ignore std::pair::pair();      

// Parse the original header file
%include "pair.h"

// Instantiate some templates

%template(pairii) std::pair<int,int>;
%template(pairdi) std::pair<double,int>;
```

Those last two lines show that you have to define these templates for all the possible scenarios and it can be a pain to upkeep. If you forget one, then you're out of luck. Let's move on to some better options for most people.

# Conquering the Dark Arts

## Boost

All bow before the Boost library. Any C++ developer knows the beauty and pain that this brings. Boost is a large, comprehensive library with extensive functionality. With all this power you get significant overhead, but when you need robust, mature functionality, this will do it. I have used Boost in the field for this and it does work great and let's you bind complex C++ to python.

Here is an [example](https://wiki.python.org/moin/boost.python/ExportingClasses)

```c
#include <iostream>
#include <string>

namespace { // Avoid cluttering the global namespace.

  // A friendly class.
  class hello
  {
    public:
      hello(const std::string& country) { this->country = country; }
      std::string greet() const { return "Hello from " + country; }
    private:
      std::string country;
  };

  // A function taking a hello object as an argument.
  std::string invite(const hello& w) {
    return w.greet() + "! Please come soon!";
  }
}
```

```c
#include <boost/python.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(getting_started2)
{
    // Create the python type object for our extension class and define __init__ function.
    class_<hello>("hello", init<std::string>())
        .def("greet", &hello::greet)  // Add a regular member function.
        .def("invite", invite)  // Add invite() as a regular function to the module.
    ;

    def("invite", invite); // Even better, invite() can also be made a member of module!!!
}
```

Now the python

```python
>>> from getting_started2 import *
>>> hi = hello('California')
>>> hi.greet()
'Hello from California'
>>> invite(hi)
'Hello from California! Please come soon!'
>>> hi.invite()
'Hello from California! Please come soon!'
```

Well this seems a lot better and it is. The issue with Boost is incorporating it into the build process like `cmake` or your companies devops pipelines...Your CMake will need a lot of Boost dependencies and your Docker images will have to suffer with the bloat. This is much better and can be a great option if you need other boost libraries in your program. If you do not need boost then there is a lighter weight more commonly used option called `pybind11`.

## PyBind11

A header only library! For those not used to C a header only library is generally easier to incorporate into your builds or system. You only need the header files.

based on their [example](https://pybind11.readthedocs.io/en/stable/basics.html#keyword-arguments) lets see some code

```c
int add(int i, int j) {
    return i + j;
}
```

```c
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m, py::mod_gil_not_used()) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
```

The beauty of this header library is we don't have to link any special libs.

```bash
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3 -m pybind11 --extension-suffix)
```

Use it in python

```python
import example
example.add(1, 2)
3
```

This is pretty minimal and powerful. This macro is binding the python to the C call `PYBIND11_MODULE(example, m, py::mod_gil_not_used())`. Another neat thing is in python 3.13+ you can take advantage of free-threading. With all this power you can avoid the GIL. This is experimental and means you have to put on your big boy pants and make sure that your C++ code is thread safe.

## Super Saiyan Speed 

Let's write our own binding. We will write a C++ program that will perform a few different algorithms and then we will bind that to python. We will do a speed test and write these algorithms again in pure python to compare.

**C++**

```c
#include "algorithms.h"
#include <cmath>
#include <random>
#include <unordered_map>
#include <algorithm>

std::vector<int> find_primes(int limit) {
    std::vector<bool> is_prime(limit + 1, true);
    std::vector<int> primes;
    
    is_prime[0] = is_prime[1] = false;
    
    for (int i = 2; i * i <= limit; ++i) {
        if (is_prime[i]) {
            for (int j = i * i; j <= limit; j += i) {
                is_prime[j] = false;
            }
        }
    }
    
    for (int i = 2; i <= limit; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }
    
    return primes;
}

std::vector<double> matrix_multiply(const std::vector<double>& a,
                                   const std::vector<double>& b,
                                   int n) {
    std::vector<double> c(n * n, 0.0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    
    return c;
}

long long fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
```

**PYTHON**

```python
def find_primes(limit):
    """Sieve of Eratosthenes in pure python"""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]

def matrix_multiply(a, b, n):
    """Matrix multiplication in pure python"""
    c = [0.0] * (n * n)
    
    for i in range(n):
        for j in range(n):
            sum_val = 0.0
            for k in range(n):
                sum_val += a[i * n + k] * b[k * n + j]
            c[i * n + j] = sum_val
    
    return c

def fibonacci(n):
    """Fibonacci with memoization in pure python"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

**BINDINGS**

```c
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
```

### Timing Results

benchmark code

```python
#!/usr/bin/env python3
"""
Performance benchmark comparing pure python implementations with C++ bindings.
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
    sys.exit(1)

# Import python implementations
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
    Format the performance comparison between python and C++.
    
    Args:
        py_time: python execution time in seconds
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
    Run a single benchmark comparing python and C++ implementations.
    
    Args:
        name: Benchmark name for display
        py_func: python function to benchmark
        cpp_func: C++ function to benchmark
        *args, **kwargs: Arguments to pass to both functions
    
    Returns:
        Tuple of (python_time, cpp_time) in seconds
    """
    print_header(name)
    
    # Run python version
    py_time, py_result = time_function(py_func, *args, **kwargs)
    print(f"python:  {py_time:8.4f} seconds", end="")
    
    # Run C++ version
    cpp_time, cpp_result = time_function(cpp_func, *args, **kwargs)
    
    # Display results
    speedup = format_comparison(py_time, cpp_time)
    print(f"  │  C++: {cpp_time:8.4f} seconds  │  Speedup: {speedup:>6}")
    
    # Verify results match (if they're comparable)
    if hasattr(py_result, '__len__') and hasattr(cpp_result, '__len__'):
        if len(py_result) != len(cpp_result):
            print(f"⚠️  Warning: Result mismatch! python: {len(py_result)}, C++: {len(cpp_result)}")
    
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
    print(f"\n{'Algorithm':<20} {'python (s)':>12} {'C++ (s)':>12} {'Speedup':>10}")
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
    print(f"C++ with pybind11 is on average {avg_speedup:.1f}x faster than pure python!")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

Let's look at the results

```
======================================================================
               PYTHON vs C++ PERFORMANCE BENCHMARK
                    Using pybind11 bindings
======================================================================

Prime Numbers (up to 100,000)
-----------------------------
python:    0.0047 seconds  │  C++:   0.0005 seconds  │  Speedup:   9.9x

Matrix Multiplication (100x100)
-------------------------------
python:    0.0667 seconds  │  C++:   0.0007 seconds  │  Speedup:  92.8x

Fibonacci (n=35)
----------------
python:    0.8162 seconds  │  C++:   0.0081 seconds  │  Speedup: 101.0x

======================================================================
                              SUMMARY
======================================================================

Algorithm              python (s)      C++ (s)    Speedup
--------------------------------------------------------
Prime Sieve                0.0047       0.0005       9.9x
Matrix Multiply            0.0667       0.0007      92.8x
Fibonacci                  0.8162       0.0081     101.0x
--------------------------------------------------------
TOTAL                      0.8876       0.0093      95.7x

======================================================================
C++ with pybind11 is on average 95.7x faster than pure python!
======================================================================
```

**IMPORTANT NOTE: These Speedups are in the context of these algorithms**

These results are quite impressive. Sometimes I forget just how slow python is. Granted when possible I try to write python in a way that is uses a lot of C bindings like numpy and other high performant libraries. You should try hard to not write sucky slow python code. You should try to be an adult and solve problems all the way through. Sometimes its just impossible to write the python code to be as performant as you want and now with `Boost` and `PyBind11` you have the tools to solve this problem.

# Dark Arts Fire Back

Congrats you made it down this dark path of segfaults and dangling pointers. You have now made your stand as the go to guy when everything sucks and performance is awful. Casting these dark arts comes with a price. You now have to maintain a completely different ecosystem. You are not just in python world anymore. You finally made your escape into an actual programming language but unfortunately you have to come running back to python since no one knows how to use anything else. It is now up to you to manage this package and create build scripts and setup scripts that allow users to install this package. Remember now that you casted this dark magic you force users to compile and build the C code.

In the above example I did something kinda crappy. 

```python
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import C++ module
try:
    import algorithms_cpp
except ImportError as e:
    print(f"ERROR: C++ module 'algorithms_cpp' not found!")
    sys.exit(1)
```

This is not a good way to find the shared object file. It is brittle and can easily break. In order to have the shared object file compile and in the correct location for the python import statements you need to make sure you create a proper pip package that will build the shared library and put it in the correct place during the package install.

Assuming this is used in a production setting or an important setting you now have to add test and a whole other executable and build script for testing. If you choose to use something like gtest then you will need to build out a CMake and executable. This means you now have to maintain both Python tests and C++ tests.

Some Docker images will need to increase in size since you have to add all the C libs. Hope your DevOps team is ready for this new python package that needs C libs. It's better if you just make the whole pipeline for them but sometimes that is not possible.

# Conclusion: Dark Arts Grant Power and Death

Let's recap what you should do if you need more performance in python

1) Vectorize your code with numpy or other array libraries

2) Numba or other JIT optimizations

3) When your back is against the wall use `pybind11` to write your bindings

There really is no reason to bring in the behemoth `boost` library unless it has something you can't live without.

As much as I can't stand Python in many cases, it is best if you can keep your application using Python without traveling down this binding road. Don't write poor solutions in Python. Solve the problems in a meaningful and purposeful manner. If you can vectorize with numpy do that. If you can use a JIT to compile certain operations go for it. This keeps your Python environment cleaner and reduces the amount of complexity and insanity you have to maintain. Just because you made something work does not mean you solved all the problems.

Even when you try all of these options sometimes you still need to reach for the monster that calls your name. C++ will always be there for when you are ready to grow up and become an adult. As for now in 2025 the ML community still champions Python so it's back to bottles and lego blocks. At least now you know you can turn around and pick up that dusty black wizard robe and begin casting spells.

[Code For Bindings](https://github.com/hinsonan/hinsonan.github.io/blob/master/code_examples/python_bindings)