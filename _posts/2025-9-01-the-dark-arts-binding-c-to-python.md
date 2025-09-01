---
layout: post
title: "Mastering the Dark Arts: Binding C to Python"
date: 2025-08-24
categories: ML
---

So you want to dabble in the dark arts and gain the sweet performance boost of `C/C++` while still using `python`. I got some news for you buddy, this road is paved with dead corpses and you should turn back now. The only problem is your company dropped you off here and that one crucial program written in python is too slow for your needs. No matter how much you pray to the JIT compiler and spam `numba` decorators your program is slower than your dead tortoise.

No one on your team even knows what garbage collection is. They have heard rumors of the GIL but still think it's something fish use to breath. The senior ML dev you approached for help? They're suddenly "sick" indefinitely after you dared suggest there's life beyond Jupyter notebooks.

Time to dawn the black robes and learn the dark arts. This python program is about to perform at a rate man cannot comprehend

# I'm all Bound Up

Bindings are a bridge between C and Python. The Python interpreter can call compiled C/C++ code. This is important since C++ programs can operate at much higher speeds than interpreting an entire python program. Most of all the heavy duty computations in Python are just calling C executables. Numpy and Pytorch all have a large portion of their operations in C/C++.

The reason that python can call C/C++ is because the interpreter is written in C and Python has an API for the ctypes in python. When you are making a binding you are doing the following

* Creating a shared library that has your C/C++ program
* Wrapping the code and translating between Python types and C types
* Exposing certain functions you want to call in Python

## Options for casting the Dark Arts

1) (Manual Python/C API)[https://docs.python.org/3/c-api/intro.html#]: This is a pretty rough option. It requires you to know a lot about the Python and C ecosystem. I have used this in the past for primitive functions and operations with simple data types. I don't think this is the best choice if you need more complex programs done is a shorter development timeline.

Here is an example of how to sum a list using this API

```C
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
```

Even for a simple sum function this has a lot of checks to make sure things are the right size. In more complicated examples you end up having to manage a lot of references and memory allocation that do not feel as natural as normal C/C++. You also have to manage and convert all your python types to C types. Let's not even mention the GIL. There is a lot of room for error here.

2) Use `ctypes`

```C
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

This seems like a good choice except if you ever use a C++ feature then `ctypes` is not compatible. This is meant for C types obviously. The other issue is loading the dll. depending on the OS you may need to load different dlls based on what the user is using for their OS. This line won't work on every machine `testlib = ctypes.CDLL('/full/path/to/testlib.so')`. The question of memory is still up for grabs since it could be the compiled C program or python that needs to manage it.

3) `CFFI` (C Foreign Function Interface for Python)

Another interface that allows you to call C and once again is not compatible with C++. You are going to have to write some nasty wrappers. The build process also grows quickly

Here is a basic example

```C
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

Let's be honest when you first see this you have no clue what is going on. The `cdef` is a string... This is the simplest example and it only gets harder from here. I have also seen some people complain about padding when you compile using this so the ways that the data gets shifted could cause issues for you. Due to the lack of c++ support, building complexity, and memory management this is not a great choice. It is probably a bit easier and portable than `ctypes` since you don't have to define all the ctypes manually but that still does not gain you much. CFFI can also check your binding types at compile time and save you some runtime crashes.