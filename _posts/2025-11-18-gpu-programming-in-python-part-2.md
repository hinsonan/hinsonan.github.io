---
layout: post
title: "GPU Programming in Python Part 2: Merging with the green machine using cuda-python"
date: 2025-11-18
categories: ML
---

Have you ever wanted to escape the clutches of python and finally convince your team and managers that this language should not be used for real programming? Are you ready to commit to the world of C/C++ and write some performant code? Perhaps you want to write some CUDA and run at speeds unimaginable to man. Well I got a surprise for you. You ain't ever leaving this god-forsaken wasteland of python because now nvidia has exposed all of that CUDA goodness right into python. You can now write all the kernels you want in python. You can call the exact same bindings that you would in C/C++. You are stuck forever in this language. Yugi's Grandpa has banished us into the shadow realm and we are never leaving. It's time to get comfortable and get good with `cuda-python`.

# What is Cuda Python

As you can expect it is literally CUDA in python. What a wild trip this is. [Cuda Python](https://github.com/NVIDIA/cuda-python) is a metapackage that is made up of different subpackages. They are broken up into

* cuda.core: Pythonic access to CUDA Runtime and other core functionalities
* cuda.bindings: Low-level Python bindings to CUDA C APIs
* cuda.pathfinder: Utilities for locating CUDA components installed in the user's Python environment
* cuda.cccl.cooperative: A Python module providing CCCL's reusable block-wide and warp-wide device primitives for use within Numba CUDA kernels
* cuda.cccl.parallel: A Python module for easy access to CCCL's highly efficient and customizable parallel algorithms, like sort, scan, reduce, transform, etc. that are callable on the host
* numba.cuda: Numba's target for CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model.
* nvmath-python: Pythonic access to NVIDIA CPU & GPU Math Libraries, with both host and device (nvmath.device) APIs. It also provides low-level Python bindings to host C APIs (nvmath.bindings).

for this article we are going to focus on `cuda.bindings` and `cuda.core`

# Installing

Install [cuda-bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html)

Install [cuda-core](https://nvidia.github.io/cuda-python/cuda-core/0.3.2/install.html)

as of time of this writing I am using `cuda-bindings==12.9.3`, `cuda-python==12.9.3`, and `cuda-core==0.3.2`

# Gaussian Blur with CUDA Python

Our goal will be to perform a gaussian using cuda bindings and cuda core. 

## Going Low Level with CUDA Bindings

Let's start with the low level operations and use cuda bindings to perform gaussian blur on an image. This goes pretty deep and I will hit the highlights during this article and have links for the full code at the bottom. The first thing we have to do is create a C++ kernel string...yes we are creating a CUDA kernel in a string.

```python
def generate_gaussian_kernel_code(kernel_size, sigma):
    # Create Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # Scale to integers for better precision in GPU
    scale_factor = 1000
    kernel_int = np.round(kernel * scale_factor).astype(int)
    print(f"Kernel: {kernel_int}")
    kernel_sum = kernel_int.sum()
    print(f"Kernel Sum: {kernel_sum}")

    # Generate kernel array as C++ string
    kernel_str = "{\n"
    for row in kernel_int:
        kernel_str += "        {" + ", ".join(map(str, row)) + "},\n"
    kernel_str += "    }"

    # Generate CUDA code
    offset = kernel_size // 2

    code = f"""
extern "C" __global__
void gaussian_blur(const unsigned char* input, 
                   unsigned char* output,
                   int width, int height, int channels)
{{
    // Calculate 2D pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // {kernel_size}x{kernel_size} Gaussian kernel (sigma={sigma})
    const float kernel[{kernel_size}][{kernel_size}] = {kernel_str};
    const float kernel_sum = {kernel_sum}.0f;
    
    // Process each channel (grayscale=1, RGB=3, RGBA=4)
    for (int c = 0; c < channels; c++) {{
        float sum = 0.0f;
        
        // Apply convolution: weighted sum of neighbors
        for (int ky = 0; ky < {kernel_size}; ky++) {{
            for (int kx = 0; kx < {kernel_size}; kx++) {{
                // Calculate neighbor position
                int nx = x + kx - {offset};
                int ny = y + ky - {offset};
                
                // Clamp to image edges
                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));
                
                // Accumulate weighted pixel value
                int idx = (ny * width + nx) * channels + c;
                sum += input[idx] * kernel[ky][kx];
            }}
        }}
        
        // Write normalized result
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = (unsigned char)(sum / kernel_sum);
    }}
}}
"""
    return code
```

this string ends up looking like this

```c++
extern "C" __global__
void gaussian_blur(const unsigned char* input, 
                   unsigned char* output,
                   int width, int height, int channels)
{
    // Calculate 2D pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 7x7 Gaussian kernel (sigma=1.5)
    const float kernel[7][7] = {
        {18, 56, 108, 135, 108, 56, 18},
        {56, 169, 329, 411, 329, 169, 56},
        {108, 329, 641, 801, 641, 329, 108},
        {135, 411, 801, 1000, 801, 411, 135},
        {108, 329, 641, 801, 641, 329, 108},
        {56, 169, 329, 411, 329, 169, 56},
        {18, 56, 108, 135, 108, 56, 18},
    };
    const float kernel_sum = 13644.0f;

    // Process each channel (grayscale=1, RGB=3, RGBA=4)
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;

        // Apply convolution: weighted sum of neighbors
        for (int ky = 0; ky < 7; ky++) {
            for (int kx = 0; kx < 7; kx++) {
                // Calculate neighbor position
                int nx = x + kx - 3;
                int ny = y + ky - 3;

                // Clamp to image edges
                nx = max(0, min(nx, width - 1));
                ny = max(0, min(ny, height - 1));

                // Accumulate weighted pixel value
                int idx = (ny * width + nx) * channels + c;
                sum += input[idx] * kernel[ky][kx];
            }
        }

        // Write normalized result
        int out_idx = (y * width + x) * channels + c;
        output[out_idx] = (unsigned char)(sum / kernel_sum);
    }
}
```

The power of a GPU is the parallel processing. For a CPU based approach, gaussian blur requires sequential operations. you need to loop through the rows and columns and apply the filter for all pixels. For the GPU we can process the blur for each pixel simultaneously using all the threads that are available on the GPU