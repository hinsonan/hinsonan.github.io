---
layout: post
title: "GPU Programming in Python Part 2: Merging with the Green Machine using cuda-python"
date: 2025-11-16
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

For this article we are going to focus on `cuda.bindings` and `cuda.core`

# Installing

Install [cuda-bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html)

Install [cuda-core](https://nvidia.github.io/cuda-python/cuda-core/0.3.2/install.html)

as of time of this writing I am using `cuda-bindings==12.9.3`, `cuda-python==12.9.3`, and `cuda-core==0.3.2`

# Gaussian Blur with CUDA Python

Our goal will be to perform a gaussian blur using cuda bindings and cuda core. 

## Going Low Level with CUDA Bindings

Let's start with the low level operations and use cuda bindings to perform gaussian blur on an image. This goes pretty deep and I will hit the highlights during this article and have links for the full code at the bottom. The first thing we have to do is create a C++ kernel string...yes we are creating a CUDA kernel in a string.

{% raw %}
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
{% endraw %}

this string ends up looking like this

{% raw %}
```c
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
{% endraw %}

The power of a GPU is the parallel processing. For a CPU based approach, gaussian blur requires sequential operations. You need to loop through the rows and columns and apply the filter for all pixels. For the GPU we can process the blur for each pixel simultaneously using all the threads that are available on the GPU.

Now in order to use this kernel code we have to do the following steps

<div class="mermaid">
graph TD
    
    A["Input Image (CPU)"] --> B["Initialize CUDA"]
    B --> C["Get GPU device handle"]
    C --> D["Compile Kernel"]
    D --> E["CUDA C++ to PTX"]
    E --> F["Load Module"]
    F --> G["PTX to GPU machine code"]
    G --> H["Allocate Memory"]
    H --> I["Reserve space on GPU"]
    I --> J["Transfer to GPU"]
    J --> K["Copy image data\n(CPU → GPU)"]
    K --> L["Launch Kernel"]
    L --> M["Execute parallel blur"]
    M --> N["Transfer from GPU"]
    N --> O["Copy result\n(GPU → CPU)"]
    O --> P["Cleanup"]
    P --> Q["Free all resources"]
    Q --> R["Output Image (CPU)"]
</div>

Now get ready to write some CUDA using python. At this point you might be wanting to switch over to C but remember we are trapped in a prison and python is all we have. The full code will be posted at the bottom so we will hit the highlights in this article.

## Finding Our GPU

```python
# cuInit(Flags) - Initialize the CUDA Driver API
# Must be called before any other Driver API function
# Flags: Must be 0 (reserved for future use)
# Returns: (CUresult, None) - error code and no return value
# This function initializes the CUDA driver and discovers available GPUs
checkCudaErrors(driver.cuInit(0))

# cuDeviceGet(ordinal) - Get handle for compute device
# ordinal: Device index (0 = first GPU, 1 = second GPU, etc.)
# Returns: (CUresult, CUdevice) - error code and device handle
# CUdevice is an opaque handle (integer) representing the physical GPU
device = checkCudaErrors(driver.cuDeviceGet(0))

# Get device information for display and compilation

# cuDeviceGetName(len, dev) - Get device name string
# len: Maximum length of name buffer (typically 256)
# dev: Device handle from cuDeviceGet
# Returns: (CUresult, bytes) - error code and device name as bytes
# Example return: b'NVIDIA GeForce RTX 3080'
device_name = checkCudaErrors(driver.cuDeviceGetName(256, device))

# cuDeviceGetAttribute(attrib, dev) - Query device attribute
# CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Major version number
# Compute capability defines available GPU features (e.g., 7.5, 8.0, 8.6, 9.0)
# Returns: (CUresult, int) - error code and attribute value
# Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
major = checkCudaErrors(driver.cuDeviceGetAttribute(
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))

# CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Minor version number
# Together with major: determines instruction set and GPU features available
# Example: major=7, minor=5 → compute capability 7.5 (Turing architecture)
minor = checkCudaErrors(driver.cuDeviceGetAttribute(
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))
    
```
## PTX Code Incoming

These are the bindings that tell CUDA your GPU card and the CUDA capabilities that GPU has. Now let's go over how to compile our beautiful C++ string that we created.

```python
# Reference: https://docs.nvidia.com/cuda/nvrtc/index.html
    print(f"\n{'='*60}")
    print("Step 2: Compile CUDA Kernel")
    print(f"{'='*60}")
    
    # Generate CUDA C++ kernel source code as string
    kernel_code = generate_gaussian_kernel_code(kernel_size, sigma)
    
    # nvrtcCreateProgram(src, name, numHeaders, headers, includeNames)
    # Create NVRTC program object from source code
    # src: CUDA C++ source code as bytes (kernel code)
    # name: Program name for error messages (virtual filename)
    # numHeaders: Number of header files to include (0 = none)
    # headers: Array of header file contents ([] = empty)
    # includeNames: Array of header file names ([] = empty)
    # Returns: (nvrtcResult, nvrtcProgram) - error code and program handle
    # The program handle is used for subsequent compilation operations
    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(
        kernel_code.encode(),      # Convert string to bytes (UTF-8)
        b"gaussian_blur.cu",       # Virtual filename (for error messages)
        0,                         # No additional headers
        [],                        # No header contents
        []))                       # No header names
    
    # Construct architecture flag for GPU compute capability
    # Format: "--gpu-architecture=compute_XY" where X=major, Y=minor
    # Example: "compute_75" for compute capability 7.5
    # This tells the compiler which GPU instruction set to target
    # Using wrong architecture may cause "PTX JIT compilation failed" errors
    arch_arg = f'--gpu-architecture=compute_{major}{minor}'.encode()
    
    # nvrtcCompileProgram(prog, numOptions, options)
    # Compile CUDA C++ source code to PTX (Parallel Thread Execution) assembly
    # prog: Program handle from nvrtcCreateProgram
    # numOptions: Number of compiler flags (1 in this case)
    # options: Array of compiler flags as bytes
    # Common options:
    #   --gpu-architecture=compute_XX: Target architecture
    #   --fmad=false: Disable fused multiply-add optimizations
    #   --use_fast_math: Enable fast math optimizations
    #   -O3: Optimization level 3
    # Returns: (nvrtcResult, None) - error code and no return value
    # PTX is an intermediate assembly language that can run on any CUDA GPU
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, 1, [arch_arg]))
    
    # nvrtcGetPTXSize(prog) - Get size of compiled PTX code in bytes
    # prog: Compiled program handle
    # Returns: (nvrtcResult, size_t) - error code and size in bytes
    # We need to know the size before allocating a buffer to hold the PTX
    ptx_size = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    
    # Create buffer to hold PTX code
    # PTX is text-based assembly language (human-readable)
    # Initialize with spaces - these will be overwritten by nvrtcGetPTX
    ptx = b" " * ptx_size
    
    # nvrtcGetPTX(prog, ptx) - Retrieve compiled PTX assembly code
    # prog: Compiled program handle
    # ptx: Pre-allocated buffer to receive PTX code (modified in-place)
    # Returns: (nvrtcResult, None) - error code and no return value
    # The PTX code is written directly into the buffer we provided
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
    
    print(f"  Kernel compiled successfully")

    print("\n" + "="*60)
    print("GENERATED PTX CODE")
    print("="*60)
    print(ptx.decode('utf-8'))
    print("="*60 + "\n")
```

Since our code is self contained we don't need additional headers. If you had macros or other functions you could add them. Notice how for our output we have some PTX (Parallel Thread Execution) virtual assembly language. This is the most portable option and can be used across different GPUs. It is also human readable and can be stepped through and checked for any potential issues.

<details>
<summary>PTX Code</summary>

{% raw %}
<pre><code>//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-35583870
// Cuda compilation tools, release 12.8, V12.8.93
// Based on NVVM 20.0.0
//

.version 8.7
.target sm_120
.address_size 64

	// .globl	gaussian_blur

.visible .entry gaussian_blur(
	.param .u64 gaussian_blur_param_0,
	.param .u64 gaussian_blur_param_1,
	.param .u32 gaussian_blur_param_2,
	.param .u32 gaussian_blur_param_3,
	.param .u32 gaussian_blur_param_4
)
{
	.reg .pred 	%p<7>;
	.reg .b16 	%rs<50>;
	.reg .b32 	%r<314>;
	.reg .f32 	%f<100>;
	.reg .b64 	%rd<107>;

	ld.param.u32 	%r156, [gaussian_blur_param_2];
	ld.param.u32 	%r159, [gaussian_blur_param_3];
	ld.param.u32 	%r160, [gaussian_blur_param_4];
	mov.u32 	%r161, %ctaid.x;
	mov.u32 	%r162, %ntid.x;
	mov.u32 	%r163, %tid.x;
	mad.lo.s32 	%r1, %r161, %r162, %r163;
	mov.u32 	%r164, %ctaid.y;
	mov.u32 	%r165, %ntid.y;
	mov.u32 	%r166, %tid.y;
	mad.lo.s32 	%r167, %r164, %r165, %r166;
	setp.ge.s32 	%p1, %r1, %r156;
	setp.ge.s32 	%p2, %r167, %r159;
	or.pred  	%p3, %p1, %p2;
	setp.lt.s32 	%p4, %r160, 1;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	$L__BB0_3;
	mad.lo.s32 	%r168, %r156, %r167, %r1;
	mul.lo.s32 	%r264, %r168, %r160;
	add.s32 	%r169, %r1, -3;
	add.s32 	%r170, %r167, -3;
	add.s32 	%r171, %r156, -1;
	add.s32 	%r172, %r159, -1;
	min.s32 	%r173, %r169, %r171;
	max.s32 	%r174, %r173, 0;
	add.s32 	%r175, %r167, 3;
	min.u32 	%r176, %r175, %r172;
	mul.lo.s32 	%r177, %r176, %r156;
	min.s32 	%r178, %r170, %r172;
	max.s32 	%r179, %r178, 0;
	mul.lo.s32 	%r180, %r179, %r156;
	add.s32 	%r181, %r1, -2;
	min.s32 	%r182, %r181, %r171;
	max.s32 	%r183, %r182, 0;
	add.s32 	%r184, %r1, -1;
	min.s32 	%r185, %r184, %r171;
	max.s32 	%r186, %r185, 0;
	min.s32 	%r187, %r1, %r171;
	max.s32 	%r188, %r187, 0;
	add.s32 	%r189, %r1, 1;
	min.s32 	%r190, %r189, %r171;
	max.s32 	%r191, %r190, 0;
	add.s32 	%r192, %r1, 2;
	min.s32 	%r193, %r192, %r171;
	max.s32 	%r194, %r193, 0;
	add.s32 	%r195, %r1, 3;
	min.s32 	%r196, %r195, %r171;
	max.s32 	%r197, %r196, 0;
	add.s32 	%r198, %r167, -2;
	min.s32 	%r199, %r198, %r172;
	max.s32 	%r200, %r199, 0;
	mul.lo.s32 	%r201, %r200, %r156;
	max.s32 	%r202, %r167, 1;
	add.s32 	%r203, %r202, -1;
	mul.lo.s32 	%r204, %r203, %r156;
	add.s32 	%r205, %r174, %r204;
	mul.lo.s32 	%r279, %r205, %r160;
	add.s32 	%r206, %r183, %r204;
	mul.lo.s32 	%r280, %r206, %r160;
	add.s32 	%r207, %r186, %r204;
	mul.lo.s32 	%r281, %r207, %r160;
	add.s32 	%r208, %r188, %r204;
	mul.lo.s32 	%r282, %r208, %r160;
	add.s32 	%r209, %r191, %r204;
	add.s32 	%r210, %r194, %r204;
	mul.lo.s32 	%r284, %r210, %r160;
	add.s32 	%r211, %r197, %r204;
	mul.lo.s32 	%r285, %r211, %r160;
	min.u32 	%r212, %r167, %r172;
	mul.lo.s32 	%r213, %r212, %r156;
	add.s32 	%r214, %r174, %r213;
	add.s32 	%r215, %r183, %r213;
	add.s32 	%r216, %r186, %r213;
	add.s32 	%r217, %r188, %r213;
	add.s32 	%r218, %r194, %r213;
	add.s32 	%r219, %r197, %r213;
	add.s32 	%r220, %r167, 1;
	min.u32 	%r221, %r220, %r172;
	mul.lo.s32 	%r222, %r221, %r156;
	add.s32 	%r223, %r167, 2;
	min.u32 	%r224, %r223, %r172;
	mul.lo.s32 	%r225, %r224, %r156;
	add.s32 	%r226, %r197, %r177;
	mul.lo.s32 	%r313, %r160, %r226;
	add.s32 	%r227, %r194, %r177;
	mul.lo.s32 	%r312, %r160, %r227;
	add.s32 	%r228, %r191, %r177;
	mul.lo.s32 	%r311, %r160, %r228;
	add.s32 	%r229, %r188, %r177;
	mul.lo.s32 	%r310, %r160, %r229;
	add.s32 	%r230, %r186, %r177;
	mul.lo.s32 	%r309, %r160, %r230;
	add.s32 	%r231, %r183, %r177;
	mul.lo.s32 	%r308, %r160, %r231;
	add.s32 	%r232, %r174, %r177;
	mul.lo.s32 	%r307, %r160, %r232;
	add.s32 	%r233, %r197, %r225;
	mul.lo.s32 	%r306, %r160, %r233;
	add.s32 	%r234, %r194, %r225;
	mul.lo.s32 	%r305, %r160, %r234;
	add.s32 	%r235, %r191, %r225;
	mul.lo.s32 	%r304, %r160, %r235;
	add.s32 	%r236, %r188, %r225;
	mul.lo.s32 	%r303, %r160, %r236;
	add.s32 	%r237, %r186, %r225;
	mul.lo.s32 	%r302, %r160, %r237;
	add.s32 	%r238, %r183, %r225;
	mul.lo.s32 	%r301, %r160, %r238;
	add.s32 	%r239, %r174, %r225;
	mul.lo.s32 	%r300, %r160, %r239;
	add.s32 	%r240, %r197, %r222;
	mul.lo.s32 	%r299, %r160, %r240;
	add.s32 	%r241, %r194, %r222;
	mul.lo.s32 	%r298, %r160, %r241;
	add.s32 	%r242, %r191, %r222;
	mul.lo.s32 	%r297, %r160, %r242;
	add.s32 	%r243, %r188, %r222;
	mul.lo.s32 	%r296, %r160, %r243;
	add.s32 	%r244, %r186, %r222;
	mul.lo.s32 	%r295, %r160, %r244;
	add.s32 	%r245, %r183, %r222;
	mul.lo.s32 	%r294, %r160, %r245;
	add.s32 	%r246, %r174, %r222;
	mul.lo.s32 	%r293, %r160, %r246;
	mul.lo.s32 	%r292, %r160, %r219;
	mul.lo.s32 	%r291, %r160, %r218;
	add.s32 	%r247, %r191, %r213;
	mul.lo.s32 	%r290, %r160, %r247;
	mul.lo.s32 	%r289, %r160, %r217;
	mul.lo.s32 	%r288, %r160, %r216;
	mul.lo.s32 	%r287, %r160, %r215;
	mul.lo.s32 	%r286, %r160, %r214;
	mul.lo.s32 	%r283, %r160, %r209;
	add.s32 	%r248, %r197, %r201;
	mul.lo.s32 	%r278, %r160, %r248;
	add.s32 	%r249, %r194, %r201;
	mul.lo.s32 	%r277, %r160, %r249;
	add.s32 	%r250, %r191, %r201;
	mul.lo.s32 	%r276, %r160, %r250;
	add.s32 	%r251, %r188, %r201;
	mul.lo.s32 	%r275, %r160, %r251;
	add.s32 	%r252, %r186, %r201;
	mul.lo.s32 	%r274, %r160, %r252;
	add.s32 	%r253, %r183, %r201;
	mul.lo.s32 	%r273, %r160, %r253;
	add.s32 	%r254, %r174, %r201;
	mul.lo.s32 	%r272, %r160, %r254;
	add.s32 	%r255, %r197, %r180;
	mul.lo.s32 	%r271, %r160, %r255;
	add.s32 	%r256, %r194, %r180;
	mul.lo.s32 	%r270, %r160, %r256;
	add.s32 	%r257, %r191, %r180;
	mul.lo.s32 	%r269, %r160, %r257;
	add.s32 	%r258, %r188, %r180;
	mul.lo.s32 	%r268, %r160, %r258;
	add.s32 	%r259, %r186, %r180;
	mul.lo.s32 	%r267, %r160, %r259;
	add.s32 	%r260, %r183, %r180;
	mul.lo.s32 	%r266, %r160, %r260;
	add.s32 	%r261, %r174, %r180;
	mul.lo.s32 	%r265, %r160, %r261;
	neg.s32 	%r263, %r160;
$L__BB0_2:
	ld.param.u64 	%rd106, [gaussian_blur_param_1];
	ld.param.u64 	%rd105, [gaussian_blur_param_0];
	cvt.s64.s32 	%rd3, %r313;
	cvta.to.global.u64 	%rd4, %rd105;
	add.s64 	%rd5, %rd4, %rd3;
	cvt.s64.s32 	%rd6, %r312;
	add.s64 	%rd7, %rd4, %rd6;
	cvt.s64.s32 	%rd8, %r311;
	add.s64 	%rd9, %rd4, %rd8;
	cvt.s64.s32 	%rd10, %r310;
	add.s64 	%rd11, %rd4, %rd10;
	cvt.s64.s32 	%rd12, %r309;
	add.s64 	%rd13, %rd4, %rd12;
	cvt.s64.s32 	%rd14, %r308;
	add.s64 	%rd15, %rd4, %rd14;
	cvt.s64.s32 	%rd16, %r307;
	add.s64 	%rd17, %rd4, %rd16;
	cvt.s64.s32 	%rd18, %r306;
	add.s64 	%rd19, %rd4, %rd18;
	cvt.s64.s32 	%rd20, %r305;
	add.s64 	%rd21, %rd4, %rd20;
	cvt.s64.s32 	%rd22, %r304;
	add.s64 	%rd23, %rd4, %rd22;
	cvt.s64.s32 	%rd24, %r303;
	add.s64 	%rd25, %rd4, %rd24;
	cvt.s64.s32 	%rd26, %r302;
	add.s64 	%rd27, %rd4, %rd26;
	cvt.s64.s32 	%rd28, %r301;
	add.s64 	%rd29, %rd4, %rd28;
	cvt.s64.s32 	%rd30, %r300;
	add.s64 	%rd31, %rd4, %rd30;
	cvt.s64.s32 	%rd32, %r299;
	add.s64 	%rd33, %rd4, %rd32;
	cvt.s64.s32 	%rd34, %r298;
	add.s64 	%rd35, %rd4, %rd34;
	cvt.s64.s32 	%rd36, %r297;
	add.s64 	%rd37, %rd4, %rd36;
	cvt.s64.s32 	%rd38, %r296;
	add.s64 	%rd39, %rd4, %rd38;
	cvt.s64.s32 	%rd40, %r295;
	add.s64 	%rd41, %rd4, %rd40;
	cvt.s64.s32 	%rd42, %r294;
	add.s64 	%rd43, %rd4, %rd42;
	cvt.s64.s32 	%rd44, %r293;
	add.s64 	%rd45, %rd4, %rd44;
	cvt.s64.s32 	%rd46, %r292;
	add.s64 	%rd47, %rd4, %rd46;
	cvt.s64.s32 	%rd48, %r291;
	add.s64 	%rd49, %rd4, %rd48;
	cvt.s64.s32 	%rd50, %r290;
	add.s64 	%rd51, %rd4, %rd50;
	cvt.s64.s32 	%rd52, %r289;
	add.s64 	%rd53, %rd4, %rd52;
	cvt.s64.s32 	%rd54, %r288;
	add.s64 	%rd55, %rd4, %rd54;
	cvt.s64.s32 	%rd56, %r287;
	add.s64 	%rd57, %rd4, %rd56;
	cvt.s64.s32 	%rd58, %r286;
	add.s64 	%rd59, %rd4, %rd58;
	cvt.s64.s32 	%rd60, %r285;
	add.s64 	%rd61, %rd4, %rd60;
	cvt.s64.s32 	%rd62, %r284;
	add.s64 	%rd63, %rd4, %rd62;
	cvt.s64.s32 	%rd64, %r283;
	add.s64 	%rd65, %rd4, %rd64;
	cvt.s64.s32 	%rd66, %r282;
	add.s64 	%rd67, %rd4, %rd66;
	cvt.s64.s32 	%rd68, %r281;
	add.s64 	%rd69, %rd4, %rd68;
	cvt.s64.s32 	%rd70, %r280;
	add.s64 	%rd71, %rd4, %rd70;
	cvt.s64.s32 	%rd72, %r279;
	add.s64 	%rd73, %rd4, %rd72;
	cvt.s64.s32 	%rd74, %r278;
	add.s64 	%rd75, %rd4, %rd74;
	cvt.s64.s32 	%rd76, %r277;
	add.s64 	%rd77, %rd4, %rd76;
	cvt.s64.s32 	%rd78, %r276;
	add.s64 	%rd79, %rd4, %rd78;
	cvt.s64.s32 	%rd80, %r275;
	add.s64 	%rd81, %rd4, %rd80;
	cvt.s64.s32 	%rd82, %r274;
	add.s64 	%rd83, %rd4, %rd82;
	cvt.s64.s32 	%rd84, %r273;
	add.s64 	%rd85, %rd4, %rd84;
	cvt.s64.s32 	%rd86, %r272;
	add.s64 	%rd87, %rd4, %rd86;
	cvt.s64.s32 	%rd88, %r271;
	add.s64 	%rd89, %rd4, %rd88;
	cvt.s64.s32 	%rd90, %r270;
	add.s64 	%rd91, %rd4, %rd90;
	cvt.s64.s32 	%rd92, %r269;
	add.s64 	%rd93, %rd4, %rd92;
	cvt.s64.s32 	%rd94, %r268;
	add.s64 	%rd95, %rd4, %rd94;
	cvt.s64.s32 	%rd96, %r267;
	add.s64 	%rd97, %rd4, %rd96;
	cvt.s64.s32 	%rd98, %r266;
	add.s64 	%rd99, %rd4, %rd98;
	cvt.s64.s32 	%rd100, %r265;
	add.s64 	%rd101, %rd4, %rd100;
	cvt.s64.s32 	%rd102, %r264;
	cvta.to.global.u64 	%rd103, %rd106;
	add.s64 	%rd104, %rd103, %rd102;
	ld.global.u8 	%rs1, [%rd101];
	cvt.rn.f32.u16 	%f1, %rs1;
	fma.rn.f32 	%f2, %f1, 0f41900000, 0f00000000;
	ld.global.u8 	%rs2, [%rd99];
	cvt.rn.f32.u16 	%f3, %rs2;
	fma.rn.f32 	%f4, %f3, 0f42600000, %f2;
	ld.global.u8 	%rs3, [%rd97];
	cvt.rn.f32.u16 	%f5, %rs3;
	fma.rn.f32 	%f6, %f5, 0f42D80000, %f4;
	ld.global.u8 	%rs4, [%rd95];
	cvt.rn.f32.u16 	%f7, %rs4;
	fma.rn.f32 	%f8, %f7, 0f43070000, %f6;
	ld.global.u8 	%rs5, [%rd93];
	cvt.rn.f32.u16 	%f9, %rs5;
	fma.rn.f32 	%f10, %f9, 0f42D80000, %f8;
	ld.global.u8 	%rs6, [%rd91];
	cvt.rn.f32.u16 	%f11, %rs6;
	fma.rn.f32 	%f12, %f11, 0f42600000, %f10;
	ld.global.u8 	%rs7, [%rd89];
	cvt.rn.f32.u16 	%f13, %rs7;
	fma.rn.f32 	%f14, %f13, 0f41900000, %f12;
	ld.global.u8 	%rs8, [%rd87];
	cvt.rn.f32.u16 	%f15, %rs8;
	fma.rn.f32 	%f16, %f15, 0f42600000, %f14;
	ld.global.u8 	%rs9, [%rd85];
	cvt.rn.f32.u16 	%f17, %rs9;
	fma.rn.f32 	%f18, %f17, 0f43290000, %f16;
	ld.global.u8 	%rs10, [%rd83];
	cvt.rn.f32.u16 	%f19, %rs10;
	fma.rn.f32 	%f20, %f19, 0f43A48000, %f18;
	ld.global.u8 	%rs11, [%rd81];
	cvt.rn.f32.u16 	%f21, %rs11;
	fma.rn.f32 	%f22, %f21, 0f43CD8000, %f20;
	ld.global.u8 	%rs12, [%rd79];
	cvt.rn.f32.u16 	%f23, %rs12;
	fma.rn.f32 	%f24, %f23, 0f43A48000, %f22;
	ld.global.u8 	%rs13, [%rd77];
	cvt.rn.f32.u16 	%f25, %rs13;
	fma.rn.f32 	%f26, %f25, 0f43290000, %f24;
	ld.global.u8 	%rs14, [%rd75];
	cvt.rn.f32.u16 	%f27, %rs14;
	fma.rn.f32 	%f28, %f27, 0f42600000, %f26;
	ld.global.u8 	%rs15, [%rd73];
	cvt.rn.f32.u16 	%f29, %rs15;
	fma.rn.f32 	%f30, %f29, 0f42D80000, %f28;
	ld.global.u8 	%rs16, [%rd71];
	cvt.rn.f32.u16 	%f31, %rs16;
	fma.rn.f32 	%f32, %f31, 0f43A48000, %f30;
	ld.global.u8 	%rs17, [%rd69];
	cvt.rn.f32.u16 	%f33, %rs17;
	fma.rn.f32 	%f34, %f33, 0f44204000, %f32;
	ld.global.u8 	%rs18, [%rd67];
	cvt.rn.f32.u16 	%f35, %rs18;
	fma.rn.f32 	%f36, %f35, 0f44484000, %f34;
	ld.global.u8 	%rs19, [%rd65];
	cvt.rn.f32.u16 	%f37, %rs19;
	fma.rn.f32 	%f38, %f37, 0f44204000, %f36;
	ld.global.u8 	%rs20, [%rd63];
	cvt.rn.f32.u16 	%f39, %rs20;
	fma.rn.f32 	%f40, %f39, 0f43A48000, %f38;
	ld.global.u8 	%rs21, [%rd61];
	cvt.rn.f32.u16 	%f41, %rs21;
	fma.rn.f32 	%f42, %f41, 0f42D80000, %f40;
	ld.global.u8 	%rs22, [%rd59];
	cvt.rn.f32.u16 	%f43, %rs22;
	fma.rn.f32 	%f44, %f43, 0f43070000, %f42;
	ld.global.u8 	%rs23, [%rd57];
	cvt.rn.f32.u16 	%f45, %rs23;
	fma.rn.f32 	%f46, %f45, 0f43CD8000, %f44;
	ld.global.u8 	%rs24, [%rd55];
	cvt.rn.f32.u16 	%f47, %rs24;
	fma.rn.f32 	%f48, %f47, 0f44484000, %f46;
	ld.global.u8 	%rs25, [%rd53];
	cvt.rn.f32.u16 	%f49, %rs25;
	fma.rn.f32 	%f50, %f49, 0f447A0000, %f48;
	ld.global.u8 	%rs26, [%rd51];
	cvt.rn.f32.u16 	%f51, %rs26;
	fma.rn.f32 	%f52, %f51, 0f44484000, %f50;
	ld.global.u8 	%rs27, [%rd49];
	cvt.rn.f32.u16 	%f53, %rs27;
	fma.rn.f32 	%f54, %f53, 0f43CD8000, %f52;
	ld.global.u8 	%rs28, [%rd47];
	cvt.rn.f32.u16 	%f55, %rs28;
	fma.rn.f32 	%f56, %f55, 0f43070000, %f54;
	ld.global.u8 	%rs29, [%rd45];
	cvt.rn.f32.u16 	%f57, %rs29;
	fma.rn.f32 	%f58, %f57, 0f42D80000, %f56;
	ld.global.u8 	%rs30, [%rd43];
	cvt.rn.f32.u16 	%f59, %rs30;
	fma.rn.f32 	%f60, %f59, 0f43A48000, %f58;
	ld.global.u8 	%rs31, [%rd41];
	cvt.rn.f32.u16 	%f61, %rs31;
	fma.rn.f32 	%f62, %f61, 0f44204000, %f60;
	ld.global.u8 	%rs32, [%rd39];
	cvt.rn.f32.u16 	%f63, %rs32;
	fma.rn.f32 	%f64, %f63, 0f44484000, %f62;
	ld.global.u8 	%rs33, [%rd37];
	cvt.rn.f32.u16 	%f65, %rs33;
	fma.rn.f32 	%f66, %f65, 0f44204000, %f64;
	ld.global.u8 	%rs34, [%rd35];
	cvt.rn.f32.u16 	%f67, %rs34;
	fma.rn.f32 	%f68, %f67, 0f43A48000, %f66;
	ld.global.u8 	%rs35, [%rd33];
	cvt.rn.f32.u16 	%f69, %rs35;
	fma.rn.f32 	%f70, %f69, 0f42D80000, %f68;
	ld.global.u8 	%rs36, [%rd31];
	cvt.rn.f32.u16 	%f71, %rs36;
	fma.rn.f32 	%f72, %f71, 0f42600000, %f70;
	ld.global.u8 	%rs37, [%rd29];
	cvt.rn.f32.u16 	%f73, %rs37;
	fma.rn.f32 	%f74, %f73, 0f43290000, %f72;
	ld.global.u8 	%rs38, [%rd27];
	cvt.rn.f32.u16 	%f75, %rs38;
	fma.rn.f32 	%f76, %f75, 0f43A48000, %f74;
	ld.global.u8 	%rs39, [%rd25];
	cvt.rn.f32.u16 	%f77, %rs39;
	fma.rn.f32 	%f78, %f77, 0f43CD8000, %f76;
	ld.global.u8 	%rs40, [%rd23];
	cvt.rn.f32.u16 	%f79, %rs40;
	fma.rn.f32 	%f80, %f79, 0f43A48000, %f78;
	ld.global.u8 	%rs41, [%rd21];
	cvt.rn.f32.u16 	%f81, %rs41;
	fma.rn.f32 	%f82, %f81, 0f43290000, %f80;
	ld.global.u8 	%rs42, [%rd19];
	cvt.rn.f32.u16 	%f83, %rs42;
	fma.rn.f32 	%f84, %f83, 0f42600000, %f82;
	ld.global.u8 	%rs43, [%rd17];
	cvt.rn.f32.u16 	%f85, %rs43;
	fma.rn.f32 	%f86, %f85, 0f41900000, %f84;
	ld.global.u8 	%rs44, [%rd15];
	cvt.rn.f32.u16 	%f87, %rs44;
	fma.rn.f32 	%f88, %f87, 0f42600000, %f86;
	ld.global.u8 	%rs45, [%rd13];
	cvt.rn.f32.u16 	%f89, %rs45;
	fma.rn.f32 	%f90, %f89, 0f42D80000, %f88;
	ld.global.u8 	%rs46, [%rd11];
	cvt.rn.f32.u16 	%f91, %rs46;
	fma.rn.f32 	%f92, %f91, 0f43070000, %f90;
	ld.global.u8 	%rs47, [%rd9];
	cvt.rn.f32.u16 	%f93, %rs47;
	fma.rn.f32 	%f94, %f93, 0f42D80000, %f92;
	ld.global.u8 	%rs48, [%rd7];
	cvt.rn.f32.u16 	%f95, %rs48;
	fma.rn.f32 	%f96, %f95, 0f42600000, %f94;
	ld.global.u8 	%rs49, [%rd5];
	cvt.rn.f32.u16 	%f97, %rs49;
	fma.rn.f32 	%f98, %f97, 0f41900000, %f96;
	div.rn.f32 	%f99, %f98, 0f46553000;
	cvt.rzi.u32.f32 	%r262, %f99;
	st.global.u8 	[%rd104], %r262;
	add.s32 	%r313, %r313, 1;
	add.s32 	%r312, %r312, 1;
	add.s32 	%r311, %r311, 1;
	add.s32 	%r310, %r310, 1;
	add.s32 	%r309, %r309, 1;
	add.s32 	%r308, %r308, 1;
	add.s32 	%r307, %r307, 1;
	add.s32 	%r306, %r306, 1;
	add.s32 	%r305, %r305, 1;
	add.s32 	%r304, %r304, 1;
	add.s32 	%r303, %r303, 1;
	add.s32 	%r302, %r302, 1;
	add.s32 	%r301, %r301, 1;
	add.s32 	%r300, %r300, 1;
	add.s32 	%r299, %r299, 1;
	add.s32 	%r298, %r298, 1;
	add.s32 	%r297, %r297, 1;
	add.s32 	%r296, %r296, 1;
	add.s32 	%r295, %r295, 1;
	add.s32 	%r294, %r294, 1;
	add.s32 	%r293, %r293, 1;
	add.s32 	%r292, %r292, 1;
	add.s32 	%r291, %r291, 1;
	add.s32 	%r290, %r290, 1;
	add.s32 	%r289, %r289, 1;
	add.s32 	%r288, %r288, 1;
	add.s32 	%r287, %r287, 1;
	add.s32 	%r286, %r286, 1;
	add.s32 	%r285, %r285, 1;
	add.s32 	%r284, %r284, 1;
	add.s32 	%r283, %r283, 1;
	add.s32 	%r282, %r282, 1;
	add.s32 	%r281, %r281, 1;
	add.s32 	%r280, %r280, 1;
	add.s32 	%r279, %r279, 1;
	add.s32 	%r278, %r278, 1;
	add.s32 	%r277, %r277, 1;
	add.s32 	%r276, %r276, 1;
	add.s32 	%r275, %r275, 1;
	add.s32 	%r274, %r274, 1;
	add.s32 	%r273, %r273, 1;
	add.s32 	%r272, %r272, 1;
	add.s32 	%r271, %r271, 1;
	add.s32 	%r270, %r270, 1;
	add.s32 	%r269, %r269, 1;
	add.s32 	%r268, %r268, 1;
	add.s32 	%r267, %r267, 1;
	add.s32 	%r266, %r266, 1;
	add.s32 	%r265, %r265, 1;
	add.s32 	%r264, %r264, 1;
	add.s32 	%r263, %r263, 1;
	setp.ne.s32 	%p6, %r263, 0;
	@%p6 bra 	$L__BB0_2;
$L__BB0_3:
	ret;

}
</code></pre>
{% endraw %}

</details>


There are other options to compile to. NVRTC can be compiled down to binary for better performance but this makes it specific to the GPU architecture that you are using.

## Launching the Kernel

Now we will launch this kernel and blur our image.

```python
# ============================================================
    # Step 6: Configure and Launch Kernel
    # ============================================================
    # Reference: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
    print(f"\n{'='*60}")
    print("Step 6: Launch Kernel")
    print(f"{'='*60}")
    
    # Configure 2D grid of thread blocks for parallel image processing
    # Each thread processes exactly one pixel
    # Grid = collection of blocks, Block = collection of threads
    
    # Block dimensions: threads per block in each dimension
    # 16×16 = 256 threads per block (common choice for 2D workloads)
    # Block size affects:
    #   - Occupancy (threads per SM)
    #   - Register usage per thread
    #   - Shared memory per block
    # Typical block sizes: 16×16, 32×8, or 256×1
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    
    # Grid dimensions: number of blocks needed to cover entire image
    # Formula: ceil(dimension / block_size) = (dimension + block_size - 1) // block_size
    # This ensures we have enough threads even if image size isn't divisible by block size
    # Example: width=800, BLOCK_SIZE_X=16 → GRID_SIZE_X = (800+15)//16 = 50 blocks
    # Some threads in the last block may be out of bounds (handled by kernel)
    GRID_SIZE_X = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    GRID_SIZE_Y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    print(f"  Grid:  {GRID_SIZE_X} × {GRID_SIZE_Y} blocks")
    print(f"  Block: {BLOCK_SIZE_X} × {BLOCK_SIZE_Y} threads")
    print(f"  Total: {GRID_SIZE_X * GRID_SIZE_Y * BLOCK_SIZE_X * BLOCK_SIZE_Y:,} threads")
    
    # ============================================================
    # Prepare Kernel Arguments (NumPy approach)
    # Reference: https://nvidia.github.io/cuda-python/cuda-bindings/latest/overview.html#preparing-kernel-arguments
    # ============================================================
    
    # Kernel signature (from generated CUDA code):
    # extern "C" __global__ void gaussian_blur(
    #     const unsigned char* input,    // Pointer → np.intp
    #     unsigned char* output,         // Pointer → np.intp
    #     int width,                     // Integer → np.int32
    #     int height,                    // Integer → np.int32
    #     int channels)                  // Integer → np.int32
    
    # CRITICAL: Each argument must be wrapped in a NumPy array
    # This provides a stable memory location and ctypes.data attribute
    # Type mapping reference:
    #   - Pointers (any type*): np.intp (platform-aware pointer size)
    #   - int: np.int32
    #   - unsigned int: np.uint32
    #   - float: np.float32
    #   - double: np.float64
    
    # Prepare each kernel argument as a NumPy array
    args_data = [
        # Argument 1: input pointer (GPU memory address)
        # int(d_input) converts CUdeviceptr to Python int
        # np.array([...], dtype=np.intp) wraps in platform-aware pointer type
        # On 64-bit systems: np.intp = np.int64
        # On 32-bit systems: np.intp = np.int32
        np.array([int(d_input)], dtype=np.intp),
        
        # Argument 2: output pointer (GPU memory address)
        np.array([int(d_output)], dtype=np.intp),
        
        # Argument 3: width (image width in pixels)
        # np.int32 matches C int type (4 bytes, signed)
        np.array([width], dtype=np.int32),
        
        # Argument 4: height (image height in pixels)
        np.array([height], dtype=np.int32),
        
        # Argument 5: channels (number of color channels: 1, 3, or 4)
        np.array([channels], dtype=np.int32)
    ]
    
    # Create contiguous array of pointers to each argument
    # cuLaunchKernel expects: void** kernelParams (pointer to array of pointers)
    # arg.ctypes.data gives memory address of each NumPy array
    # We create an array containing these addresses
    # Conceptually: void* args[] = {&input, &output, &width, &height, &channels}
    # Then kernelParams points to args
    args = np.array([arg.ctypes.data for arg in args_data], dtype=np.intp)
    
    # cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
    #                sharedMemBytes, hStream, kernelParams, extra)
    # Launch CUDA kernel with specified configuration
    # Reference: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
    #
    # f: Kernel function handle from cuModuleGetFunction
    # gridDimX, gridDimY, gridDimZ: Grid dimensions (blocks per grid)
    #   - Total blocks = gridDimX × gridDimY × gridDimZ
    #   - We use 2D grid (gridDimZ=1) for image processing
    # blockDimX, blockDimY, blockDimZ: Block dimensions (threads per block)
    #   - Total threads per block = blockDimX × blockDimY × blockDimZ
    #   - Max threads per block: typically 1024 (depends on GPU)
    # sharedMemBytes: Dynamic shared memory per block in bytes (0 = none)
    #   - Static shared memory declared in kernel doesn't count toward this
    # hStream: Stream to execute kernel in (for async execution)
    # kernelParams: Pointer to array of argument pointers (void**)
    #   - Must point to contiguous array in memory
    #   - Each element is a pointer to an argument
    # extra: Reserved parameter (must be 0, for future CUDA versions)
    #
    # Returns: (CUresult, None) - error code and no return value
    # Kernel launch is asynchronous - returns immediately
    checkCudaErrors(driver.cuLaunchKernel(
        kernel,                    # Function: Compiled kernel to execute
        GRID_SIZE_X,               # Grid X: Blocks in X dimension
        GRID_SIZE_Y,               # Grid Y: Blocks in Y dimension
        1,                         # Grid Z: 1 (2D grid, not 3D)
        BLOCK_SIZE_X,              # Block X: Threads per block in X
        BLOCK_SIZE_Y,              # Block Y: Threads per block in Y
        1,                         # Block Z: 1 (2D block, not 3D)
        0,                         # Shared memory: 0 bytes (none needed)
        stream,                    # Stream: Execute in this command queue
        args.ctypes.data,          # Arguments: Pointer to argument array
        0))                        # Extra: Reserved (must be 0)
    
    print(f"  Kernel launched")
```

The first thing we need to grasp is the blocks and grid size. 

<style>
.gpu-table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
  font-size: 0.9em;
}

.gpu-table thead tr {
  background-color: #2c3e50;
  color: #ffffff;
  text-align: left;
}

.gpu-table th,
.gpu-table td {
  padding: 12px 15px;
  border: 1px solid #ddd;
}

.gpu-table tbody tr {
  border-bottom: 1px solid #dddddd;
}

.gpu-table tbody tr:nth-of-type(even) {
  background-color: #f3f3f3;
}

.gpu-table tbody tr:nth-of-type(odd) {
  background-color: #ffffff;
}

.gpu-table tbody tr:hover {
  background-color: #e8f4f8;
  cursor: pointer;
}

.gpu-table tbody td:first-child {
  font-weight: bold;
  background-color: #ecf0f1;
}
</style>

<div class="gpu-table-wrapper" markdown="block">

| Concept | What It Is | Size Limits | Can Communicate? | Where It Runs | Example |
|---------|-----------|-------------|------------------|---------------|---------|
| **Thread** | Single execution unit | N/A (smallest unit) | With threads in same block only | Inside a block on an SM | Thread 5 processes pixel (85, 160) |
| **Block** | Group of threads | 1-1,024 threads<br/>Max dimensions: 1024×1024×64 | Threads within block can sync/share | Entire block on one SM | Block (5,10) with 256 threads processes 16×16 pixel region |
| **Grid** | All blocks in kernel | X: 2³¹-1 blocks<br/>Y,Z: 65,535 blocks | Blocks are independent | Distributed across all SMs | Grid of 1,900 blocks covers entire 800×600 image |
| **Warp** | 32 consecutive threads | Always 32 threads | Execute same instruction | Scheduled together on SM | Threads 0-31 in a block = 1 warp |
| **SM** | Streaming Multiprocessor | Hardware unit | N/A | Physical GPU chip | RTX 5090 has 170 SMs, each runs multiple blocks |
{: .gpu-table}

</div>

### Helpful Definitions

**Thread**: A thread is a unit of execution and each thread will operate on different data

**Block**: A group of threads that can be executed sequentially or in parallel. Threads in the same block can share some memory and communicate

**Grid**: CUDA blocks are grouped together and form grids

**Warp**: GPUs execute groups of threads known as warps. Warps are always 32 parallel threads that execute the same instruction simultaneously. (NVIDIA has always used 32 threads in order to be backwards compatible and performant when grabbing memory)

**SM (Streaming Multiprocessor)**: Blocks get distributed among the amount of SMs that your GPU has.

Here is a flow of how this blur will work on a 5090.

<div class="mermaid">
graph TB
    GPU["<b>NVIDIA RTX 5090 (Blackwell)</b><br/>170 SMs × 128 CUDA cores/SM = 21,760 CUDA cores<br/>32 GB GDDR7 @ 1,792 GB/s bandwidth<br/>575W TGP"]
    
    Image["<b>IMAGE: 800×600 RGB</b><br/>480,000 pixels × 3 channels<br/>1,440,000 bytes"]
    
    subgraph Kernel["<b>7×7 GAUSSIAN KERNEL</b>"]
        K1["<b>49 weights per pixel</b><br/>Centered at target pixel<br/>Larger kernel = more blur"]
    end
    
    subgraph GridLevel["<b>GRID: 50×38 blocks</b>"]
        Grid["<b>1,900 blocks total</b><br/>Cover entire 800×600 image<br/>(800÷16) × (600÷16)"]
    end
    
    subgraph BlockLevel["<b>BLOCK: 16×16 threads</b>"]
        Block["<b>256 threads per block</b><br/>Each thread = 1 pixel<br/>Organized in 8 warps of 32 threads"]
    end
    
    subgraph ThreadWork["<b>EACH THREAD:</b>"]
        T1["<b>Position:</b> x = blockIdx.x×16 + threadIdx.x<br/>y = blockIdx.y×16 + threadIdx.y"]
        T2["<b>Read:</b> 49 neighbor pixels<br/>7×7 window around (x,y)"]
        T3["<b>Compute:</b> Weighted sum<br/>pixel[i,j] × kernel[i,j]<br/>for all 49 positions"]
        T4["<b>Write:</b> Normalized result<br/>sum ÷ kernel_total"]
    end
    
    subgraph Parallel["<b>RTX 5090 EXECUTION:</b>"]
        SM["<b>170 SMs available</b><br/>Each SM runs multiple warps<br/>~15,200 warps total<br/>(1,900 blocks × 8 warps/block)"]
        Concurrent["<b>Parallelism:</b><br/>486,400 threads launched<br/>Thousands execute simultaneously"]
    end
    
    GPU --> Image
    Image --> Kernel
    Kernel --> Grid
    Grid --> Block
    Block --> T1
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> SM
    SM --> Concurrent
    
    style GPU fill:#2c3e50,stroke:#3498db,stroke-width:5px,color:#ecf0f1
    style Image fill:#34495e,stroke:#95a5a6,stroke-width:4px,color:#ecf0f1
    style Kernel fill:#e8f4f8,stroke:#5dade2,stroke-width:4px,color:#2c3e50
    style K1 fill:#d6eaf8,stroke:#5dade2,stroke-width:3px,color:#2c3e50
    style GridLevel fill:#e8f8f5,stroke:#52be80,stroke-width:4px,color:#196f3d
    style Grid fill:#d5f4e6,stroke:#52be80,stroke-width:3px,color:#196f3d
    style BlockLevel fill:#fef9e7,stroke:#f39c12,stroke-width:4px,color:#7d6608
    style Block fill:#fcf3cf,stroke:#f39c12,stroke-width:3px,color:#7d6608
    style ThreadWork fill:#fdecea,stroke:#e74c3c,stroke-width:4px,color:#922b21
    style T1 fill:#fadbd8,stroke:#e74c3c,stroke-width:3px,color:#922b21
    style T2 fill:#fadbd8,stroke:#e74c3c,stroke-width:3px,color:#922b21
    style T3 fill:#fadbd8,stroke:#e74c3c,stroke-width:3px,color:#922b21
    style T4 fill:#fadbd8,stroke:#e74c3c,stroke-width:3px,color:#922b21
    style Parallel fill:#f4ecf7,stroke:#8e44ad,stroke-width:4px,color:#4a235a
    style SM fill:#ebdef0,stroke:#8e44ad,stroke-width:3px,color:#4a235a
    style Concurrent fill:#ebdef0,stroke:#8e44ad,stroke-width:3px,color:#4a235a
</div>

## Viewing Our Results

I may have asked for a png image to be generated using shapes in the theme of Dragon Ball Z. This is currently SOTA shape creation so do not complain about artistic abilities. Once we launch our kernel we can finally see the results of our effort.

![blur](/assets/images/gaussian_blur_comparison.png)

It worked we now have a blurry image. Now let's examine the same thing but using the higher level more pythonic library called cuda core

# Cuda Core

Cuda core offers a higher level API than cuda bindings. It's a bit more pythonic so if you love python you may like this more.

One of the larger differences between the two is the compiling and launching code

```python
# Generate CUDA C++ kernel source code as string
# This function creates a templated kernel with Gaussian weights embedded
kernel_code = generate_gaussian_kernel_code_cuda_core(kernel_size, sigma)

# Get architecture string for compilation
# compute_capability is a tuple (major, minor), e.g., (7, 5)
# We join them to create "75" for sm_75 architecture flag
arch = "".join(f"{i}" for i in dev.compute_capability)

# ProgramOptions - Specify compiler options for NVRTC
# This is a dataclass that holds compilation settings
# Parameters:
#   std: C++ standard to use (e.g., "c++11", "c++14", "c++17", "c++20")
#   arch: Target GPU architecture (e.g., "sm_75" for compute capability 7.5)
#        Format: "sm_XY" where X=major, Y=minor
#   Other options can include: optimize, debug, include_paths, etc.
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.ProgramOptions.html
program_options = ProgramOptions(
    std="c++17",           # Use C++17 standard (for modern C++ features)
    arch=f"sm_{arch}"      # Target specific GPU architecture
)

# Program() - Create a program object from source code
# This represents a compilation unit (similar to nvrtcCreateProgram in cuda.bindings)
# Parameters:
#   code: Source code as string (CUDA C++ kernel code)
#   code_type: Type of source code
#       - "c++": CUDA C++ source (most common)
#       - "ptx": PTX assembly
#       - "cubin": Binary CUBIN
#   options: ProgramOptions object with compiler settings
# The Program object manages the NVRTC program lifecycle
# No compilation happens yet - that's done by compile()
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.Program.html
prog = Program(
    kernel_code,           # CUDA C++ source code string
    code_type="c++",       # Specify source is C++ (not PTX/CUBIN)
    options=program_options  # Compiler options
)

# compile() - Compile the program to binary format
# This performs the actual compilation using NVRTC
# Parameters:
#   target_type: Output format
#       - "cubin": Compile to CUBIN (binary, GPU-specific, faster to load)
#       - "ptx": Compile to PTX (assembly, portable across GPUs)
#       - "ltoir": LTO-IR for link-time optimization
#   name_expressions: Tuple of kernel/function names to compile
#       - For templates, specify instantiation: "kernel<float>"
#       - Can compile multiple: ("kernel<float>", "kernel<int>")
#       - Only specified templates are compiled (saves time/space)
# Returns: ObjectCode object containing the compiled binary
# The ObjectCode can be used to extract kernels
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.Program.html#cuda.core.experimental.Program.compile
module = prog.compile(
    "cubin",  # Compile to CUBIN (binary format, faster than PTX)
    name_expressions=("gaussian_blur<unsigned char>",)  # Template instantiation
)

# get_kernel() - Extract a specific kernel from compiled code
# Parameters:
#   name: Kernel function name (must match what was compiled)
#         For templates, use full instantiated name: "kernel<type>"
# Returns: Kernel object that can be launched
# Unlike cuda.bindings (cuModuleGetFunction), no need for bytes encoding
# The Kernel object is a managed wrapper around CUfunction
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.Kernel.html
kernel = module.get_kernel("gaussian_blur<unsigned char>")

print(f"  Kernel compiled successfully")
```

Notice how in this example we compile to binary instead of PTX just to change it up.Now we can launch the kernel

## Launching the Kernel

```python
# ============================================================
# Step 4: Configure and Launch Kernel
# ============================================================
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.launch.html
print(f"\n{'='*60}")
print("Step 4: Launch Kernel")
print(f"{'='*60}")

# Configure 2D grid of thread blocks for parallel image processing
# Each thread processes exactly one pixel

# Block dimensions: threads per block in each dimension
# 16×16 = 256 threads per block (common for 2D image processing)
BLOCK_SIZE_X = 16
BLOCK_SIZE_Y = 16

# Grid dimensions: number of blocks needed to cover entire image
# Formula: ceil(dimension / block_size)
# This ensures enough threads even if image size isn't divisible by block size
GRID_SIZE_X = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
GRID_SIZE_Y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y

print(f"  Grid:  {GRID_SIZE_X} x {GRID_SIZE_Y} blocks")
print(f"  Block: {BLOCK_SIZE_X} x {BLOCK_SIZE_Y} threads")
print(f"  Total: {GRID_SIZE_X * GRID_SIZE_Y * BLOCK_SIZE_X * BLOCK_SIZE_Y:,} threads")

# LaunchConfig - Configure kernel launch parameters (Pythonic!)
# This replaces the complex grid/block parameter passing in cuda.bindings
# Parameters:
#   grid: Grid dimensions as tuple or int
#       - Tuple: (grid_x, grid_y, grid_z) for 2D/3D grids
#       - Int: grid_x (for 1D grid)
#   block: Block dimensions as tuple or int
#       - Tuple: (block_x, block_y, block_z) for 2D/3D blocks
#       - Int: block_x (for 1D block)
#   shared_mem: Dynamic shared memory per block in bytes (default: 0)
#   stream: Stream to execute in (default: None = default stream)
# Much cleaner than cuda.bindings' 11-parameter cuLaunchKernel!
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.LaunchConfig.html
config = LaunchConfig(
    grid=(GRID_SIZE_X, GRID_SIZE_Y),   # 2D grid dimensions
    block=(BLOCK_SIZE_X, BLOCK_SIZE_Y)  # 2D block dimensions
)

# launch() - Launch CUDA kernel (the Pythonic way!)
# This is the high-level kernel launch function
# Parameters:
#   stream: Stream to execute kernel in
#   config: LaunchConfig object with grid/block configuration
#   kernel: Kernel object to execute
#   *args: Kernel arguments (variable length)
# 
# Kernel argument passing:
#   - For GPU pointers: Use .data.ptr from CuPy arrays
#       d_input.data.ptr gets the GPU memory address
#   - For scalars: Pass Python int/float directly
#       cuda.core automatically converts to correct C type
#   - For CuPy types: Use cp.int32(), cp.uint64(), etc. for explicit types
# 
# Type mapping (automatic):
#   Python int → C int (usually int32)
#   Python float → C double
#   cp.int32(x) → C int32_t
#   cp.uint64(x) → C uint64_t
#   cp.float32(x) → C float
# 
# No manual argument preparation needed!
# Compare to cuda.bindings:
#   - No need to wrap args in NumPy arrays
#   - No need to create args array with ctypes.data
#   - No need to manually convert pointers to integers
# 
# The kernel signature in CUDA C++:
#   __global__ void gaussian_blur(
#       const unsigned char* input,   // d_input.data.ptr
#       unsigned char* output,        // d_output.data.ptr  
#       int width,                    // width (Python int → C int)
#       int height,                   // height (Python int → C int)
#       int channels)                 // channels (Python int → C int)
# 
# Returns: None
# Kernel launch is asynchronous - returns immediately
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental.launch.html
launch(
    stream,              # Stream: Execute kernel in this stream
    config,              # Config: Grid and block configuration
    kernel,              # Kernel: Compiled kernel to execute
    d_input.data.ptr,    # Arg 1: Input GPU pointer (unsigned char*)
    d_output.data.ptr,   # Arg 2: Output GPU pointer (unsigned char*)
    width,               # Arg 3: Image width (int)
    height,              # Arg 4: Image height (int)
    channels             # Arg 5: Number of channels (int)
)

print(f"  Kernel launched")

# ============================================================
# Step 5: Synchronize and Retrieve Results
# ============================================================
print(f"\n{'='*60}")
print("Step 5: Synchronize and Get Results")
print(f"{'='*60}")

# sync() - Synchronize stream (wait for all operations to complete)
# This blocks the CPU thread until all operations in the stream finish
# Operations include:
#   - Kernel launches
#   - Memory copies
#   - Any other async operations
# Similar to cuStreamSynchronize in cuda.bindings
# After sync(), d_output contains the blurred image on GPU
# Reference: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.experimental._stream.Stream.html#cuda.core.experimental._stream.Stream.sync
stream.sync()

print(f"  Stream synchronized")

# cp.asnumpy() - Convert CuPy array back to NumPy (copy GPU→CPU)
# This automatically:
#   1. Copies data from GPU to CPU (like cuMemcpyDtoH)
#   2. Returns a NumPy array
# The copy is synchronous - waits for GPU to finish
# CuPy GPU memory remains allocated until d_output is garbage collected
output_array = cp.asnumpy(d_output)

print(f"  Result copied to CPU: {output_array.nbytes:,} bytes")
```

One benefit over cuda core is you do not have to manually clean up the memory like we had to do for the bindings.

## Viewing Result

Again the image is the same

![blur](/assets/images/gaussian_blur_comparison.png)


## Timing Comparisons

GPUs go burrr

```
====================================================================================================
SUMMARY
====================================================================================================
Image Size           CPU (ms)     Bindings (ms)   Core (ms)    Bindings     Core        
                                  (PTX)           (CUBIN)      Speedup      Speedup     
----------------------------------------------------------------------------------------------------
400×300                    1.56           0.08         0.09        19.12x        16.45x
800×600                    6.13           0.35         0.30        17.31x        20.30x
1280×720 (HD)             14.12           0.50         0.49        28.29x        28.60x
1920×1080 (Full HD)       30.68           0.78         0.90        39.34x        34.21x
2560×1440 (2K)            62.76           1.36         1.57        46.20x        39.93x
====================================================================================================

AVERAGE PERFORMANCE ACROSS ALL SIZES:
  CPU:            23.05 ms
  cuda.bindings:  0.61 ms  (37.50x speedup)
  cuda.core:      0.67 ms  (34.31x speedup)
```

![timings](/assets/images/cuda_python_timings.png)

So we can see that cuda bindings and cuda core are much faster. The question is when do we choose to use PTX or CUBIN. The answer to this question is it depends on the amount of cold starts you think you will have in the system. If you think the first initial run being slower will have large issues in your system then CUBIN is the better option.

## Timing Cold Start

The difference is pretty large on cold starts.

```
================================================================================
RESULTS
================================================================================
Metric                              PTX (JIT)       CUBIN (Pre-compiled)
--------------------------------------------------------------------------------
Cold-start time                     97.515          14.766          ms
Warm per-launch                     4.386           4.395           µs
```

![cold_start](/assets/images/cold_start_cuda_times.png)

When using CUBIN you dont have to use a JIT which takes a while on a cold start. Cold starts happen for a variety of reasons, new container/process, serverless function got invoked, services were rebooted, etc...

If this is an issue that you need to mitigate then this would be when you want to prioritize CUBIN if this type of problem is very common for you.

After everything is "warmed up" then both are very fast. Warming up is a term for allocating and caching all of the modules, memory, and hardware states. These are the major areas that are "warmed up" and make it so any other runs are extremely quick

1) JIT compilation cache (PTX only)

2) GPU module cache (both)

3) GPU memory allocator pools (both)

4) GPU context initialization (both)

Once all these operations are done or cached subsequent operations are very fast and you will hit super sonic speeds.

# Conclusion

Now we have covered multiple tools for you to use when you need more speed and power to your algorithms. GPUs provide a level of parallel operations that transcend the CPU and allow for algorithms to run at speeds that make your head spin. Running algorithms on the GPU is not an automatic win. GPUs work best when the algorithm or stages of the algorithm can be done in parallel.

You are not handcuffed by the slowness of python anymore. You have been unchained and know more tools that you can use when necessary. CuPy, Numba, and cuda python are all very valuable to those that care about performance. Granted you won't ever break free of that ball and chain called python. ML/AI industry is cursed and bound by blood to a snake.