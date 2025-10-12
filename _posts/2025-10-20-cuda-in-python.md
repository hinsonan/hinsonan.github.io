---
layout: post
title: "GPU Programming in Python: We gotta go fast"
date: 2025-10-20
categories: ML
---

We all know Python is slow but everyone knows the secret to python programming is to try and avoid python at all cost and call a C/Rust binding beneath the hood.

Since we are all ML nerds let's take this a step further and use python to call gpu accelerated libraries or even write CUDA code. Many people know about using numpy or fast cpu operations and we know Pytorch allows you to easily put models on the gpu but it goes deeper than that.

If you are doing intense computations you don't need to feel that you can only write optimized cpu algorithms in python. You can utilize the speed of the gpu when you have algorithms suited for parallel processing.

# GPU vs CPU Recap

{% mermaid %}
```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#1a1a1a','primaryTextColor':'#ffffff','primaryBorderColor':'#ffffff','lineColor':'#ffffff','secondaryColor':'#000000','tertiaryColor':'#333333','fontSize':'16px','fontFamily':'Arial'}}}%%

graph TB
    START["Array of 10,000 elements - Task: Add 10 to each"]
    
    subgraph CPU[" "]
        direction TB
        CPU_TITLE["CPU: 4 Cores with AVX-256 SIMD"]
        CPU_NOTE["Each core processes 8 elements at once - Total: 32 elements per cycle"]
        
        C1["Core 1<br/>Elements 1-8"]
        C2["Core 2<br/>Elements 9-16"]
        C3["Core 3<br/>Elements 17-24"]
        C4["Core 4<br/>Elements 25-32"]
        
        BATCH["313 batches needed<br/>10,000 ÷ 32"]
        
        CPU_TITLE --> CPU_NOTE
        CPU_NOTE --> C1 & C2 & C3 & C4
        C1 & C2 & C3 & C4 --> BATCH
    end
    
    subgraph GPU[" "]
        direction TB
        GPU_TITLE["GPU: SIMT Architecture (Single Instruction, Multiple Threads)"]
        GPU_NOTE["Single instruction broadcast - Thousands of threads execute simultaneously"]
        
        W1["Warp 1<br/>32 threads"]
        W2["Warp 2<br/>32 threads"]
        W3["Warp 3<br/>32 threads"]
        WDOTS["..."]
        WN["Warp 313<br/>32 threads"]
        
        SIMUL["All warps execute in parallel<br/>Effective: 1-10 batches"]
        
        GPU_TITLE --> GPU_NOTE
        GPU_NOTE --> W1 & W2 & W3 & WDOTS & WN
        W1 & W2 & W3 & WDOTS & WN --> SIMUL
    end
    
    START --> CPU
    START --> GPU
    
    BATCH --> CPU_RES["Sequential batches<br/>Lower throughput<br/>Better for complex logic"]
    SIMUL --> GPU_RES["Parallel batches<br/>Higher throughput<br/>Best for same operation"]
    
    style START fill:#ffcc00,stroke:#000000,stroke-width:4px,color:#000000
    
    style CPU fill:#000000,stroke:#00ff00,stroke-width:4px,color:#ffffff
    style GPU fill:#000000,stroke:#ff00ff,stroke-width:4px,color:#ffffff
    
    style CPU_TITLE fill:#003300,stroke:#00ff00,stroke-width:3px,color:#ffffff
    style GPU_TITLE fill:#330033,stroke:#ff00ff,stroke-width:3px,color:#ffffff
    
    style CPU_NOTE fill:#004400,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style GPU_NOTE fill:#440044,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    
    style C1 fill:#006600,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style C2 fill:#006600,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style C3 fill:#006600,stroke:#00ff00,stroke-width:2px,color:#ffffff
    style C4 fill:#006600,stroke:#00ff00,stroke-width:2px,color:#ffffff
    
    style BATCH fill:#008800,stroke:#00ff00,stroke-width:2px,color:#ffffff
    
    style W1 fill:#660066,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style W2 fill:#660066,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style W3 fill:#660066,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    style WN fill:#660066,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    
    style SIMUL fill:#880088,stroke:#ff00ff,stroke-width:2px,color:#ffffff
    
    style CPU_RES fill:#00aa00,stroke:#00ff00,stroke-width:4px,color:#ffffff
    style GPU_RES fill:#aa00aa,stroke:#ff00ff,stroke-width:4px,color:#ffffff
```
{% endmermaid %}

CPUs have very powerful cores and are responsible for managing a lot more than a GPU core. CPUs can handle many different types of operations, branchings, etc...

GPUs are have a lot of "dumb" cores that are really meant to process calculations and do operations in parallel. While CPUs have SIMD (Single Instruction Multiple Data) and can perform parallel operations they do not scale at the level GPUs do.

GPUs use SIMT (Single Instruction Multiple Threads) which as you can guess means they treat operations and parallel processing as a high priority.

You can go really deep into this topic but we want to look at how we can use this in our day to day lives as we are stuck in python land. The gist of this is that GPUs can perform calculations much quicker than a CPU can due to having more cores. There are some caveats such as the process needs to be non-blocking or no branching in order to take full advantage of the GPU.

CPUs can be faster than the GPU but if you have a operation that can be done in parallel then the GPU is much faster. This is why ML models run on GPUs since they are running thousands of matrix operations on repeat.

# CuPy

[CuPy](https://cupy.dev/) 