---
layout: post
title: Addressing of GPGPU
description: Addressing scheme of GPU.
tags: GPU Scaling-Up Scaling-Out
giscus_comments: true
date: 2025-3-24
featured: false
bibliography: 2025-3-26-GPU-Addressing.bib
---
# Unified Memory Addressing (UMA)
In contemporary computational architecture, the significance of the GPU software ecosystem far outweighs the emphasis placed on hardware performance. Despite the predominant focus of research efforts on enhancing GPU performance metrics, the aspect of programmability remains critically overlooked. Programmability is fundamental to the establishment and evolution of a robust software ecosystem. Specifically, addressing mechanisms in GPUs serve as a pivotal interface for programming practices and memory management strategies. The progression towards greater programming ease has seen the transition from physical addressing to virtual addressing, culminating in the current paradigm of unified memory addressing. This study endeavors to explore both the hardware and software facets inherent to unified addressing, aiming to furnish a deeper understanding of its implications and applications within the realm of GPU architectures.

Following code snippet shows how unified memory address simplifies the program:

{% highlight c++ linenos %}
#include <iostream>
#include <math.h>
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(void) {
  int N = 1<<20;
  float *x, *y;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
 
  // Launch kernel on 1M elements on the GPU, NO need to copy data to GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y); //Using the same X/Y pointer
 
  // Wait for GPU to finish before accessing on host, NO need to copy data from GPU
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}
{% endhighlight %}

Besides simplifying programming, **UMA** also provides following benefits:
1. **Enable Large Data Models**:  supports oversubscribe GPU memory Allocate up to system memory size.
2. **Simpler Data Access**: CPU/GPU Data coherence Unified memory atomic operations.
3. **Performance Turning with prefetching**: Usage hints via _cudaMemAdvise_ API Explicit prefetching API.

The underlying hardware architecture are illustrated in following diagram {% cite allen2021depth --file 2025-3-26-GPU-Addressing %}.

![UVM-Arch](/assets/img/UMA-Arch.png){:width="50%"}

1. Page Fault propagates to the GPU memory management unit (GMMU), which sends a hardware interrupt to the host. The GMMU writes the corresponding fault information into the GPU Fault Buffer (circular buffer, configured and managed by the UVM driver).

2. GPU sends an interrupt over the interconnect to alert the host UVM driver of a page fault, the host retrieves the complete fault information from the GPU Fault Buffer.

3. Host instructs the GPU to copy pages into its memory via hardware copy engine, and update the page tables.

4. Host instructs the GPU to 'replay' the fault, causing uTLB to fetch the page table in GPU DRAM.


# Addressing for Scaling
In contemporary computational environments, the capabilities for Scaling Up and Scaling Out constitute critical features for modern Graphics Processing Units (GPUs). Although NVIDIA has pioneered solutions such as NVLink and NVSwitch to address scalability, there remains a notable absence of an open industry standard for connecting GPUs through ultra-efficient interconnects. Addressing this gap, two emerging initiatives—the [UALink consortium](https://www.ualinkconsortium.org/) and the [UltraEthernet Consortium (UEC)](https://ultraethernet.org/)—are actively working towards overcoming these scalability challenges. These organizations have meticulously defined the hardware specifications and communication protocols necessary for such advancements. However, their frameworks offer limited discourse on addressing schemes and software programming paradigms. This post aims to propose a potential addressing strategy tailored to mainstream parallel and distributed programming models.

### Parallel Programming Models

There are two parallel programming models in large distributed system: shared memory (SHMEM) and Message Passing Interface (MPI), as illustrated in follow diagram. 

![Scaling Programming Model](/assets/img/Scaling-Programming-Model.png){:width="100%"}

**SHMEM** 
* Shared data are allocated in **Symmetric Heap**, but each PE manages its memory and the allocated buffer are **different virtual address**. Only size and alignment are coherent between PEs.
* Memory are accessed using **One Sided** api, which means remote nodes is not aware when and who is access the shared memory.

**MPI** 
* No shared memory, only local buffer is used for **temporary** storage.
* Memory are accessed using **Two Sided** api, which means remote nodes needs to **acknowledge** the transaction.

**Fabric API**: 

Libraries that aim to provide low-level, high-performance communication interfaces for applications in high-performance computing (HPC), cloud, data analytics, and other fields requiring efficient network communication. 

* **UCX**: [https://openucx.org/](https://openucx.org/), unified API that handles many of the complexities of multi-transport environments.
* **Libfabic**: [https://ofiwg.github.io/libfabric/](https://ofiwg.github.io/libfabric/), Fine-grained control over their network operations.


### Scaling Systems

This diagrams shows a general scaling system: 

![Scaling System](/assets/img/Scaling-System.png){:width="100%"}

**Scaling Up**
* **Host**: 8X GPU/Host
* **GPU Domain**: direct GPU to GPU communication domain, via NVLink/NVSwitch/UALink/UEC etc.

**Scaling Out**
* Host to Host communication, via high speed ethernet fabric or InfiniBand.

---
### Reference
{% bibliography --file 2025-3-26-GPU-Addressing %}


