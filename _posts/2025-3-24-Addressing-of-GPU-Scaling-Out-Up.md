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

This diagrams shows a general scaling system：

![Scaling System](/assets/img/Scaling-System.png){:width="100%"}

**Scaling Up**
Scaling up means integrating more GPUs in a GPU domain, which shares a single unified memory address space. A typical scaling up system consists of multiple Hosts and GPUs.

* **GPU Domain**: direct GPU to GPU communication domain, via NVLink/NVSwitch/UALink/UEC etc.
* **Host**: 8X GPU/Host

**Scaling Out**
Scaling out means more GPU domains connected by high speed network, but with different memory address space.

* Host to Host communication, via high speed ethernet fabric or InfiniBand.

**Addressing in Scaling Up**

Recent scaling up system incorporates **unified memory addressing** (e.g., UALink), to facilitate remote memory access, 
especially small data type accesses (such as word or double word). But this not necessary in distributed programme model.

In a shared memory programming model, illustrated in following diagram, we suppose the GPU/GPU are connected with build-in Ethernet Controllers and Ethernet switches.

![Scaling Up Address](/assets/img/Scaling-Up-Address.png){:width="100%"}

The scaling up system uses two types of addressing:
* **System Physical Address**: which is mapped to local GPU physical memories, like HBM.
* **Network Physical Address**: which is mapping to remote GPU. The **NPA** contains the GPU ID that will be used to find the correct MAC address of the destination GPU.

**Addressing importing and exporting**

In scaling up and scaling out system, the GPU under the same OS has is own **private** virtual address space, and not remote access is not allowed. The setup remote memory access, the remote GPU must **export** part of its memory, and other GPU must **import** this memory to its own address space.

These __export__ and __import__ involves multiple soft modules. A **memory handle** is used to pass the information between these modules. For example in HIP programming, __hipIpcMemHandle_t__ is defined for these purpose:

{% highlight c++ linenos %}

#define hipIpcMemLazyEnablePeerAccess 0x01

 #define HIP_IPC_HANDLE_SIZE 64

 // The structure of remote memory handle.
 typedef struct hipIpcMemHandle_st {
     char reserved[HIP_IPC_HANDLE_SIZE];
 } hipIpcMemHandle_t;

 //Internal structure of IPC memory handle.
#define IHIP_IPC_MEM_HANDLE_SIZE   32

 typedef struct ihipIpcMemHandle_st {
  char ipc_handle[IHIP_IPC_MEM_HANDLE_SIZE];  ///< ipc memory handle on ROCr
  size_t psize;
  size_t poffset;
  int owners_process_id;
  char reserved[IHIP_IPC_MEM_RESERVED_SIZE];
} ihipIpcMemHandle_t;

{% endhighlight %}

* **export**: 

When a remote GPU whens to export a memory region, it calls __hipIpcGetMemHandle()__ to get an memory handle, and passed it to other GPUs that wants to import the memory.

{% highlight c++ linenos %}
 // export the local device memory, which then be passed to other GPUs for remote access.
hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
{% endhighlight %}

* **import**: 

The GPU which wants to import the memory calls __hipIpcOpenMemHandle__ to import the remote gpu address space:

{% highlight c++ linenos %}
 // Maps memory exported from another process with hipIpcGetMemHandle into the current device address space.
 // hipIpcMemHandles from each device in a given process may only be opened by one context per device per other process.
hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
{% endhighlight %}

* **close**: 

After remote GPU used the remote memory, it calls __hipIpcCloseMemHandle__ to delete the remote gpu address space from its own address space:

{% highlight c++ linenos %}
//close the imported remote memory handle.
hipError_t hipIpcCloseMemHandle(void* devPtr);
{% endhighlight %}

![Scaling Import and Export ](/assets/img/Scaling-Import-Export.png){:width="70%"}

During these import/export and the following remote memory process,  these MMUs are involved:
Three MMUs are used during GPU to GPU communication:
* **GPU MMU**: setup the TLB table during import, and translate translate virtual address to NPA. 
* **Port MMU**: A table in Ethernet controller, or just a software implementation which maps GPU-ID to network MAC address.
* **R-MMU**:Target GPU MMU translate NPA to local SPA, also do accessible control and checks.


**Addressing mapping in Shared Programming Model**

In shared memory programming model, the MMU setup and translation process can be described as Following:

![Scaling-Up-Shared-Mem](/assets/img/Scaling-Shared-Mem.png){:width="100%"}

1. **shmem_init()**:  The Shared Memory library will build up a **segment table** first, which will record the **shared** memory segments, including the start address and size. This table is **shared** between all PEs and will be referenced when accessed remotely.

2. **shmem_malloc()**: User programming applies memories in the Symmetric shared heap. 
  * All PE will do the same **malloc()** action and the **shmem_malloc()** will only return after all PE completes its operation.
  * shmem_malloc() returns **Local Virtual Address** and different PE returns **different VA**, this VA only valid in this PE.
  * The library will **register** the allocated memory regions in *R-MMU**, so remote access are allowed for this memory region, 
     and **PIN** the corresponding physical pages so OS won't swap out the pages.
  
3. **shmem_access()**: User program access remote memory via **Local Virtual Address** returned by malloc() and destination PE ID:
  *  **Symmetric Offset**  are calculated using __segment table__ and __local VA__.
  *  **Remote VA** are generated using __segment table__ and __Symmetric Offset__.
  *  **Network Packet** are composed using __remote VA__, __data__ and __command__.
  *  **Remote PE**: Remote PE parsed the network packet, extract the __remote VA__ as its __local VA__, and do the accesses. __local VA__ are then translated to physical address with **R-MMU**.

---

### Reference
{% bibliography --file 2025-3-26-GPU-Addressing %}


