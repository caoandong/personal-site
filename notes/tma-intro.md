# The Hitchhiker’s Guide to the Tensor Memory Accelerator (TMA)

_This page serves as a guide to the TMA interfaces provided in [Lab\_\
_9](https://accelerated-computing.academy/fall25/labs/lab9). Each interface described below is a simple wrapper around a_
_single inline PTX instruction, exposing you to the lowest-level interfaces_
_NVIDIA provides. Our goal is to introduce the essential concepts you’ll_
_need for Lab 9, rather than to provide exhaustive documentation. For complete_
_details on these operations, we encourage you to explore the linked NVIDIA_
_documentation throughout this guide._

_The interfaces are provided in_
_[`tma-interface.cuh`](https://github.com/accelerated-computing-class/lab9/blob/main/tma-interface.cuh)._

## Async Proxy

Before we get started with interfaces we provide interact with TMA, you
will need to understand the concept of a **proxy**.

### What is a Proxy?

NVIDIA defines a
**[proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#proxies)** as
“an abstract label applied to a method of memory access.” In simpler terms, a
proxy is a **tag** that the hardware associates with different types of memory
operations to track which operations might interact with each other.

Think of proxies as different “channels” through which memory operations can
flow. Operations in different proxies **may not be immediately visible** to each
other. To ensure that operations in one proxy are visible to operations in
another proxy, we need to insert an **explicit fence** between them.

### The Two Proxies You Need to Know

Until now, most memory operations we’ve performed course have happened in the
**generic proxy**. These include:

- Regular loads and stores from global memory
- Reads and writes to shared memory

TMA operations, on the other hand, are performed in a different proxy called the
**[asynchronous\**\
**proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy)**
(or **async proxy** for short). The async proxy is specifically designed to
handle specialized asynchronous data movement operations like TMA.

### Visibility Rules Between Proxies

When working with both proxies, it’s critical to understand how operations in
one proxy become visible to operations in the other. The visibility rules are
**[asymmetric](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy)**:

1. **Generic Proxy to Async Proxy:** Operations that occur in the generic proxy
   are **not automatically visible** to the async proxy. To make them visible,
   you must insert an **explicit fence**.

   **Example:** If you initialize an `mbarrier` in shared memory using a regular
   store (which happens in the generic proxy), you must insert a fence before
   launching TMA operations that will interact with that barrier. Otherwise, the
   TMA unit may not see the initialized barrier state.

   Or, if you write to data in shared memory using regular stores in the generic
   proxy, you must insert a fence to ensure the TMA engine sees the updated
   values.

2. **Async Proxy to Generic Proxy:** Operations that occur in the async proxy
   **are automatically visible** to the generic proxy after their completion, no
   fence is required.

   **Example:** Once a TMA load _completes_ and writes data to shared memory, that
   data is immediately visible to regular loads from shared memory. You don’t
   need a fence to see the TMA’s results (but you will need to wait until the
   load completes!)


In `tma-interface.cuh`, you can use the `async_proxy_fence()` function which
inserts a fence that ensures all prior operations in the generic proxy are
visible to subsequent operations in the async proxy.

```
__device__ static __forceinline__ void async_proxy_fence()
```

## `mbarriers`

As described in [Lab 9](https://accelerated-computing.academy/fall25/labs/lab9), an [`mbarrier object`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)
is a synchronization primitive that implements a **split barrier**.

### How `mbarriers` track completion

An `mbarrier` is initialized with an **arrival count**, which specifies the
number of arrivals that must be registered before the barrier completes. But,
arrival count isn’t the only thing an `mbarrier` tracks. It also maintains a
**transaction byte count** that tracks how many bytes have been transferred by
asynchronous TMA operations. **The barrier**
**[completes](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-contents)**
**when both the expected arrivals and expected transaction bytes have been**
**satisfied.**

Beyond tracking these counts, `mbarriers` also maintain a **[phase\**\
**bit](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-phase)**
that alternates between 0 and 1 each time the barrier completes. This allows the
same barrier to be reused multiple times without reinitialization. Lanes can
distinguish between different “generations” of barrier completion by tracking
which phase they’re waiting for.

We’ll discuss the general interface for `mbarriers` first (arrival count), and
then cover the [TMA-specific transaction\\
tracking](https://accelerated-computing.academy/fall25/resources/tma-interface/(https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier))
details (transaction byte count).

Please note that all of the `mbarrier` interface functions described below are **per-lane**
**operations**. This means that when you call one of these functions, it executes
independently for each lane that calls it.

### Initializing an mbarrier

```
void init_barrier(uint64_t *bar, int arrival_count);
```

The `init_barrier` function initializes an `mbarrier` allocated in **shared**
**memory**. The `arrival_count` parameter specifies how many arrivals must be
registered before the barrier completes.

When first initialized, the `mbarrier` starts in **phase 0** with an arrival
count of **0**. Each time the expected number of arrivals is reached, the
barrier completes, the phase flips (0 →\\to→ 1 or 1 →\\to→ 0), and the arrival
count resets to
0.

**Usage notes:**

- **Single-lane initialization:** Only **one lane** should call `init_barrier`.
  All other lanes that intend to use the barrier must wait until initialization
  is complete. Use `__syncthreads()` (or an appropriate synchronization
  instruction) to ensure all lanes wait for the lane performing the
  `init_barrier` instruction before using the barrier.

- **Alignment requirement:** The `mbarrier` must be [aligned to **8**\\
  **bytes**](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).
  Use `alignas(8)` when declaring the barrier in shared memory.

- **TMA visibility:** When using the barrier with TMA operations, insert an
  `async_proxy_fence()` after initialization to ensure the barrier is visible to
  the TMA unit. Without this fence, the TMA unit may observe uninitialized
  values.


### Arriving at the Barrier

```
void arrive(uint64_t *bar, int count);
```

To
[arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
at an `mbarrier`, you will need to use the `arrive` function. The `arrive`
function signals that a lane has reached a synchronization point, incrementing
the barrier’s arrival count by `count`. This is a **non-blocking** operation:
the calling lane does not wait for the barrier to complete. To wait for
completion, you must separately call `wait` or `test_wait`.

Each call to `arrive` is counted as a **distinct arrival**, even if the same
lane calls it multiple times. For example, if you initialize a barrier with
`arrival_count = 32`, you could have 32 different lanes each call
`arrive(&barrier, 1)`, or a single thread could call `arrive(&barrier, 32)`, or
any combination that sums to 32.

### Waiting for Barrier Completion

To [wait for a barrier\\
completion](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait),
we provide two wait functions (`test_wait` and `try_wait`) and a convenience
wrapper `wait` built on top of `try_wait`.

```
int test_wait(uint64_t *bar, int phaseParity);
```

The `test_wait` function is a **non-blocking** test that checks whether the
specified phase of the barrier has completed. It returns **1** if the phase is
complete and **0** if not, then immediately returns control to the caller. You
should use `test_wait` when you want to poll the barrier status without yielding
control of the lane.

```
int try_wait(uint64_t *bar, int phaseParity);
```

The `try_wait` function tests whether the specified phase of the barrier has
completed, returning **1** if complete and **0** if not. Unlike `test_wait`,
this is a **potentially blocking** instruction.

The key difference between `test_wait` and `try_wait` is that if the phase is
not yet complete, the hardware **may suspend** the calling lane to allow other
warps to make progress. The suspended thread will resume when either:

- The specified phase completes, **or**
- A hardware-determined timeout elapses

**Note:** The underlying PTX instruction supports an optional `suspendTimeHint`
parameter that can provide hints about how long to suspend for but we do not
expose this parameter in our interface.

Because of this timeout behavior, a single call to `try_wait` does not mean that
the phase has completed, it may just mean the timeout was reached. If you need
to guarantee waiting until completion, wrap `try_wait` in a loop:

```
while (!try_wait(&barrier, phase)) { /* spin */ }
```

This pattern also matches the [PTX example](https://accelerated-computing.academy/fall25/resources/tma-interface/(https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)) from NVIDIA’s documentation:

```ptx
waitLoop:
    mbarrier.try_wait.shared::cta.b64  complete, [barrier], phase;
    @!complete bra waitLoop;
```

**Note:** You must track which phase the barrier is in yourself. Recall that
barriers start in phase 0 and flip between 0 and 1 on each completion.

```
void wait(uint64_t *bar, int phaseParity);
```

The `wait` function **blocks** until the specified phase of the barrier
completes. This is a convenience function implemented as a loop around
`try_wait`, so you don’t have to write the loop yourself. For most cases where
you need to block until the barrier completes, use `wait` instead of `try_wait`
or `test_wait`.

### Additional Reading

The `mbarrier` interface includes additional functionality that we haven’t
covered here, such as invalidating barriers for reinitialization, reading
pending arrival counts, and more. We’ve focused on the subset of operations
you’ll need for Lab 9, but if you’re curious about the full capabilities of
`mbarriers`, you can explore the complete documentation in the [PTX ISA\\
Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier).

## TMA Loads

So far, we’ve discussed how `mbarriers` track arrival counts. But recall
from our earlier discussion that `mbarriers` track two types of progress:
arrivals and transaction bytes.

The **transaction byte count** (often called **“tx-count”**) tracks how many bytes
have been [transferred by asynchronous TMA\\
operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations).
This allows a single barrier to coordinate both lane arrivals and data movement:
you can wait on a barrier knowing that when it completes, not only have all
expected threads arrived, but all expected data has also been transferred.

### Registering Expected Bytes

```
void expect_bytes(uint64_t *bar, int num_bytes);
```

The `expect_bytes` function increments the expected transaction byte count for
the **current phase** of the barrier by `num_bytes` using the
[exptect-tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)
operation. When you wait on the barrier, your thread will block until **both**
the expected number of arrivals
**and** the expected number of transaction bytes have been satisfied.

There are three important things to keep in mind:

1. **Partial transfers:** If a TMA operation transfers more bytes than you
   specified with `expect_bytes`, the barrier will complete as soon as
   `num_bytes` have been transferred, even if the full TMA operation hasn’t
   finished.

2. **Race condition:** If you call `expect_bytes` **after** lanes have already
   arrived at the barrier and the barrier has completed, you have a race
   condition. The `expect_bytes` call may register the bytes with the current
   phase (which has already completed) or with the next phase, depending on
   timing. To avoid this, always call `expect_bytes` **before** at least one
   lane (often the lane issuing the TMA operation) arrives at the barrier.

3. **Transaction counts reset with each phase:** The expected transaction byte
   count must be set for each phase by calling `expect_bytes` before each
   transfer. Unlike the arrival count, which is set once during initialization
   and persists across all phases, the transaction byte count resets to zero
   every time the barrier completes.


```
void expect_bytes_and_arrive(uint64_t * bar, int bytes);
```

The `expect_bytes_and_arrive` function [combines two\\
operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive):

1. It increments the expected transaction byte count for the current phase by
   `bytes` (just like `expect_bytes`)
2. It **increments the arrival count** of the barrier.

Using `expect_bytes_and_arrive` avoids the race condition by first registering
the expected bytes and then arriving at the barrier in a single operation.

You can choose either `expect_bytes` or `expect_bytes_and_arrive` based on your
needs, but be consistent in your approach. If you use `expect_bytes_and_arrive`,
remember to account for its arrival when initializing your barrier’s
`arrival_count`.

## Moving Data from Global to Shared Memory

With the synchronization mechanism out of way, we can finally look at the
interfaces for moving tiles of data using the TMA!

TMA uses [specialized\\
instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
for moving multi-dimensional tiles of data from global memory directly to shared
memory. All TMA operations require a **tensor map** (of type `CUtensorMap`),
which is a descriptor that encodes the shape, layout, and tiling parameters of
your data. You create this tensor map on the host using
[`cuTensorMapEncodeTiled`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7).
The tensor map tells the TMA hardware the dimensionality (1D through 5D), tile
size, element size, and memory layout of the data. This is how TMA knows exactly
how many bytes to transfer and from where.

Each global-to-shared transfer is a **per-lane** operation. The typical workflow is:

1. Call `expect_bytes` (or `expect_bytes_and_arrive`) on the barrier to register
   how many bytes you expect the TMA operation to transfer.
2. Issue the global-to-shared transfer, passing the barrier pointer.
3. Wait on the barrier to know when the data has arrived in shared memory.

When the transfer executes, it automatically increments the barrier’s
transaction byte count as data is transferred, eventually satisfying the bytes
you registered with `expect_bytes`. You will want to remember two important
details:

- **Alignment:** The destination in shared memory must be **128-byte aligned**.
  This is necessary because TMA can swizzle data in particular modes, as you may
  have noticed in the [`cuTensorMapEncodeTiled`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7) interface.
- **Coordinate interpretation:** The coordinates (`c0`, `c1`, etc.) represent
  the **starting index** in the coordinate space of your tensor, not tile
  indices. For example, consider a 1D tensor where each tile contains 16
  elements. To load the second tile, you pass coordinate `16` (the starting
  element of that tile), not `1`.

Now, we provide the interfaces for using the TMA to load tiles from 1D to 5D:

```
__device__ static __forceinline__
void cp_async_bulk_tensor_1d_global_to_shared(
          void* smem_dest,
          const CUtensorMap* tensor_map,
          int c0,
          uint64_t* bar)
```

Loads a 1D tile at coordinate `c0` from global memory to shared memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_2d_global_to_shared(
          void* smem_dest,
          const CUtensorMap* tensor_map,
          int c0, int c1,
          uint64_t* bar)
```

Loads a 2D tile at coordinates `(c0, c1)` from global memory to shared memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_3d_global_to_shared(
          void* smem_dest,
          const CUtensorMap* tensor_map,
          int c0, int c1, int c2,
          uint64_t* bar)
```

Loads a 3D tile at coordinates `(c0, c1, c2)` from global memory to shared memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_4d_global_to_shared(
          void* smem_dest,
          const CUtensorMap* tensor_map,
          int c0, int c1, int c2, int c3,
          uint64_t* bar)
```

Loads a 4D tile at coordinates `(c0, c1, c2, c3)` from global memory to shared memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_5d_global_to_shared(
          void* smem_dest,
          const CUtensorMap* tensor_map,
          int c0, int c1, int c2, int c3, int c4,
          uint64_t* bar)
```

Loads a 5D tile at coordinates `(c0, c1, c2, c3, c4)` from global memory to shared memory.

## Moving Data from Shared to Global Memory

The TMA also provides instructions to move multi-dimensional tiles of data **from shared**
**memory to global memory**. These operations are symmetric to the
global-to-shared operations we just discussed. They, too, require a tensor map
and use coordinates to specify where in global memory the tile should be
written.

Unlike global-to-shared transfers (which use `mbarriers` for completion
tracking), shared-to-global transfers use a **[commit\**\
**group](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-bulk-tensor-copy-completion)**
mechanism for tracking completion. This is the same mechanism we saw with async
copies in [Lab 5](https://accelerated-computing.academy/fall25/labs/lab5). After issuing shared-to-global TMA instructions,
you must:

1. Commit them to a commit group using `tma_store_commit_group()`
2. Wait until at most `N` commit groups are pending using
   `tma_store_wait_until_pending<N>()`

### Shared-to-Global Transfer Functions

Here are the interfaces we provide for using the TMA to store 1D to 5D tiles.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_1d_shared_to_global(
          const CUtensorMap* tensor_map,
          int c0,
          const void* smem_src)
```

Stores a 1D tile at coordinate `c0` from shared memory to global memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_2d_shared_to_global(
          const CUtensorMap* tensor_map,
          int c0, int c1,
          const void* smem_src)
```

Stores a 2D tile at coordinates `(c0, c1)` from shared memory to global memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_3d_shared_to_global(
          const CUtensorMap* tensor_map,
          int c0, int c1, int c2,
          const void* smem_src)
```

Stores a 3D tile at coordinates `(c0, c1, c2)` from shared memory to global memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_4d_shared_to_global(
          const CUtensorMap* tensor_map,
          int c0, int c1, int c2, int c3,
          const void* smem_src)
```

Stores a 4D tile at coordinates `(c0, c1, c2, c3)` from shared memory to global memory.

```
__device__ static __forceinline__
void cp_async_bulk_tensor_5d_shared_to_global(
          const CUtensorMap* tensor_map,
          int c0, int c1, int c2, int c3, int c4,
          const void* smem_src)
```

Stores a 5D tile at coordinates `(c0, c1, c2, c3, c4)` from shared memory to global memory.

### Managing Transfer Completion

To manage asynchrony, we also provide a wrapper over the `commit_group` interface:

```
__device__ static __forceinline__ void tma_commit_group()
```

Commits all previously issued shared-to-global transfers to a commit group. This
marks the boundary between groups of transfers, allowing you to track their
completion independently.

```
template <int N>
void tma_wait_until_pending();
```

Blocks until at most `N` commit groups are still pending. This function ensures
that older transfer groups have completed before proceeding. For example,
`tma_store_wait_until_pending<0>()` waits until all transfer groups have
completed, while `tma_store_wait_until_pending<1>()` waits until at most one
group remains pending.

## Acknowledgements

The implementations provided in this guide draw from several sources:

- **[CUDA’s\**\
  **`cuda/barrier.h`](https://github.com/NVIDIA/cccl/blob/6f6f38d8cc2ace70c1295052036328be2bacad24/libcudacxx/include/cuda/__ptx/instructions/generated/cp_async_bulk_tensor.h):**
  Many of the instructions and patterns for working with `mbarriers` are adapted
  from NVIDIA’s CUDA Standard Library.

- **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens):** The
  `test_wait`, `try_wait` and `wait` instruction implementation has been
  directly adapted from this library.