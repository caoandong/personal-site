# Nand to Transformers

## A constructionist campaign for mastering GPU and ML systems from first principles

The north star is simple:

> Build a small but complete machine from logic gates to a programmable GPU, build its compiler and runtime, implement the kernels of modern machine learning, and finally run a tiny transformer on the machine you constructed.

This is not a sequence of disconnected tutorials. It is one cumulative artifact. Every checkpoint adds a component that later checkpoints genuinely use.

The curriculum should feel like _Nand2Tetris_ crossed with a GPU laboratory, a compiler course, and a miniature ML-systems apprenticeship. The learner should be able to zoom continuously through this causal chain:

```text
transformer output
  -> runtime schedule
  -> kernel graph
  -> tensor program
  -> compiler IR
  -> toy assembly
  -> warp instructions
  -> pipeline events
  -> register and memory movement
  -> logic and state elements
```

The ambition is not merely to finish the game. The main quest builds an intuition pump; the expert raids then connect that intuition to real PTX, SASS, CUDA, compiler internals, modern Tensor Cores, production megakernels, and distributed training.

Companion notes:

- [`cuda-study-plan.md`](./cuda-study-plan.md) contains the advanced CUDA, attention, serving, megakernel and low-precision specialization campaign.
- [`ultralearning.md`](./ultralearning.md) contains the learning principles used for retrieval, direct practice, drills, feedback and retention.

---

# 1. Why this structure

The strongest constructionist curricula share several properties:

1. **One machine grows throughout the course.** Nand2Tetris moves from Boolean logic through arithmetic, memory, architecture, machine language, assembler, VM, compiler, and operating system. Later projects consume earlier ones rather than discarding them.
2. **The machine is radically simplified but causally honest.** It omits industrial complexity while preserving the mechanisms that explain behavior.
3. **Every abstraction can be opened.** Ripes makes pipelines, registers, caches, instructions, and memory visible rather than presenting the processor as a black box.
4. **Exercises begin with direct manipulation.** GPU Puzzles teaches mapping, shared memory, and reductions by making learners write kernels and immediately see accesses and results.
5. **There is a path from teaching model to real system.** `tiny-gpu` provides a minimal documented Verilog GPU; Vortex provides an open full hardware/software GPU stack; Accel-Sim provides a deeper performance-simulation target.
6. **Compiler construction is staged.** LLVM's Kaleidoscope and MLIR's Toy tutorial move from syntax and ASTs through IR, transformations, lowering, and code generation.
7. **The final ML system stays readable.** MiniTorch, tinygrad, and `llm.c` demonstrate the value of small end-to-end systems whose tensors, autograd, compiler, kernels, and training loop remain inspectable.

The curriculum should borrow those structures, not simply assemble their reading lists.

---

# 2. The cumulative artifact: TGPU

The learner builds **TGPU**, a deliberately small teaching GPU and ML computer.

TGPU has two execution modes:

## Functional mode

Functional mode executes instructions and kernels quickly enough to run a complete tiny transformer. It preserves architectural semantics but does not pretend to be cycle accurate.

Use it for:

- whole programs;
- kernel correctness;
- compiler validation;
- transformer inference;
- tiny training runs;
- multi-GPU functional experiments.

## Timing mode

Timing mode executes selected kernels cycle by cycle with explicit pipelines, queues, banks, caches, scoreboards, and links.

Use it for:

- architectural experiments;
- performance prediction;
- memory transactions;
- divergence;
- hazards and barriers;
- scheduling and latency hiding;
- communication overlap.

This separation is essential. A browser simulator detailed enough to teach pipelines would be painfully slow for an entire transformer. A simulator fast enough for a transformer would otherwise conceal the mechanisms the course is meant to teach.

## The initial teaching machine

The exact constants can change after prototyping, but the initial target should remain small enough to visualize:

```text
2 compute cores
4 resident warps per core
8 lanes per warp
16 vector registers per lane
8 scalar registers per warp
4-bank shared memory per core
byte-addressed global memory
small L1 per core and shared L2
integer, FP32, BF16/FP16, and matrix instructions added progressively
```

The ISA grows with the course:

```text
integer ALU and control flow
-> load/store
-> vector/SIMT execution
-> predicates and masks
-> barriers and atomics
-> floating-point FMA and conversions
-> warp collective operations
-> MMA tile instruction
-> asynchronous copy and transaction barriers
```

Do not expose every feature at the beginning. Each new mechanism should solve a limitation the learner has already felt.

## The final tiny transformer

The first complete target should be intentionally small:

```text
decoder-only transformer
vocabulary: 64 or 128 symbols
sequence length: 16 or 32
d_model: 32
heads: 4
layers: 2
MLP width: 64 or 128
FP16/BF16 storage with FP32 accumulation
character or byte-level tokenizer
```

The machine must:

1. load a deterministic weight file;
2. run embedding, normalization, attention, MLP, residual, and LM-head kernels;
3. match a simple CPU oracle within declared tolerances;
4. generate tokens autoregressively;
5. expose the kernel graph and cost breakdown;
6. let the learner descend from a surprising token or latency spike into the responsible operator, kernel, instruction, and hardware event.

Training is a later checkpoint. Inference provides the earliest satisfying end-to-end milestone.

---

# 3. The web game

The web application is not a decorative wrapper around prose. It is the laboratory.

## The four permanent views

Every checkpoint reuses four synchronized views:

1. **Build view:** wire gates, assemble components, configure pipelines, or connect graph nodes.
2. **Code view:** edit HDL, assembly, kernel code, tensor programs, or compiler passes.
3. **Trace view:** step forward and backward through signals, instructions, lanes, memory requests, barriers, and runtime events.
4. **Measure view:** see cycles, operations, bytes, transactions, bank conflicts, stalls, occupancy, queue delay, link utilization, and numerical error.

Selecting an object in one view highlights its corresponding objects in every other view.

## The game loop

```text
observe a limitation
-> predict the cause
-> build a mechanism
-> pass correctness tests
-> survive hostile cases
-> predict performance
-> measure
-> optimize under a budget
-> explain the causal chain
-> unlock the next machine capability
```

## What makes it fun

- Signals animate through gates and pipelines.
- Lanes have visible identities and masks.
- Global-memory requests merge into transactions in real time.
- Shared-memory banks light up and collide.
- Warps visibly sleep and wake on scoreboards.
- Cache lines and network packets move across the machine.
- Tensor tiles occupy registers and shared-memory slots like physical pieces.
- A timeline can be scrubbed backward after a race or deadlock.
- Optimization challenges impose cycle, byte, area, register, or energy budgets.
- Boss levels use hidden shapes and seeded failure cases.
- Replays can be shared as compact deterministic traces.

The score must never reward speed before correctness. A run with a race, invalid barrier, out-of-bounds access, numerical-contract violation, or invalid benchmark receives no performance rank.

## The hint ladder

Hints should preserve learning:

1. point to the failing invariant;
2. reveal the first divergent trace event;
3. ask a targeted causal question;
4. show a smaller analogous puzzle;
5. reveal pseudocode;
6. reveal a solution only after the learner can explain it.

AI may act as a Socratic debugger, generate new hostile tests, and question explanations. It should not silently replace the learner's construction.

---

# 4. Main quest

The main quest reaches transformer inference without requiring every industrial detail. Each checkpoint has one visible capability, one playful project, and one hard gate.

## World 0 — Signals become a computer

### Checkpoint 0: The Signal Garden

**Build:** bits, truth tables, NAND, NOT, AND, OR, XOR, multiplexers and demultiplexers.

**Web project:** wire a robot's sensor panel using NAND gates. Watch signals propagate and race. Minimize gate count after correctness.

**Hard gate:** synthesize several hidden Boolean functions using only NAND and explain propagation delay through the longest path.

### Checkpoint 1: Arithmetic Foundry

**Build:** half adder, full adder, ripple-carry adder, comparator, shifter, multiplier and integer ALU.

**Web project:** compete in an ALU arena where programs demand different arithmetic operations under gate-count and critical-path budgets.

**Hard gate:** construct a signed ALU, pass overflow and edge cases, and predict which inputs exercise its critical path.

### Checkpoint 2: Clockwork Vault

**Build:** latches, flip-flops, registers, counters, RAM and a program counter.

**Web project:** build a combination-lock machine whose state can be inspected and rewound clock by clock.

**Hard gate:** diagnose seeded setup, stale-state and addressing bugs; then build a small RAM hierarchy from verified components.

### Checkpoint 3: The Scalar Core

**Build:** a small scalar ISA, instruction decoder, register file, fetch/decode/execute loop and load/store path.

**Web project:** guide a rover through a grid by stepping its machine instructions while watching control signals and register updates.

**Hard gate:** implement the CPU from its ISA contract and run hidden programs containing branches, loops, memory accesses and function-like calling conventions.

### Checkpoint 4: Assembly Workshop

**Build:** assembler, labels, symbols, relocatable data, executable image and loader.

**Web project:** write assembly to draw pixels, compute a dot product, and control the rover. The UI links each source line to encoding and datapath activity.

**Hard gate:** extend the ISA with one instruction across hardware, assembler, disassembler, emulator and tests.

### World boss: The Scalar Neural Machine

Implement fixed-point scalar matrix multiplication, ReLU, and a two-layer MLP. Classify a tiny set of glyphs or points. Every output must be traceable to instructions and arithmetic units.

---

## World 1 — One instruction commands many lanes

### Checkpoint 5: Lane Factory

**Build:** an eight-lane SIMD datapath with vector registers, lane IDs and masked execution.

**Web project:** color a strip of pixels in parallel. Rearrange work across lanes to minimize instruction count.

**Hard gate:** map vector addition, SAXPY and thresholding onto lanes, including tails that do not fill the machine width.

### Checkpoint 6: Warp Weaver

**Build:** SIMT threads, per-lane state, predicates, active masks, branch divergence and reconvergence.

**Web project:** lead eight adventurers through branching paths. Divergent lanes visibly split and reconverge while inactive lanes become ghosts.

**Hard gate:** predict active masks for hidden control-flow graphs and repair a kernel whose divergence makes its apparent parallelism mostly idle.

### Checkpoint 7: Block Dispatcher

**Build:** thread blocks, local and global IDs, a block dispatcher, multiple cores and kernel launch metadata.

**Web project:** schedule hundreds of workers across two factories. Change block sizes and watch tail waves and imbalance.

**Hard gate:** derive ownership and bounds for 1D and 2D problems, including awkward dimensions and more blocks than the machine can host.

### Checkpoint 8: The Memory Highway

**Build:** global-memory controllers, cache lines, memory transactions and request coalescing.

**Web project:** lanes send delivery trucks to addresses. Adjacent requests share trucks; strided requests flood the highway.

**Hard gate:** write and repair copy, gather, transpose and structure-layout kernels while predicting exact transaction counts.

### Checkpoint 9: Bank Conflict Heist

**Build:** banked shared memory, shared loads/stores and block barriers.

**Web project:** steal tiles from a bank vault. Simultaneous requests to one bank queue visibly; padding changes the vault layout.

**Hard gate:** implement tiled transpose, eliminate bank conflicts, and explain every barrier and shared-memory address.

### Checkpoint 10: Reduction Reactor

**Build:** atomics, warp shuffles, reduction trees and synchronization scopes.

**Web project:** combine unstable energy cells into one value without races. Different reduction trees expose divergence and numerical differences.

**Hard gate:** implement sum and maximum reductions for irregular sizes; hidden tests include missing barriers, non-associativity and contention.

### World boss: Tiled Matrix Multiplication

Start from one output per thread, then introduce cooperative tiling. The boss records global bytes, shared-memory traffic, lane utilization, synchronization and arithmetic intensity.

---

## World 2 — Latency becomes a scheduling problem

### Checkpoint 11: Scoreboard Station

**Build:** instruction latencies, dependency tracking, ready warps and a warp scheduler.

**Web project:** operate a train station where memory loads are slow trains and independent warps keep platforms busy.

**Hard gate:** predict the schedule of dependent and independent instruction streams; add resident warps until latency is hidden, then explain why more warps stop helping.

### Checkpoint 12: Cache Cartographer

**Build:** configurable L1/L2 caches, tags, replacement, misses, write policy and memory partitions.

**Web project:** draw heatmaps of reuse while changing line size, associativity and traversal order.

**Hard gate:** explain three kernels whose performance ordering reverses under different cache and memory configurations.

### Checkpoint 13: Floating-Point Observatory

**Build:** FP32 and simplified FP16/BF16 encoding, rounding, special values, FMA and conversion.

**Web project:** adjust exponent and mantissa bits while watching representable numbers, rounding boundaries, cancellation, overflow and underflow.

**Hard gate:** predict failures in naive summation, softmax and normalization; implement stable alternatives and state their numerical contracts.

### Checkpoint 14: Tensor Core Forge

**Build:** a small matrix-multiply-accumulate instruction, fragment ownership and mixed-precision accumulation.

**Web project:** arrange matrix fragments across lanes like a tile puzzle; incorrectly placed fragments produce visibly scrambled output.

**Hard gate:** trace every matrix element through lane-owned operands and accumulators, then compare scalar-FMA and MMA execution.

### Checkpoint 15: Pipeline Canyon

**Build:** asynchronous tile copy, double buffering, transaction barriers and pipeline startup/steady-state/drain.

**Web project:** move ore from global memory through shared-memory carts into the Tensor Core furnace without starving either side.

**Hard gate:** draw the dependency graph, produce visible overlap, and diagnose deadlock, premature reuse and resource-exhaustion cases.

### World boss: The GEMM Tournament

Optimize GEMM across square, skinny, small and ragged shapes. The learner must choose different schedules, predict bottlenecks, and defend the result rather than producing one leaderboard number.

---

## World 3 — Programs become kernels

### Checkpoint 16: TensorScript

**Build:** a small high-level language with scalar expressions, tensors, shapes, loops, reductions and kernel functions.

**Web project:** create procedural pixel and tensor programs with immediate output and AST visualization.

**Hard gate:** implement lexer, parser, types and source diagnostics; hidden programs stress precedence, scopes, shape errors and malformed syntax.

### Checkpoint 17: The IR Museum

**Build:** graph IR, control-flow blocks, SSA values, tensor types, layouts and explicit memory effects.

**Web project:** drag operations between high-level graph, loop IR and low-level instruction rooms; def-use edges illuminate across rooms.

**Hard gate:** lower TensorScript into verified IR and implement constant folding, dead-code elimination and common subexpression elimination.

### Checkpoint 18: Parallel Cartographer

**Build:** mapping from tensor coordinates to blocks, warps, lanes, registers and addresses.

**Web project:** paint ownership onto a tensor; the same layout appears simultaneously as logical coordinates, lane colors, addresses and banks.

**Hard gate:** lower map, zip, reduction, transpose and tiled matmul to the TGPU execution model, including masks and barriers.

### Checkpoint 19: Optimization Garden

**Build:** loop interchange, tiling, fusion, vectorization, shared-memory promotion and recomputation.

**Web project:** grow alternative program graphs and compare their live ranges, traffic, operations and synchronization.

**Hard gate:** every transformation must preserve randomized differential tests; optimize a multi-operator graph under a memory budget.

### Checkpoint 20: Register Tetris

**Build:** liveness, instruction scheduling, register allocation and spilling.

**Web project:** pack live values into a limited register board while scheduling instructions around dependencies. Spills fall into slow-memory slots.

**Hard gate:** compile hidden programs without clobbering live values, then trade instruction-level parallelism against register pressure.

### Checkpoint 21: The Autotuner Arena

**Build:** schedule parameterization, benchmark protocol, search, caching and workload distributions.

**Web project:** race candidate tile sizes and stage counts across a season of shapes rather than a single track.

**Hard gate:** predict promising regions before search, prevent benchmark leakage, and select a robust schedule for a held-out shape distribution.

### World boss: Compile GEMM from TensorScript

Compile one high-level matrix multiplication through AST, IR, optimization, mapping, scheduling, allocation, assembly and execution. A source expression must be traceable all the way to a hardware event.

---

## World 4 — Build the machine-learning stack

### Checkpoint 22: Tensor Workshop

**Build:** storage, shape, stride, offset, views, broadcasting and dtype semantics.

**Web project:** reshape and slice physical blocks while a storage panel shows which transformations move data and which only reinterpret it.

**Hard gate:** implement tensors with non-contiguous views and differential-test indexing against a simple reference.

### Checkpoint 23: Gradient River

**Build:** computational graph, reverse-mode automatic differentiation, saved values and gradient accumulation.

**Web project:** release gradient dye at the loss and watch it flow backward, split, accumulate and expose disconnected paths.

**Hard gate:** implement autodiff for broadcasting, reductions, matmul and nonlinearities; verify with finite differences and adversarial shapes.

### Checkpoint 24: Kernel Guild

**Build:** dispatch and correct kernels for elementwise maps, reductions, transpose, GEMM and fused expressions.

**Web project:** choose which guild member executes each graph node, then fuse nodes to reduce materialized intermediates.

**Hard gate:** preserve correctness across dtype, layout and shape grids; performance claims require a named strong baseline.

### Checkpoint 25: Numerical Alchemist

**Build:** stable softmax, online softmax, LayerNorm/RMSNorm and accurate accumulation.

**Web project:** hostile logits attack the laboratory with overflow, underflow, masks and outliers. The learner constructs stable state variables to survive.

**Hard gate:** derive forward and backward formulas from memory, pass extreme-input tests, and explain every higher-precision accumulator.

### Checkpoint 26: Attention Observatory

**Build:** Q/K/V projection, masking, score computation, softmax and value accumulation.

**Web project:** click a token to watch its queries inspect keys and mix values. Toggle causal masks, head layouts and sequence lengths.

**Hard gate:** implement naive attention first, then streaming online attention without materializing the score matrix; compare traffic and numerical behavior.

### Checkpoint 27: Transformer Assembly

**Build:** embeddings, positional representation, RMSNorm, attention, residuals, gated MLP and LM head.

**Web project:** assemble a two-layer decoder from the components built throughout the campaign. Probe activations and descend into any kernel or instruction.

**Hard gate:** match deterministic CPU logits and intermediate checkpoints for multiple prompts, then generate the same token sequence under a declared sampling rule.

### World boss: The Machine Speaks

Load a tiny trained model and generate text on TGPU. Present a complete cost and correctness trace from input tokens to output token. This is the first ending of the game.

---

# 5. Expansion worlds

The main quest establishes the complete vertical map. Expansion worlds develop the depth needed for production and research leadership.

## World 5 — Training and model systems

### Checkpoint 28: Optimizer Engine

Build cross-entropy, SGD and AdamW. Train a tiny classifier and then the tiny transformer. Visualize parameter, optimizer-state, activation and gradient memory.

**Game:** tune the optimizer in a control room where unstable loss, exploding gradients, stale gradients and corrupted parameter updates leave distinct traces.

**Gate:** match a trusted loss curve and locate injected forward, backward, optimizer and numerical defects.

### Checkpoint 29: Memory Strategist

Build activation saving, recomputation, gradient accumulation, mixed precision and loss scaling.

**Game:** fit the largest model through a fixed memory budget without violating the numerical contract.

**Gate:** derive the memory/time tradeoff and verify it against the simulator.

### Checkpoint 30: Graph Commander

Build lazy execution, graph capture, fusion boundaries, buffer reuse and command submission.

**Game:** reduce launches and peak memory while avoiding a fusion that raises register pressure enough to lose.

**Gate:** explain end-to-end changes through kernel and memory timelines.

---

## World 6 — Inference runtime and megakernels

### Checkpoint 31: KV-Cache Warehouse

Build prefill, decode, KV-cache layout, allocation, paging and prefix reuse.

**Game:** serve requests whose cache blocks appear as warehouse shelves; avoid fragmentation and eviction disasters.

**Gate:** trace every live cache block to requests and attention reads under irregular arrivals.

### Checkpoint 32: Continuous-Batching Control Room

Build request admission, batching, chunked prefill, scheduling and sampling.

**Game:** balance TTFT, per-token latency, throughput and fairness while requests arrive in real time.

**Gate:** defend a policy on hidden prefill-heavy, decode-heavy and mixed traces.

### Checkpoint 33: Device Queue Dungeon

Build persistent kernels, atomic work counters, heterogeneous task queues, device events and worker specialization.

**Game:** assign workers to changing task types without starvation, deadlock or wasted residency.

**Gate:** survive empty, overloaded and adversarial queues with deterministic replay.

### Checkpoint 34: Tiny Megakernel

Build one persistent decoder layer and then a device-resident token loop.

**Game:** collapse a launch graph into an on-device runtime while watching scheduler overhead and resource conflicts.

**Gate:** improve a declared end-to-end workload, not merely launch count, and explain every dependency and buffer lifetime.

---

## World 7 — A cluster of machines

### Checkpoint 35: Link Laboratory

Build PCIe/NVLink-like links with latency, bandwidth, queues, routing and topology.

**Game:** connect TGPUs into rings, trees, meshes and switched fabrics while packets animate between devices.

**Gate:** predict transfer time and congestion for hidden message schedules.

### Checkpoint 36: Collective Orchestra

Build broadcast, reduce, all-gather, reduce-scatter, all-reduce and all-to-all.

**Game:** conduct chunks across devices; links and compute units must remain busy without violating dependencies.

**Gate:** implement ring and tree variants, derive their cost models, and choose correctly across message sizes and topologies.

### Checkpoint 37: Parallelism Architect

Build data, tensor, pipeline, context and expert parallel mappings.

**Game:** partition the tiny transformer across a cluster by moving layers, tensor dimensions and tokens between devices.

**Gate:** reproduce expected outputs and predict the first scaling bottleneck before running the simulation.

### Checkpoint 38: Failure Storm

Build timeouts, checksums, checkpointing, restart, straggler detection and deterministic replay.

**Game:** links slow, workers stall and devices disappear during training.

**Gate:** recover without silent model corruption and attribute lost throughput to a concrete failure mechanism.

### World boss: Distributed Tiny Transformer

Train or serve the tiny transformer across several simulated TGPUs. Explain the compute, memory, communication and scheduling critical path.

---

# 6. Real-hardware raids

The toy system is a model, not the destination. Each raid maps a completed abstraction onto a real stack.

## Raid A: WebGPU bridge

Add a WGSL/WebGPU backend for TensorScript. Compare the toy execution model with real browser workgroups, buffers, command encoders and compute pipelines.

**Project:** run the same matmul, softmax and tiny transformer through TGPU functional mode and WebGPU. Preserve correctness while documenting every semantic mismatch.

## Raid B: CUDA and PTX bridge

Add CUDA C++ and PTX backends. Study modules, launches, memory spaces, barriers and the CUDA memory model.

**Project:** compile one TensorScript kernel to TGPU assembly, WGSL, CUDA and PTX; compare how each target represents ownership, memory and synchronization.

## Raid C: SASS microscope

Use `nvcc`, NVVM IR, PTX, `ptxas`, cubins, `cuobjdump` and `nvdisasm`.

**Project:** build a browser report that aligns CUDA source, compiler IR, PTX, SASS, resource usage, profiler metrics and source-level hypotheses.

**Gate:** predict the major generated instruction and dependency structures before compilation and explain discrepancies afterward.

## Raid D: Open GPU implementation

Rebuild important TGPU components in Verilog and compare them with `tiny-gpu`. Then extend or instrument Vortex and validate performance ideas with Accel-Sim.

**Project:** add one instruction or scheduling feature across RTL, simulator, compiler, runtime and kernel.

## Raid E: Modern Tensor Core architectures

Build architecture packs for Ampere, Hopper and Blackwell concepts: instruction shapes, shared-memory layouts, asynchronous transfer, barriers, warp specialization, TMEM and cluster cooperation.

**Project:** port the GEMM and attention worlds to each architecture pack and make the learner discover which old optimization becomes the new bottleneck.

## Raid F: Production attention

Reconstruct FlashAttention from online softmax, then FA2/FA3/FA4-style work partitioning and pipelines.

**Project:** receive a new mask, head layout or attention recurrence and optimize it for a selected real architecture without reading a production implementation first.

## Raid G: Production runtime

Integrate a kernel backend into a compact serving engine, then a production engine. Study CUDA Graphs, KV paging, batching, scheduling and sampling.

**Project:** trace one latency change from request policy through kernel dispatch and machine instructions to hardware stalls.

## Raid H: Real distributed training

Implement small collectives, compare NCCL and NVSHMEM, and build data/tensor/pipeline parallel training before studying full frameworks.

**Project:** train a small transformer across multiple GPUs and then nodes; predict scaling, inject failures, and explain the first saturation point.

---

# 7. Research endgame

The final levels are not predefined implementations. They test whether the learner can transfer the constructed mental model.

## New-hardware expedition

Given an unfamiliar architecture:

1. extract its resource and execution model;
2. construct instruction and memory microbenchmarks;
3. add an architecture pack to the simulator;
4. predict how existing kernels will fail or regress;
5. port GEMM and attention;
6. publish a falsifiable optimization guide.

## New-attention expedition

Given a novel attention algorithm:

1. derive its exact and approximate numerical contract;
2. derive operations, bytes and state;
3. choose ownership and tiles;
4. construct a pipeline simulator;
5. implement a correct reference and optimized kernel;
6. integrate it into inference or training;
7. explain whether algorithm changes are required by hardware asymmetry.

## Architecture/compiler/kernel co-design

Add one proposed hardware feature, compiler abstraction or instruction to TGPU. Implement the compiler and kernel changes needed to exploit it. Compare area/complexity proxies, generality and end-to-end transformer value.

## Production defense

A final claim must include:

- differential correctness;
- adversarial and numerical testing;
- compiler and machine-code inspection;
- profiler evidence;
- shape and workload distributions;
- end-to-end measurements;
- failure analysis;
- an external expert review;
- a clear statement of what remains unexplained.

---

# 8. Mastery system

## Ranks

Each checkpoint has four ranks:

| Rank          | Evidence                                                                     |
| ------------- | ---------------------------------------------------------------------------- |
| **Builder**   | The component passes visible functional tests.                               |
| **Debugger**  | It passes hidden, adversarial and seeded-failure tests.                      |
| **Engineer**  | The learner predicts and improves a measured cost under constraints.         |
| **Architect** | The learner transfers the mechanism to a new problem and defends the design. |

Completion of prose or videos earns no rank.

## The permanent notebook

For every checkpoint record:

```text
mechanism:
invariant:
resource model:
prediction:
first failure:
root cause:
repair:
measurement:
remaining uncertainty:
transfer rule:
```

## Spaced reconstruction

- End of session: reconstruct the mechanism without notes.
- Next day: rebuild the critical component from a blank scaffold.
- One week later: solve a changed version.
- One month later: use it inside a higher-level system without reopening the original solution.

## External feedback

Self-testing is necessary but insufficient for elite performance. Require periodic defenses with specialists in:

- digital design and microarchitecture;
- compilers;
- numerical computing;
- GPU kernels;
- ML runtimes;
- networks and distributed training.

The best evidence is code and analysis that survive upstream review, independent reproduction and production workloads.

---

# 9. Implementation architecture for the web laboratory

The educational system should itself preserve clean boundaries.

## Simulator core

- Pure, deterministic state transition functions.
- Separate functional and timing engines sharing ISA semantics.
- Versioned machine configuration and ISA.
- Golden reference models independent of learner components.
- Snapshot and reverse-step support.
- Seeded deterministic randomness.

## Trace schema

Every layer emits a common event envelope:

```text
time
layer
component
operation
inputs
outputs
dependencies
resource deltas
source location
parent event
```

The parent link is what permits vertical zoom from transformer output to hardware event.

## Execution isolation

- Run learner code in a Worker or sandbox.
- Apply instruction and memory budgets.
- Make infinite loops interruptible.
- Keep hidden tests outside the learner-visible bundle where possible.
- Validate serialized projects before loading.

## Performance strategy

- TypeScript is sufficient for the first functional models and UI.
- Move hot simulation loops to WebAssembly only after profiling.
- Use virtualized trace views and bounded trace retention.
- Run full transformers in functional mode.
- Capture representative kernel windows for timing-mode inspection.
- Use WebGPU as a later compilation target, not as the implementation of the teaching simulator.

## Content format

Each checkpoint should be data-driven:

```text
brief
machine capabilities
starter artifact
visible tests
hidden tests
cost model
hint ladder
unlock artifact
reference explanation
transfer challenge
```

This makes the curriculum extendable without entangling educational content with simulator code.

---

# 10. Recommended campaign cadence

The main quest should be playable in a concentrated several-month campaign. The complete mastery path is a multi-year apprenticeship.

Use gates rather than fixed calendar promises:

```text
main quest to transformer inference
-> training and runtime expansions
-> distributed expansion
-> real-hardware raids
-> original research and production work
```

An aggressive full-time rhythm is:

- 55% construction and debugging;
- 15% derivation and targeted reading;
- 15% experimentation and measurement;
- 10% retrieval, explanation and writing;
- 5% external review and curriculum adjustment.

Every week should end with:

1. one reusable component;
2. one hostile test set;
3. one prediction-versus-measurement report;
4. one closed-book explanation;
5. one transfer problem.

---

# 11. Scheduled intensive calendar

The first complete campaign is scheduled from **July 14, 2026 through September 3, 2028**. It contains 783 consecutive two-hour sessions: approximately 112 weeks and 1,566 focused hours.

This is an aggressive foundation-to-research apprenticeship, not a claim that calendar completion alone establishes mastery. Advancement is earned by the gate at the end of each phase.

| Phase                                        | Dates                     | Duration | Hours | Outcome                                                                                                                   |
| -------------------------------------------- | ------------------------- | -------: | ----: | ------------------------------------------------------------------------------------------------------------------------- |
| **0. Baseline and laboratory**               | Jul 14–26, 2026           |  13 days |    26 | Placement tests, deficiency map, reproducible CPU/GPU laboratory, evidence ledger and TGPU simulator skeleton             |
| **1. NAND to scalar neural machine**         | Jul 27–Sep 20, 2026       |  8 weeks |   112 | Checkpoints 0–4: logic, arithmetic, memory, ISA, processor, assembler and fixed-point MLP                                 |
| **2. SIMT architecture and memory**          | Sep 21–Nov 29, 2026       | 10 weeks |   140 | Checkpoints 5–10: lanes, divergence, blocks, coalescing, shared memory, atomics, reductions and tiled GEMM                |
| **3. Timing, numerics and Tensor Cores**     | Nov 30, 2026–Jan 24, 2027 |  8 weeks |   112 | Checkpoints 11–15: scoreboards, scheduling, caches, floating point, MMA and asynchronous pipelines                        |
| **4. Compiler, PTX and SASS**                | Jan 25–Apr 18, 2027       | 12 weeks |   168 | Checkpoints 16–21: TensorScript, SSA IR, lowering, optimization, scheduling, register allocation and machine-code tracing |
| **5. CUDA kernel forge**                     | Apr 19–Jul 25, 2027       | 14 weeks |   196 | Production primitive ladder, sanitizers, profilers, CUTLASS/CuTe and architecture-specific optimization                   |
| **6. ML stack, attention and transformer**   | Jul 26–Oct 31, 2027       | 14 weeks |   196 | Checkpoints 22–30: tensors, autodiff, numerics, attention, transformer, optimization, mixed precision and graph execution |
| **7. Runtime, serving and megakernels**      | Nov 1, 2027–Jan 23, 2028  | 12 weeks |   168 | Checkpoints 31–34: KV cache, batching, scheduling, device queues, persistent execution and serving integration            |
| **8. Distributed GPU systems**               | Jan 24–Apr 16, 2028       | 12 weeks |   168 | Checkpoints 35–38: links, collectives, parallelism, overlap, checkpointing, failures and multi-node scaling               |
| **9. Real hardware and research expedition** | Apr 17–Sep 3, 2028        | 20 weeks |   280 | WebGPU/CUDA/RTL raids followed by one original architecture/compiler/kernel/runtime research or production contribution   |

## Daily 4:30–6:30 PM protocol

The study block deliberately occupies a low-energy late-afternoon period. It must therefore minimize startup decisions and force active work before fatigue can turn the session into passive reading.

```text
16:30–16:45  closed-book retrieval and state today's prediction
16:45–17:45  one pre-scoped construction or derivation
17:45–17:55  walk and reset
17:55–18:20  hostile tests, disassembly, profiling or measurement
18:20–18:30  error log, preserve artifact, write tomorrow's first action
```

The final ten minutes are part of the next day's session: the learner must leave a runnable command, failing test, open trace, or precise first edit. The exhausted learner should never begin by choosing a resource or deciding what to build.

## Weekly rhythm

| Day           | Direct performance                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| **Monday**    | Derive and map the week's mechanism from first principles. State invariants and performance predictions.   |
| **Tuesday**   | Build the smallest correct implementation.                                                                 |
| **Wednesday** | Add hostile cases, instrumentation and the next layer of realism.                                          |
| **Thursday**  | Optimize one causal variable and inspect the resulting trace or machine code.                              |
| **Friday**    | Debugging, profiler, disassembly, layout or numerical drill.                                               |
| **Saturday**  | Integration project or boss challenge under a fixed two-hour constraint.                                   |
| **Sunday**    | Closed-book reconstruction, held-out transfer test, evidence review and pre-scope the next seven sessions. |

## Gate rule

A calendar boundary does not override evidence. If a phase gate fails:

1. name the failed invariant or transfer skill;
2. use the next phase's first sessions for the smallest corrective drill;
3. retake a changed version of the gate;
4. advance only after the failure is explained and repaired;
5. record the schedule variance rather than quietly lowering the standard.

The calendar supplies pressure and consistency. The gates preserve truth.

---

# 12. Research references

## Constructionist computing and visual machines

- [Nand2Tetris projects](https://www.nand2tetris.org/course) — cumulative construction from Boolean logic through hardware, machine language, compiler and OS.
- [NandGame](https://nandgame.com/) — browser-native gate-to-computer puzzle structure.
- [Ripes](https://github.com/mortbopet/Ripes) — visual processor, assembly, pipeline, cache and memory-mapped-I/O exploration, including a browser build.

## Teaching and open GPUs

- [GPU Puzzles](https://github.com/srush/GPU-Puzzles) — interactive kernel puzzles with visible access accounting.
- [tiny-gpu](https://github.com/adam-maj/tiny-gpu) — minimal documented Verilog GPU with an ISA, simulator, traces and matrix kernels.
- [Vortex](https://github.com/vortexgpgpu/vortex) — full-stack open RISC-V GPGPU with software simulation, RTL, FPGA targets, runtime and compiler toolchain.
- [Accel-Sim](https://github.com/accel-sim/accel-sim-framework) — detailed GPU simulation framework for advanced validation.

## Compiler construction

- [LLVM Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html) — staged language frontend and code-generation tutorial.
- [MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) — AST, dialect, transformations, shape inference, lowering and LLVM code generation.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) — NVIDIA's documented virtual GPU ISA.
- [NVVM IR specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html) — compiler IR contract for CUDA toolchains.
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) — cubin inspection, `cuobjdump`, `nvdisasm` and architecture instruction references.

## Small ML systems

- [MiniTorch](https://minitorch.github.io/) — pedagogical path through autodiff, tensors, efficiency, networks and GPU programming.
- [tinygrad](https://github.com/tinygrad/tinygrad) — compact end-to-end tensor, autograd, compiler, JIT and runtime stack.
- [`llm.c`](https://github.com/karpathy/llm.c) — readable C/CUDA GPT training with a small CPU reference and differential tests.
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — production-oriented readable inference across many hardware backends.

## Browser compute target

- [WebGPU specification](https://www.w3.org/TR/webgpu/) — browser GPU buffers, command submission and compute pipelines.
- [WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/) — workgroups, address spaces, barriers and shader semantics.
- [WebGPU samples](https://webgpu.github.io/webgpu-samples/) — executable browser examples for the real-hardware bridge.

---

# Final principle

The course succeeds when the transformer is no longer magic.

A generated token should be understandable as the visible consequence of tensor algebra, numerical choices, compiler transformations, scheduled instructions, memory movement, synchronization and hardware resources. The learner should be able to modify any layer, predict the consequences for the others, and verify the prediction.

That is the real journey from NAND to transformers.
