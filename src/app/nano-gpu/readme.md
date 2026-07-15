# Nano GPU

Nano GPU is an interactive browser laboratory for learning GPU systems from
first principles. The long-term goal is to let a learner describe a small GPU,
compile a CUDA-like kernel for it, and watch every instruction move through the
machine until the GPU can run a small transformer.

The public route is `/nano-gpu`.

## Current state

The route contains a static placeholder page and a runnable functional GPU core
in `_core/gpu.ts`. The model explicitly includes HBM, a shared L2 cache,
multiple SMs, per-SM shared memory, round-robin warp schedulers, per-warp
register files, lane-level CUDA cores, and load/store units.

`_core/gpu_test.ts` launches four warps over two SMs and checks that the complete
machine writes `0..15` through L2 into HBM. The HDL compiler, program compiler,
Web Worker, and visualizer remain intentionally unimplemented.

### Implemented functional model

```text
GPU
├── HBM
├── shared write-through L2
└── SMs
    ├── round-robin warp scheduler
    ├── warps and register files
    ├── one CUDA core per active lane
    ├── load/store unit
    └── shared memory
```

Each SM may issue one warp instruction per GPU cycle. CUDA cores execute scalar
lane operations, while the load/store unit routes global accesses through L2
and shared accesses to the SM-local memory. Every issued instruction returns a
serializable trace event containing its active lanes and register or memory
effects.

This is a deterministic educational functional model, not a cycle-accurate
NVIDIA implementation. L2 is fully associative, LRU, and write-through; memory
is word-addressed; instructions complete in one issue step; and the current
model omits divergence, barriers, pipelines, bank conflicts, coalescing, tensor
cores, and asynchronous memory operations.

## Product principles

1. **Build upward from first principles.** The learner should be able to trace a
   result from a logic gate through the GPU, instruction set, compiler, kernel,
   and model.
2. **Make hidden state visible.** Registers, active lanes, program counters,
   memory transactions, stalls, barriers, and scheduler decisions should be
   inspectable.
3. **Keep the feedback loop immediate.** Editing, compiling, stepping, and
   resetting should feel like a browser REPL rather than a remote job queue.
4. **Prefer small complete machines.** A tiny system that runs end to end teaches
   more than a large collection of disconnected production abstractions.
5. **Maintain deterministic execution.** Given the same machine, program, and
   launch configuration, the simulator must produce the same trace.
6. **Use one representation at each boundary.** Hardware compiles to Machine IR,
   programs compile to Program IR, and execution produces a Trace.

## System model

Nano GPU has two compilation paths that meet inside one deterministic simulator:

```text
HDL source ──parse and elaborate──▶ Machine IR ───────────┐
                                                           │
CUDA source ──nvcc──▶ PTX ──lower──▶ Program IR ──────────┼──▶ Simulator
                                                           │        │
Launch configuration ──────────────────────────────────────┘        ▼
                                                                  Trace
                                                                    │
                                                                    ▼
                                                               Visualizer
```

### Machine IR

Machine IR is a serializable description of the virtual hardware. It should
contain stable identifiers and source locations for:

- compute cores and execution lanes;
- warp state and scheduling policy;
- registers, predicates, and program counters;
- instruction, shared, constant, and global memory;
- functional units and supported operations;
- connections between components;
- latency and throughput values used by the simulator.

The frontend draws the hardware from this representation. The simulator also
executes this same representation, preventing the diagram and the machine from
becoming separate models.

### Program IR

Program IR is the small executable instruction format consumed by the
simulator. The first version should support only the instructions needed for a
vector-add program:

```text
MOV, ADD, MUL, LOAD, STORE, SETP, BRA, EXIT
```

PTX syntax and NVIDIA-specific behavior belong in the PTX lowering stage, not
inside the simulator. This keeps the machine capable of accepting other small
languages later without changing its execution engine.

### Trace

Every simulator cycle produces a snapshot delta and a list of semantic events.
For example:

```text
cycle_started
instruction_issued
register_written
memory_requested
warp_stalled
barrier_released
cycle_finished
```

The visualizer consumes these events to explain why state changed. Storing
deltas rather than full snapshots also gives the timeline efficient rewind and
replay.

## Target route structure

Files use lower snake case. Next.js framework filenames such as `page.tsx` and
`route.ts` retain their required names. The URL directory remains `nano-gpu`
because it defines the requested public path.

```text
src/app/nano-gpu/
├── page.tsx
├── readme.md
├── _components/
│   ├── gpu_lab.tsx
│   ├── code_editor.tsx
│   ├── machine_view.tsx
│   ├── state_view.tsx
│   ├── timeline.tsx
│   └── controls.tsx
├── _core/
│   ├── machine_ir.ts
│   ├── program_ir.ts
│   ├── hdl_parser.ts
│   ├── hdl_compiler.ts
│   ├── ptx_parser.ts
│   ├── ptx_compiler.ts
│   ├── simulator.ts
│   ├── trace.ts
│   └── examples.ts
└── _workers/
    ├── worker_protocol.ts
    └── gpu_worker.ts

src/app/api/nano-gpu/compile/
└── route.ts
```

The underscore directories are private implementation folders and do not
become Next.js routes.

Do not create this entire tree upfront. Add a file only when an executable
checkpoint needs it. If a file remains very small, keep related behavior
together until separation improves understanding.

## Runtime boundary

The browser owns the learner-facing system:

- HDL parsing and compilation;
- PTX parsing and lowering;
- simulation;
- trace generation;
- visualization and interaction.

Compilation and simulation run in a Web Worker so an infinite kernel or long
trace cannot freeze the page. The React application sends commands and receives
serializable results:

```ts
type worker_request =
  | { type: 'compile_hardware'; source: string }
  | { type: 'compile_program'; source: string }
  | { type: 'run'; launch: launch_config }
  | { type: 'step'; cycles: number }
  | { type: 'reset' }

type worker_response =
  | { type: 'compiled'; machine: machine_ir; program: program_ir }
  | { type: 'advanced'; delta: state_delta; events: trace_event[] }
  | { type: 'failed'; diagnostics: diagnostic[] }
```

The UI should never mutate simulator state directly. It renders the snapshots,
deltas, and events returned by the worker.

## Native CUDA boundary

Browsers cannot run NVIDIA's `nvcc` compiler. The first browser-only version
should therefore accept bundled or pasted PTX.

A later `POST /api/nano-gpu/compile` endpoint may compile a restricted CUDA
source file to PTX when the server environment provides the CUDA toolkit. That
endpoint must use a temporary directory, fixed compiler flags, a strict timeout,
input and output size limits, and no user-controlled shell command.

Most serverless deployments do not include `nvcc`. If the personal site is
hosted in such an environment, native CUDA compilation should live in a small
isolated compile service. The browser architecture remains unchanged because
both approaches return PTX.

## Implementation checkpoints

### 1. Clocked toy machine

Implement one core, four lanes, registers, instruction memory, and global
memory. Parse a minimal HDL description into Machine IR and animate a manually
written add program one cycle at a time.

**Done when:** changing the HDL visibly changes the rendered machine, and the
same input always produces the same trace.

### 2. Tiny assembler

Add labels, branches, predicates, loads, stores, and a compact textual assembly
language that compiles to Program IR.

**Done when:** a learner can write vector addition in the browser, run it, and
explain every register and memory change from the trace.

### 3. SIMT execution

Add warps, active masks, divergent branches, reconvergence, barriers, and a
round-robin warp scheduler.

**Done when:** the visualizer makes divergence and latency hiding obvious
without requiring a prose explanation.

### 4. PTX subset

Parse and lower only the PTX instructions required by the existing examples.
Unsupported syntax must produce source-linked diagnostics rather than silently
guessing.

**Done when:** PTX for vector addition produces the same Program IR behavior as
the handwritten assembly version.

### 5. Memory hierarchy

Add shared memory, memory coalescing, bank conflicts, configurable latency, and
simple caches only after the earlier execution model is stable.

**Done when:** two correct kernels with different access patterns have visibly
different traces and cycle counts for explainable reasons.

### 6. Matrix multiplication

Implement tiled matrix multiplication, then introduce a small tensor or matrix
instruction. Connect each optimization to a measurable machine bottleneck.

**Done when:** the learner can predict whether an optimization helps before
running it and verify the prediction in the trace.

### 7. Tiny transformer

Compose matrix multiplication, normalization, softmax, attention, and elementwise
operations into a small transformer inference graph.

**Done when:** the virtual GPU executes a complete forward pass and every output
can be traced back through instructions and hardware state.

## First executable slice

The first real implementation should stay deliberately small:

```text
page.tsx
_components/gpu_lab.tsx
_core/machine_ir.ts
_core/hdl_compiler.ts
_core/simulator.ts
_workers/gpu_worker.ts
```

Use a plain styled `textarea` initially. Add Monaco only when compiler
diagnostics need inline markers or navigation. Use SVG for the first machine
view because components, wires, labels, and click targets map naturally to the
DOM.

## Verification

Each checkpoint needs one deterministic end-to-end fixture containing:

- source text;
- expected Machine IR or Program IR;
- expected final state;
- expected significant trace events;
- a maximum cycle count that catches infinite execution.

Parser unit tests are useful, but the main acceptance test is always the same:
compile a small program, run the virtual hardware, and verify both the answer
and the explanation trace.

## Non-goals for the first versions

- Complete CUDA or PTX compatibility
- Cycle-accurate reproduction of a current NVIDIA GPU
- Realistic transistor timing or physical layout
- A generic compiler framework
- Multiple frontend applications or a monorepo
- Production-scale transformer execution

Nano GPU should first become a small, truthful machine that a learner can hold
entirely in their head. Fidelity and scale can then increase one observable
bottleneck at a time.
