# NanoGPU physical simulator architecture

## Purpose

NanoGPU is a browser laboratory for constructing a small GPU from first
principles, compiling programs for it, and watching computation travel from a
CUDA-like kernel down through instructions, pipelines, registers, gates, and
wires.

The final machine should be capable of running a tiny transformer. The same
machine should also let a learner pause inside one clock cycle and inspect why
a signal changed.

Those goals require different amounts of detail. The architecture therefore
uses one hierarchical machine description with selectable simulation fidelity:

```text
CUDA or assembly source                 HDL or visual circuit
          │                                      │
          ▼                                      ▼
      Program IR                             Machine IR
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
                     simulation worker
                    ┌────────┴────────┐
                    │                 │
             functional mode   physical-timing mode
                    │                 │
                    └────────┬────────┘
                             ▼
                    state deltas + trace
                             │
                             ▼
                       browser viewer
```

The central rule is:

> Model every boundary truthfully, but expand detail only where it teaches
> something.

A transformer needs fast instruction-level execution. An ALU lesson needs
gate-and-wire timing. A PLL lesson may need an isolated analog model. Requiring
the entire transformer to pass through every transistor would make the lab too
slow to use and harder, rather than easier, to understand.

## What “physical” means

The whole-GPU simulator is a **digital physical-timing simulator**. It models:

- logic values and unknown states;
- clock edges and clock domains;
- propagation through gates and wires;
- register setup, hold, and clock-to-output timing;
- pipelines, queues, arbitration, backpressure, and contention;
- memory banks, caches, links, and bandwidth;
- configurable geometry, latency, power, and area annotations.

It does not solve electromagnetic fields or transistor differential equations
for the entire chip. Transistor, SRAM-cell, PHY, and PLL simulations belong in
small zoom-in laboratories with explicit analog solvers.

Real timing accuracy eventually requires characterized standard cells,
placement, routing, extracted parasitics, and process-voltage-temperature
conditions. Until those exist, NanoGPU’s delays are deterministic educational
parameters rather than predictions about fabricated silicon.

## The four notions of time

The browser, learner, clock, and electrical model operate at different rates.
They must not share a single “game tick.”

| Time concept         | Meaning                                       | Recommended unit           |
| -------------------- | --------------------------------------------- | -------------------------- |
| Simulation timestamp | Physical time represented by the model        | Integer picoseconds        |
| Event step           | The next scheduled hardware transition        | One event-time batch       |
| Delta cycle          | Causal settling without physical time passing | One fixed-point iteration  |
| Render frame         | When the browser paints the current state     | About 60 frames per second |

Use `bigint` picoseconds internally:

```ts
type time_ps = bigint
```

One picosecond is the timestamp resolution, not a mandatory fixed step. If
nothing happens between 17 ps and 42 ps, the engine jumps directly from 17 ps
to 42 ps.

```text
0 ps      clock rises
5 ps      register Q changes
17 ps     value reaches the ALU through a wire
42 ps     ALU output changes
53 ps     result reaches the destination register
1000 ps   next clock edge captures the result
```

The smallest physical timestamp is therefore **1 ps**. The smallest causal
operation is a **delta cycle**, which consumes zero physical time. The normal
learner-facing step should be a **clock edge**.

## Why the engine is event-driven

A fixed-step loop would revisit every component at every picosecond, even when
almost all components are idle. NanoGPU instead uses a deterministic priority
queue ordered by:

```text
(time_ps, delta, insertion_order)
```

The engine repeatedly takes all events at the earliest timestamp, applies
them, settles affected combinational logic, and schedules future transitions.

```ts
while (event_queue.length > 0) {
  const time = event_queue.next_time()
  now = time

  while (event_queue.has_events_at(time)) {
    const events = event_queue.take_next_delta_batch(time)
    apply(events)
    evaluate_affected_components()
  }

  record_stable_boundary()
}
```

All events with the same `(time_ps, delta)` are applied as a batch. Stateful
elements sample their inputs before any state commits, so the result does not
depend on JavaScript object iteration order.

### Delta cycles

Delta cycles settle zero-delay combinational dependencies:

```text
17 ps, delta 0: register output changes
17 ps, delta 1: mux output changes
17 ps, delta 2: decoder output changes
17 ps, delta 3: circuit is stable
```

The engine must detect a circuit that never settles. A maximum delta count and
the changing signal path should be reported as a combinational-loop diagnostic.

### Determinism

Given the same machine, program, inputs, and random seed, a run must emit the
same state and trace. Determinism requires:

- integer time rather than floating-point time;
- an explicit tie-breaker for simultaneous events;
- simultaneous sampling and commit at clock edges;
- seeded jitter and arbitration when randomness is enabled;
- no simulator behavior derived from render timing.

## How a value travels through a wire

A wire is a connection with a source, one or more destinations, and a delay
model. A source transition schedules an arrival event at each destination:

```ts
for (const sink of wire.sinks) {
  schedule({
    time_ps: now + wire.delay_ps,
    target: sink,
    value: source.value,
  })
}
```

The engine does not need to divide the wire into microscopic spatial cells.
The frontend can animate a pulse between the departure and arrival timestamps.

Two delay policies are useful:

- **Transport delay:** every transition arrives later, including short pulses.
- **Inertial delay:** a pulse shorter than the component’s rejection window is
  cancelled.

Transport delay is the simplest starting model for wires. Inertial delay can
be added for gates when pulse filtering becomes an explicit lesson.

## Logic and electrical state

The timing engine should use four-state digital logic:

```ts
type logic = '0' | '1' | 'x' | 'z'
```

- `0` and `1` are driven logical values.
- `x` means unknown, conflicting, or timing-invalid.
- `z` means undriven or high impedance.

Small buses can initially be represented as arrays of logic values. Packed
bit-vectors should replace them only when profiling shows that representation
cost dominates simulation.

Unknown values are educationally important. Uninitialized registers, competing
drivers, setup violations, and unresolved memory should become visible rather
than silently turning into zero.

## Stateful elements

### Registers and flip-flops

At an active clock edge, a register:

1. samples `D` and control inputs;
2. checks setup and hold constraints when timing checks are enabled;
3. commits all sampled state simultaneously;
4. schedules `Q` after its clock-to-output delay.

```text
clock edge ──sample D──▶ internal state ──t_cq──▶ Q transition
```

A violated timing constraint should drive the affected state to `x` and emit a
source-linked diagnostic. It should not cause nondeterministic behavior.

### Memories

Large SRAM, register files, caches, and HBM should normally be behavioral
macros with explicit ports, banks, queues, throughput, and latency. Expanding
every bit cell is useful only in a dedicated memory-cell lab.

The observable contract of a memory macro includes:

- address, data, byte-enable, and control ports;
- request acceptance and response timestamps;
- bank selection and conflicts;
- read/write ordering;
- finite queues and backpressure;
- optional energy, area, and wire-distance annotations.

### Clocks and PLLs

Each clock domain is a behavioral event source:

```ts
type clock_domain = {
  id: string
  period_ps: bigint
  phase_ps: bigint
  duty_cycle: number
  jitter_ps?: bigint
  seed?: number
}
```

SM, interconnect, L2, and HBM domains may run at different periods. The global
event queue naturally interleaves their edges.

A whole-chip PLL model should expose input frequency, output frequency, phase,
lock time, jitter, and DVFS transitions. Its transistor-level charge-pump and
oscillator behavior should remain a separate zoom-in experiment.

## Hierarchical fidelity

Fidelity is selected per component, not once for the entire machine.

```text
GPU
├── HBM controller                     behavioral timing macro
├── L2 slices                          cycle/event model
├── interconnect                       routers, links, queues
└── SM
    ├── warp scheduler                 cycle/event model
    ├── register file                  banked timing macro
    ├── shared memory                  banked timing macro
    ├── load/store unit                pipeline model
    └── execution unit
        ├── instruction pipeline       cycle/event model
        └── selected ALU               gates and wires
```

Every component has a stable port contract. A behavioral component can be
replaced by a structural implementation without changing its parent:

```text
ALU behavior        ports A, B, op, result, valid
ALU pipeline        ports A, B, op, result, valid
ALU gate netlist    ports A, B, op, result, valid
```

This provides continuous zoom without requiring three disconnected teaching
machines.

### Fidelity levels

| Level          | Simulated unit                                | Best use                                 |
| -------------- | --------------------------------------------- | ---------------------------------------- |
| Functional     | Warp instruction or memory operation          | Whole programs and transformers          |
| Cycle          | Pipeline stage, queue, bank, or link transfer | GPU architecture and performance         |
| Digital timing | Gate, register, port, and wire transition     | Datapaths and timing lessons             |
| Analog lab     | Voltage, current, transistor, RC network      | PLL, SRAM cell, PHY, and circuit lessons |

Functional and physical-timing runs must agree at architectural commit points:
retired instructions, register values, memory writes, exceptions, and kernel
completion. Differential tests between the modes prevent the detailed model
from becoming a different GPU.

## The intermediate representations

### Machine IR

Machine IR is the single serializable description used to instantiate both the
simulator and diagram. It is hierarchical rather than a flat list of every
gate in the chip.

```ts
type machine_ir = {
  version: 1
  root: component_instance
  components: component_definition[]
  clocks: clock_domain[]
  initial_state: state_value[]
  source_map: source_location[]
}

type component_definition = {
  id: string
  ports: port_definition[]
  implementation:
    | { kind: 'behavioral'; model: string; parameters: object }
    | { kind: 'structural'; instances: component_instance[]; wires: wire[] }
  timing?: timing_model
  geometry?: geometry
}
```

Required properties:

- stable identifiers for every inspectable object;
- source locations back to HDL;
- explicit ports, widths, directions, clocks, and reset behavior;
- structural children and wires where expanded;
- behavioral model names where collapsed;
- timing, geometry, and physical annotations kept optional;
- no functions, class instances, DOM objects, or cyclic references.

The visualizer must draw the elaborated Machine IR. A separately maintained
diagram would eventually disagree with the machine being executed.

### Program IR

Program IR is the small executable ISA consumed by the simulated instruction
front end. CUDA, PTX, and toy assembly syntax are lowered before execution:

```text
CUDA subset ─▶ PTX subset ─┐
                           ├─▶ Program IR ─▶ instruction memory
toy assembly ──────────────┘
```

The simulator should understand the machine ISA, not CUDA grammar or PTX text.
This keeps compiler diagnostics, instruction semantics, and hardware behavior
at clean boundaries.

### Simulation state

Mutable state is stored separately from Machine IR:

```ts
type simulation_state = {
  now_ps: bigint
  delta: number
  values: Map<string, logic_vector>
  memories: Map<string, memory_state>
  component_state: Map<string, unknown>
  event_queue: event_queue
}
```

Machine IR describes what exists. Simulation state describes its current
condition. Program IR describes what it has been asked to execute.

### Trace

A trace explains causal changes rather than dumping complete state after every
event:

```ts
type trace_event = {
  id: number
  time_ps: bigint
  delta: number
  kind: string
  target_id: string
  cause_id?: number
  changes: state_delta[]
  source?: source_location
}
```

The `cause_id` creates the vertical explanation chain:

```text
kernel output
  caused by STORE instruction
    caused by warp issue
      caused by scoreboard wakeup
        caused by L2 response
          caused by HBM transaction
```

Use periodic checkpoints plus reversible deltas. Full state snapshots after
every event will exhaust browser memory quickly.

## Event types

The initial timing engine needs only a small event vocabulary:

```text
clock_edge
port_driven
wire_arrived
component_evaluate
state_commit
memory_request
memory_response
diagnostic
```

GPU-specific semantic events can be layered on top:

```text
instruction_fetched
warp_issued
lane_executed
register_written
warp_stalled
cache_hit
cache_miss
barrier_released
instruction_retired
```

Electrical events drive state. Semantic events explain the same state changes
at the learner’s current level. They should reference each other rather than
being generated by unrelated simulators.

## Browser runtime architecture

Simulation belongs in a Web Worker. React should never own or mutate live
hardware state.

```text
React UI                          Web Worker
────────                          ──────────
editor source ──compile────────▶ HDL/PTX compilers
controls      ──command────────▶ simulator
views         ◀─trace chunk──── state + event queue
breakpoints   ──filters────────▶ trace recorder
```

The worker may process many hardware events between browser frames. Rendering
must not advance simulation time.

The UI should offer these controls:

- **settle:** execute one delta cycle;
- **event:** advance to the next event-time batch;
- **edge:** advance to the next selected clock edge;
- **cycle:** advance one full period in a selected clock domain;
- **instruction:** run until the selected warp retires an instruction;
- **run:** advance until pause, breakpoint, completion, or budget exhaustion;
- **rewind:** restore the nearest checkpoint and replay deltas;
- **reset:** restore the deterministic initial state.

The frontend should have synchronized views for:

1. HDL, PTX, assembly, and source-linked diagnostics;
2. hierarchical chip layout and schematic;
3. registers, memories, queues, and component state;
4. waveform and causal timeline;
5. active lanes, warps, pipelines, and memory transactions;
6. cycles, bytes, conflicts, stalls, utilization, energy, and area estimates.

Selecting an object in any view should highlight its source, state, events,
parents, children, and physical connections.

## Scaling to a transformer

The simulator should never attempt to retain every gate transition for an
entire transformer run.

Use three complementary mechanisms:

1. **Functional fast path:** execute the complete model and identify an
   interesting kernel, instruction range, or performance anomaly.
2. **Selective expansion:** replace only chosen components with cycle or gate
   implementations while their neighbors remain behavioral.
3. **Trace windows:** record detailed events only around selected time ranges,
   components, signals, or causal chains.

Typical workflow:

```text
run tiny transformer in functional mode
  ▶ discover attention is slow
  ▶ replay the attention kernel in cycle mode
  ▶ discover shared-memory bank conflicts
  ▶ expand one shared-memory path and address decoder
  ▶ inspect the responsible gate and wire events
```

This is more informative than permanently simulating all details because it
lets the learner move between system behavior and the mechanism causing it.

## Compilation path

### HDL to runnable virtual hardware

“Compiling HDL into a GPU” means elaborating source into the Machine IR that
the event engine can instantiate:

```text
HDL text
  ▶ tokenize and parse
  ▶ resolve names and parameters
  ▶ elaborate component instances
  ▶ validate widths, drivers, clocks, and combinational loops
  ▶ attach source and timing information
  ▶ emit Machine IR
  ▶ instantiate simulation state and initial events
```

The first HDL should remain deliberately smaller than Verilog. It needs
components, ports, wires, constants, arrays, and explicit behavioral macros.
Full Verilog/SystemVerilog compatibility is a later bridge, not a prerequisite
for learning or for the first complete machine.

### CUDA to runnable instructions

```text
CUDA subset or server-produced PTX
  ▶ parse supported PTX
  ▶ validate types and address spaces
  ▶ lower virtual operations
  ▶ allocate toy registers
  ▶ encode Program IR instructions
  ▶ load instruction and constant memories
  ▶ launch grid, blocks, warps, and initial events
```

The teaching ISA may resemble PTX or SASS, but it must be documented as its own
ISA. It should not claim binary or behavioral compatibility with an NVIDIA GPU.

## Suggested internal modules

Do not create these files until an executable checkpoint needs them. They show
ownership boundaries, not mandatory scaffolding.

```text
src/app/nano-gpu/
├── page.tsx
├── readme.md
├── _core/
│   ├── gpu.ts                       current functional model
│   ├── machine_ir.ts                immutable hardware description
│   ├── program_ir.ts                executable teaching ISA
│   ├── logic.ts                     four-state values and buses
│   ├── event_queue.ts               deterministic time ordering
│   ├── timing_simulator.ts          event and delta-cycle engine
│   ├── functional_simulator.ts      whole-program fast path
│   ├── hdl_compiler.ts              HDL to Machine IR
│   ├── ptx_compiler.ts              PTX subset to Program IR
│   └── trace.ts                     deltas, checkpoints, causality
├── _components/
│   ├── gpu_lab.tsx
│   ├── machine_view.tsx
│   ├── waveform_view.tsx
│   ├── timeline.tsx
│   └── controls.tsx
└── _workers/
    ├── gpu_worker.ts
    └── worker_protocol.ts
```

Start by keeping related code together. Split a module only after it has a
second clear responsibility or becomes difficult to understand.

## Smallest complete implementation

The first physical slice should not be the entire GPU. It should prove the
simulation semantics end to end:

```text
HDL: two input pins → NAND gate → delayed wire → output register
clock: 1000 ps period
program/input: drive A and B
engine: event queue + delta cycles
trace: departure, arrival, gate result, register capture
viewer: schematic + pulse animation + waveform
```

Acceptance test:

```text
given A=1 and B=1 at 0 ps
and input wires each delay 10 ps
and NAND delay is 20 ps
and output wire delay is 10 ps
then register D becomes 0 at 40 ps
and the next clock edge captures 0
and the trace explains every causal transition
```

The second slice replaces the NAND with a one-bit ALU. The third builds an
eight-lane execution unit. Only then should the engine grow toward warps,
memories, and the whole GPU.

## Implementation sequence

### Phase 1 — Event kernel

- integer-picosecond clock;
- deterministic priority queue;
- batched simultaneous events;
- delta-cycle settling and loop detection;
- reversible state deltas;
- one NAND end-to-end fixture.

**Gate:** the expected waveform and causal trace are exact and deterministic.

### Phase 2 — Sequential circuits

- clocks and resets;
- flip-flops and registers;
- simultaneous sample/commit;
- clock-to-output delay;
- optional setup and hold diagnostics.

**Gate:** counters and small RAM behave correctly across multiple clock edges.

### Phase 3 — Hierarchical components

- behavioral and structural implementations;
- stable port contracts;
- selective collapse and expansion;
- hierarchical schematic and trace filtering.

**Gate:** a behavioral ALU and gate-level ALU agree at their output ports.

### Phase 4 — GPU cycle model

- warp scheduler and scoreboard;
- execution pipelines;
- register-file and shared-memory banks;
- caches, queues, links, and backpressure;
- multiple clock domains.

**Gate:** two correct kernels show different, causally explainable cycle counts.

### Phase 5 — Compiler connection

- tiny HDL compilation;
- toy assembly and Program IR;
- supported PTX lowering;
- source maps from program and hardware source into trace events.

**Gate:** one source line can be followed into an instruction, pipeline event,
register write, wire transition, and output.

### Phase 6 — Transformer-scale workflow

- functional fast path;
- architectural differential tests;
- detailed replay windows;
- trace budgets and breakpoints;
- tiny transformer inference.

**Gate:** the complete transformer runs quickly, and a selected attention
window can be replayed at cycle and gate-level detail.

## Required invariants

The architecture is correct only if these remain true:

1. Simulation results never depend on browser frame rate.
2. Simultaneous state changes are order-independent.
3. Every visible diagram object comes from Machine IR.
4. Every visible transition comes from simulation state or trace.
5. Functional and timing modes agree at architectural commit points.
6. Unsupported HDL, PTX, and timing behavior fails explicitly.
7. The same inputs and seed produce the same trace.
8. An infinite kernel, combinational loop, or enormous trace cannot freeze the
   browser UI.
9. Fidelity can change per component without changing its external contract.
10. A learner can follow causes both downward to wires and upward to kernels.

## Explicit non-goals

- Simulating every transistor of the full GPU.
- Treating a browser animation frame as a hardware clock.
- Updating every component every picosecond.
- Claiming cycle accuracy for an NVIDIA architecture without validation data.
- Implementing complete Verilog, CUDA, PTX, or SASS before the tiny ISA works.
- Flattening the whole GPU into a gate netlist merely because it is possible.
- Keeping exhaustive traces when filtered replay answers the question.
- Building separate diagrams and simulators that can disagree.

## Decision summary

```text
smallest stored time       1 ps
smallest causal step       one delta cycle at zero elapsed time
default interactive step   one selected clock edge
simulation algorithm       deterministic discrete-event queue
whole-chip physical model  digital timing with behavioral macros
deep physical model        selectively expanded gates and wires
analog model               isolated transistor/RC laboratories
render loop                independent from simulation time
scale strategy             functional run + selective detailed replay
```

This gives NanoGPU the feel of a physical game engine while preserving the
semantics needed for a compiler, a GPU, and eventually a transformer. The user
can watch a pulse move down a wire, but the machine remains fast enough to run
programs that matter.

## References

- [Verilator timing support](https://verilator.org/guide/latest/languages.html)
  describes delayed statements, event controls, and net delays.
- [Verilator model evaluation](https://verilator.org/guide/latest/connecting.html)
  demonstrates advancing a model to its next pending time slot.
- [OpenSTA](https://openroad.readthedocs.io/en/latest/main/src/sta/README.html)
  documents timing analysis using netlists, Liberty cells, SDF delays, SPEF
  parasitics, and clock constraints.
- [GPGPU-Sim](https://gpgpu-sim.org/manual/index.php/Main_Page) provides a useful
  reference for GPU clock domains, interconnects, pipelines, caches, memory
  partitions, and performance simulation.
