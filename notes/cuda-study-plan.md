# The GPU Kernel Campaign

> **Position in the larger curriculum:** This document is the advanced CUDA, attention, runtime and low-precision specialization track. The cumulative constructionist path from logic gates through a toy GPU, compiler, ML stack and distributed transformer is defined in [`nand-to-transformers.md`](./nand-to-transformers.md).

**Target:** a 52-week main campaign plus an 8-week endgame raid, at roughly 12–15 focused hours per week. That is about 700–900 hours from basic Python-level knowledge to meaningful work on FlashAttention-4, persistent megakernels, SGLang internals, and low-precision RL kernels. At 25 hours per week, compress the calendar, but **do not skip the unlock tests**.

The dependency tree is:

```text
Tensor math, floating point, autograd
│
├── GPU execution and data movement
│   └── CUDA → Triton → Tensor Core GEMM → CuTe layouts
│       └── online softmax → FA1/2 → FA3 → FA4
│
├── Transformer operators
│   └── KV cache → batching/scheduling → SGLang
│       └── persistent execution → megakernels
│
└── Probability and policy gradients
    └── PPO/GRPO → rollout/trainer systems
        └── FP8/FP4 kernels + fused log-prob/loss kernels
            └── low-precision, rollout-aligned RL
```

As of July 2026, the official FlashAttention-4 implementation is written in CuTe DSL and targets Hopper and Blackwell GPUs. The paper’s distinct Blackwell techniques include fully asynchronous MMA, Tensor Memory, large tiles, software exponential, conditional softmax rescaling, and 2-CTA MMA/DSMEM cooperation. ([GitHub][1])

A **megakernel** here means more than fusing neighboring operators. It means a persistent GPU program with device-resident task scheduling, queues, events or semaphores, and cross-operator execution. MPK, for example, compiles a tensor program into one launch and uses GPU-side worker and scheduler machinery. ([arXiv][2])

For low-precision RL, the endgame is not “cast rollout weights to FP8.” Current work shows that a low-precision rollout policy and higher-precision learner can represent meaningfully different policies, especially over long trajectories. Your kernels and system must therefore define and verify a **rollout–learner numerical contract**. ([arXiv][3])

---

## Game rules

You develop five stats:

| Stat            | What it measures                                                                |
| --------------- | ------------------------------------------------------------------------------- |
| **Correctness** | Outputs, gradients, masking, edge cases, determinism                            |
| **Memory**      | Bytes moved, layouts, coalescing, bank conflicts, materialized intermediates    |
| **Pipeline**    | Tensor Core utilization, overlap, barriers, register/SMEM/TMEM pressure         |
| **Systems**     | Launches, scheduling, KV-cache behavior, batching, communication                |
| **Numerics**    | Accumulation precision, scaling, exponentials, quantization and policy mismatch |

Every project has four ranks:

| Rank       | Unlock condition                                                                                |
| ---------- | ----------------------------------------------------------------------------------------------- |
| **Bronze** | Correct forward result against a trusted reference                                              |
| **Silver** | Backward or full inference path, adversarial tests, broad shape coverage                        |
| **Gold**   | You predict the bottleneck, profile it, and improve it for a stated workload                    |
| **Mythic** | The kernel is integrated into a real serving or training loop and improves an end-to-end metric |

Suggested XP ledger:

* 1 XP: derive an algorithm or cost estimate without looking it up.
* 3 XP: add a meaningful correctness or numerical test.
* 5 XP: make and verify a profiler-based hypothesis.
* 10 XP: produce a robust optimized kernel.
* 25 XP: complete an integrated boss project.

Passive reading earns no XP until it produces an annotated implementation, derivation, benchmark, or profiler report.

---

# The three first-principles spells

## 1. The performance bound

Before writing any kernel, estimate:

[
T \gtrsim \max\left(
\frac{\text{bytes transferred}}{\text{effective bandwidth}},
\frac{\text{operations}}{\text{effective compute throughput}},
\text{exposed dependency latency}
\right)

* T_{\text{launch/synchronization}}
  ]

Then ask:

1. Which memory level supplies every operand?
2. How many times is each value loaded?
3. Which units execute the math?
4. What work can overlap?
5. Which synchronization edge lies on the critical path?

Nsight Compute’s roofline combines arithmetic intensity, bandwidth, and compute ceilings, making it the primary validation tool for this reasoning. ([NVIDIA Docs][4])

## 2. Online softmax

FlashAttention becomes understandable once you can derive this recurrence from memory. For a new score block (x), with value block (V), maintain maximum (m), normalizer (l), and unnormalized output accumulator (O):

[
m' = \max(m, \max x)
]

[
l' = e^{m-m'}l + \sum_i e^{x_i-m'}
]

[
O' = e^{m-m'}O + \sum_i e^{x_i-m'}V_i
]

The final output is (O/l).

This lets you stream score tiles without materializing the complete (QK^\top) matrix. The advanced FlashAttention kernels are increasingly sophisticated ways of mapping this recurrence, its matrix multiplications, and its backward pass onto evolving hardware.

## 3. The RL policy contract

For generated token (a_t) in state (s_t), the learner uses ratios such as

[
r_t =
\exp\left(
\log \pi_{\theta}(a_t\mid s_t)
------------------------------

\log \pi_{\text{behavior}}(a_t\mid s_t)
\right)
]

If rollout actually uses a quantized policy (\pi_q), but your logged “old log-probability” came from a different BF16 computation, the ratio is not describing the process that generated the data.

Your low-precision RL work must explicitly choose one of these contracts:

* bitwise-consistent rollout and learner forward paths;
* matching low-precision forward arithmetic with high-precision master weights;
* measured mismatch corrected through importance weighting or trust-region machinery.

Never leave the contract implicit.

---

# Act I — Build the GPU mental model

## Level 0: Boot sequence — Weeks 1–2

**Master**

* Python and PyTorch tensor operations.
* Basic C++: pointers, memory ownership, templates, compilation and linking.
* Matrix multiplication, softmax, normalization, chain rule and vector–Jacobian products.
* Tensor shape, stride, storage offset and broadcasting.
* FP32, BF16 and FP16 structure; overflow, underflow and accumulation error.

**Drills**

* Implement stable softmax in NumPy and PyTorch.
* Implement matrix multiplication using explicit loops, then blocked loops.
* Write a single-head causal attention reference.
* Derive and implement softmax backward.
* Use finite differences to check attention gradients.
* Inspect contiguous, transposed and sliced tensor strides.

**Boss: The Reference Transformer**

Build a small decoder block containing RMSNorm, causal attention, RoPE and SwiGLU. It should have:

* deterministic test inputs;
* forward and gradient tests;
* shape annotations at every transformation;
* a slow, readable implementation that becomes the oracle for future kernels.

**Unlock:** You can derive stable softmax and attention tensor dimensions on paper without consulting code.

---

## Level 1: GPU physics — Weeks 3–6

**Master**

* Grid, block, warp, lane and SM.
* Registers, shared memory, L1/L2 and global memory.
* Coalescing and memory transactions.
* Shared-memory bank conflicts.
* Warp divergence.
* Occupancy versus useful concurrency.
* Latency hiding and instruction-level parallelism.
* Arithmetic intensity and roofline reasoning.

**Build**

1. Vector addition.
2. SAXPY.
3. Row reduction.
4. Column reduction.
5. Matrix transpose.
6. Histogram or scatter with atomics.

For every kernel, first write down:

```text
Useful FLOPs:
Global bytes:
Expected arithmetic intensity:
Expected bottleneck:
Expected access pattern per warp:
```

**Boss: The Prediction Trial**

Choose reduction and transpose kernels. Predict their limiting resources before profiling, then produce an Nsight Compute report explaining:

* global load/store efficiency;
* shared-memory behavior;
* occupancy and register use;
* achieved bandwidth;
* why observed performance differs from the simple bound.

**Unlock:** Your predicted bottleneck is usually correct, even when your predicted runtime is not.

---

## Level 2: CUDA dojo — Weeks 7–10

**Master**

* Shared-memory tiling.
* Warp shuffles and cooperative reductions.
* Vectorized loads.
* Predication and bounds handling.
* Streams, events and asynchronous execution.
* Kernel launch overhead.
* Custom PyTorch CUDA extensions.
* Forward/backward kernel pairs.

**Build**

* Fused softmax.
* LayerNorm or RMSNorm.
* Tiled non-Tensor-Core GEMM.
* Fused bias plus activation.
* A numerically stable reduction with FP32 accumulation.
* A custom PyTorch operation with autograd support.

**Boss: Shape-Shifter**

Your RMSNorm or softmax must pass a test grid containing:

* non-power-of-two dimensions;
* very small and very large rows;
* non-contiguous inputs where supported;
* BF16, FP16 and FP32;
* extreme-magnitude inputs;
* forward and backward comparisons.

Then produce separate optimized configurations for short and long rows.

**Unlock:** You no longer equate high occupancy with high performance.

---

## Level 3: Triton spellbook — Weeks 11–14

Triton’s official progression already forms a useful early game: vector addition, fused softmax, matrix multiplication, dropout, normalization, fused attention, persistent matmul, and block-scaled matmul. ([Triton Language][5])

**Master**

* Program instances rather than individual CUDA threads.
* Block pointers, masks and tensorized indexing.
* `tl.load`, `tl.store`, `tl.dot` and reductions.
* Autotuning over tile size, warp count and pipeline stages.
* Triton IR inspection and debugging.
* Custom backward functions.

**Build**

* Fused softmax.
* Matrix multiplication.
* RMSNorm.
* RoPE.
* SwiGLU.
* Residual plus normalization.
* At least five KernelBench Level 1 tasks and three Level 2 fusion tasks. KernelBench provides single-operator, fusion and full-model workloads suitable for this ladder. ([GitHub][6])

**Boss: The Kernel Tournament**

Choose one bandwidth-bound and one compute-bound operator. For each:

* define a realistic shape distribution;
* compare PyTorch eager, `torch.compile`, Triton and an available optimized library;
* plot latency across the shape grid;
* explain every major regime change;
* preserve a readable baseline beside the optimized version.

**Unlock:** You can decide whether Triton is sufficient before reaching for CUDA or CuTe.

---

# Act II — Tensor Cores and the FlashAttention ladder

## Level 4: Tensor Core forge and CuTe layouts — Weeks 15–19

CuTe DSL is a Python DSL intended to preserve low-level hardware control, with JIT compilation and interoperability with frameworks such as PyTorch. ([NVIDIA Docs][7])

**Master**

* MMA instruction shapes and accumulation types.
* Thread, warp and warpgroup decomposition.
* Layout algebra: shape, stride, composition, tiling and partitioning.
* Copy atoms and MMA atoms.
* Shared-memory swizzles.
* Double and multi-stage buffering.
* TMA transfers.
* WGMMA on Hopper and TCGEN05/TMEM concepts on Blackwell.
* Register and shared-memory resource models.

**Drill deck**

Given a layout, repeatedly answer:

1. Which lane owns element ((i,j))?
2. Which address does the lane load?
3. Are adjacent lanes coalesced?
4. Which shared-memory bank receives each access?
5. How is the tile partitioned over the MMA instruction?
6. Where does the accumulator reside?

Do these on paper before examining generated code.

**Build**

* A Tensor Core GEMM in Triton.
* The same conceptual GEMM in CuTe DSL.
* Bias, activation and residual epilogues.
* A persistent GEMM with an atomic tile counter.
* An autotuning harness.

**Boss: GEMM Blacksmith**

Produce a GEMM report for at least twelve shapes covering:

* square and rectangular matrices;
* small and large (M);
* large and small (K);
* contiguous and transposed operands;
* BF16 and one low-precision path.

The report must connect tile size, registers, shared memory, pipeline stages and achieved throughput.

**Unlock:** You can trace a matrix element from global memory through shared memory and Tensor Cores into its accumulator.

---

## Level 5: Transformer kernel arena — Weeks 20–22

**Master**

* Fusion boundaries.
* Recompute versus save-for-backward.
* Reduction placement.
* Quantization and dequantization fusion.
* The enormous memory cost of materializing vocabulary-sized logits.
* Selective token log-probability computation.

**Build**

* RMSNorm plus residual.
* RoPE applied during Q/K movement.
* SwiGLU or gated MLP.
* Fused bias/activation epilogues.
* Fused linear plus cross-entropy.
* Fused linear plus selected-token log-probabilities.

Liger Kernel is a strong code-reading target here: it contains Triton implementations of RMSNorm, RoPE, SwiGLU, cross-entropy and fused linear losses. Its fused linear cross-entropy avoids materializing the complete logits tensor. ([GitHub][8])

**Boss: The Vanishing Logits**

Given hidden states of shape ([B,T,H]), LM-head weights ([V,H]), and selected token IDs:

* compute selected log-probabilities;
* compute the needed loss;
* produce input and weight gradients;
* never retain a ([B,T,V]) logits tensor;
* compare memory and latency with the materializing implementation.

**Unlock:** You naturally search for the minimal sufficient intermediate, rather than fusing operations indiscriminately.

---

## Level 6: Attention labyrinth — Weeks 23–27

**Master in this order**

1. Naive attention.
2. Tiled (QK^\top).
3. Stable block softmax.
4. Online softmax.
5. Fused probability–value accumulation.
6. Causal masking.
7. Variable sequence lengths.
8. Multi-query and grouped-query attention.
9. Forward recomputation choices.
10. Backward derivation and accumulation.

**Build ladder**

* PyTorch reference attention.
* CUDA or Triton tiled attention that still writes scores.
* Streaming forward attention using online softmax.
* FA1-style IO-aware forward.
* FA2-style work partitioning.
* Backward for one constrained configuration.
* Generalized causal and variable-length handling.

Do not initially read a full production kernel linearly. First write your own simplified version with:

* fixed head dimension;
* fixed causal mode;
* no dropout;
* one datatype;
* one tile configuration.

Then add one dimension of generality at a time.

**Boss: FlashAttention-2 Clone**

The kernel must handle a declared subset such as:

```text
BF16
head_dim ∈ {64, 128}
causal and non-causal
fixed and variable sequence lengths
MHA and GQA
forward and backward
```

You must explain:

* HBM traffic avoided;
* shared-memory lifetime of Q, K and V;
* accumulator layout;
* online-softmax rescaling;
* parallelism over batch, heads and sequence tiles;
* backward atomics or reduction strategy.

**Unlock:** You can reconstruct online attention from the recurrence rather than memorizing implementation code.

---

## Level 7: Hopper citadel and FA3 — Weeks 28–30

FlashAttention-3 targets Hopper and uses asynchronous execution and warp specialization; the official release includes BF16/FP16 forward and backward and an FP8 forward path. ([GitHub][1])

**Master**

* TMA producer/consumer movement.
* WGMMA.
* Warpgroup specialization.
* Transaction barriers.
* Ping-pong scheduling.
* Overlap of softmax and matrix multiplication.
* Pipeline startup, steady state and drain.
* Why a theoretically independent operation may still serialize due to resource pressure.

**Build**

* An asynchronously staged GEMM.
* A producer–consumer attention microkernel.
* An annotated FA3 timeline.
* One modification to an FA3 tile, stage count or work partition.

**Boss: The Overlap Proof**

A profiler timeline must visibly show the overlap you intended. Your write-up should state:

* which warpgroup produces data;
* which computes MMA;
* which performs softmax or correction;
* which barrier protects each buffer;
* which dependency remains exposed.

**Unlock:** You reason in pipeline timelines, not merely source-code order.

---

## Level 8: Blackwell forge and FlashAttention-4 — Weeks 31–35

The central FA4 lesson is that hardware does not scale uniformly. On Blackwell, faster Tensor Cores make shared-memory traffic, exponentials and non-MMA work relatively more important. FA4 therefore redesigns both the algorithm and pipeline rather than simply changing MMA instructions. ([arXiv][9])

**Master**

* Fully asynchronous Blackwell MMA.
* TMEM accumulator storage.
* Larger (128\times128)-class tile strategies.
* TCGEN05-style operation.
* Software-emulated exponential.
* Conditional softmax rescaling.
* 2-CTA MMA cooperation.
* CTA clusters and DSMEM.
* Reordered backward pipelines.
* Global atomic reduction costs.
* Persistent and dynamic tile scheduling.

**Required progression**

### Stage A: Archaeology

Pin a known-good FA4 repository commit and annotate:

```text
Q load path
K/V load path
QK MMA
row-max computation
exponential path
normalizer update
PV MMA
output store
backward dQ/dK/dV paths
tile scheduler
```

### Stage B: Controlled mutation

Modify exactly one dimension:

* alternative causal or local mask;
* new supported head dimension;
* tile shape or scheduler;
* fused output transform;
* exponential approximation;
* 1-CTA versus 2-CTA path;
* deterministic accumulation strategy.

### Stage C: Reconstruction

Write a minimal CuTe DSL forward kernel using the same ideas, but with fewer supported cases. Only then broaden it.

### FA4 boss acceptance criteria

Your project passes only when it contains:

1. **Correctness:** differential forward and backward tests.
2. **Robustness:** causal, odd lengths, multiple head dimensions and stress inputs.
3. **Performance:** shape-grid comparison against at least two serious baselines.
4. **Profiler evidence:** the claimed bottleneck and its movement after your change.
5. **Resource accounting:** registers, shared memory, TMEM and CTA occupancy.
6. **Explanation:** why the change helps Blackwell specifically.

A paper-faithful TMEM and 2-CTA exercise requires Blackwell hardware. The official FA4 package also has Hopper support, so much of the source study and CuTe work can begin on H100 before reserving B200 time. ([GitHub][1])

**Unlock:** You can explain why “more Tensor Core FLOPs” can make exponentials or shared-memory movement the dominant attention bottleneck.

---

# Act III — Serving systems and megakernels

## Level 9: SGLang city — Weeks 36–39

Start with Mini-SGLang, not the full production repository. It is explicitly designed as a compact, roughly 5,000-line reference implementation of a modern serving engine. ([GitHub][10])

**Master**

* Prefill versus decode.
* Continuous batching.
* KV-cache allocation and paging.
* Prefix and radix caching.
* Chunked prefill.
* Request admission and eviction.
* Sampling kernels.
* CUDA graphs.
* Tensor parallelism.
* Scheduler/model-runner boundaries.
* TTFT, time per output token, throughput and tail latency.

A useful concrete boundary in SGLang is `ScheduleBatch -> ForwardBatch`: high-level scheduling data is largely CPU-side, while the model runner consumes GPU tensor metadata. ([GitHub][11])

**Quests**

1. Trace one request from HTTP input through tokenization, scheduling, model execution, sampling and response streaming.
2. Print every KV-cache allocation and release for a short workload.
3. Separate prefill-only and decode-only benchmarks.
4. Run a mixed arrival trace and inspect batching decisions.
5. Compare an empty-prefix request with a reused-prefix request.
6. Instrument kernel launch count per decode step.
7. Trace CUDA graph capture and replay.

**Build**

Choose one:

* custom Triton attention backend;
* custom FA4 backend integration;
* fused RoPE plus cache write;
* fused sampling/top-k operator;
* quantized KV-cache operator;
* specialized decode attention path.

The SGLang source describes separate extend/prefill-with-prefix and decode operations, and notes that its Triton path is easier to customize than its highly optimized alternative. ([GitHub][12])

**Boss: Backend Invasion**

Integrate your kernel into Mini-SGLang first, then full SGLang. Report:

* output equivalence;
* TTFT;
* per-token latency;
* tokens per second;
* p50 and p99 latency;
* GPU memory;
* launch count;
* prefill-heavy, decode-heavy and mixed workloads.

**Unlock:** You can point to the exact scheduler decision, cache layout and kernel invocation responsible for a serving latency change.

---

## Level 10: Megakernel dungeon — Weeks 40–43

A practical megakernel progression is:

```text
persistent homogeneous work
        ↓
atomic tile queue
        ↓
heterogeneous task queue
        ↓
events and dependency graph
        ↓
multiple transformer operators
        ↓
dynamic device-side scheduling
```

**Master**

* Persistent grids.
* Global work counters.
* Lock-free or low-contention queues.
* Device-side events and semaphores.
* Worker and scheduler specialization.
* Grid/cluster synchronization.
* Resource partitioning between heterogeneous tasks.
* Cross-operator software pipelining.
* Dynamic shapes and variable workloads.
* Deadlock and starvation analysis.
* Launch-overhead versus scheduler-overhead trade-offs.

ThunderKittens is useful for studying tiled Tensor Core operations, asynchronous WGMMA/TCGEN05, TMA, DSMEM and worker overlap. The HazyResearch megakernel repository includes a low-latency Llama demonstration for H100 and B200. ([GitHub][13])

**Build ladder**

1. Persistent matrix multiplication with an atomic tile counter.
2. Persistent grouped GEMM with variable task sizes.
3. A queue containing GEMM and elementwise tasks.
4. Event dependencies such as:

```text
RMSNorm
   ↓
QKV projection
   ↓
RoPE / KV write
   ↓
attention
   ↓
output projection ──┐
                    ├─ residual
MLP path ───────────┘
```

5. One persistent decoder layer.
6. Optionally, multiple decoder layers or a device-resident token loop.

**Boss: The Tiny Persistent Decoder**

Build a small one- or two-layer decoder whose work is executed by one persistent launch or a very small fixed launch set.

Measure:

* launches per generated token;
* device-queue overhead;
* scheduler SM utilization;
* worker imbalance;
* per-token latency;
* p99 latency;
* correctness over varying batch sizes;
* deadlock behavior under empty, overloaded and irregular queues.

The boss is failed if launch count falls but total latency rises without a clear reason.

**Unlock:** You understand a megakernel as an on-device runtime, not as a very large fused expression.

---

# Act IV — RL systems and low-precision training kernels

## Level 11: RL systems campaign — Weeks 44–47

GRPO was introduced as a PPO variant that uses group-relative rewards and avoids a learned value-function critic. ([arXiv][14])

**Master**

* REINFORCE.
* Baselines and advantage estimation.
* PPO ratios and clipping.
* GRPO group normalization.
* Reference-policy KL.
* Token-level versus sequence-level objectives.
* Behavior, old, current and reference policies.
* Rollout generation.
* Reward and verifier execution.
* Learner update.
* Weight synchronization.
* Staleness and asynchronous pipelines.
* Data, tensor and pipeline parallel placement.

**Do not begin with a full distributed framework.**

Build this ladder:

1. Multi-armed bandit policy gradient.
2. Tiny autoregressive model with REINFORCE.
3. PPO on a toy sequence reward.
4. GRPO with a group of sampled completions.
5. Separate rollout and learner processes.
6. Add explicit weight-version IDs.
7. Replace the rollout process with SGLang.

Then study one complete framework. `verl` exposes PPO/GRPO-style dataflows and integrations with training and serving engines; `slime` directly connects Megatron training and SGLang rollout. ([GitHub][15])

**Boss: On-Policy Inspector**

For every generated completion, store:

```text
prompt ID
sampled tokens
rollout weight version
rollout precision configuration
rollout log-probabilities
learner weight version
recomputed log-probabilities
reward
advantages
importance ratios
clip mask
KL contribution
```

Your dashboard must expose:

* policy-version staleness;
* log-probability mismatch;
* ratio tails;
* clipping fraction;
* entropy;
* response length;
* reward and KL by length bucket.

**Unlock:** You can tell whether a training failure is algorithmic, stale-policy, numerical, kernel, or scheduling related.

---

## Level 12: Low-quant RL raid preparation — Weeks 48–52

Transformer Engine supports FP8 on Hopper, Ada and Blackwell, and MXFP8 and NVFP4 on Blackwell. NVFP4 adds block and global scaling rather than being a simple four-bit cast. ([NVIDIA Docs][16])

### Skill tree A: Low-precision compute

**Master**

* FP8 E4M3 and E5M2.
* Dynamic range and saturation.
* Per-tensor, per-row and block scaling.
* Current versus delayed scaling.
* `amax` collection and reduction.
* Scale and inverse-scale layout.
* Quantize/dequantize fusion.
* FProp, DGrad and WGrad.
* High-precision master weights.
* Stochastic rounding.
* Transposed operands and scale swizzling.
* MXFP8 and NVFP4 on Blackwell.
* Which reductions must remain FP32.

**Build**

* FP8 quantize/dequantize.
* Row-scaled FP8 GEMM.
* Block-scaled GEMM.
* FP8 linear forward and backward.
* Fused quantization epilogue.
* Scale-history and saturation telemetry.
* Optional NVFP4 GEMM or Transformer Engine extension on B200.

### Skill tree B: RL loss kernels

**Master**

* Selected-token log-softmax.
* Old/current/reference log-probabilities.
* KL estimators.
* Importance ratios.
* Asymmetric clipping.
* Token masks.
* Per-sequence and per-token normalization.
* Chunked vocabulary processing.
* Gradient accumulation without logits materialization.

**Build**

* Fused LM head plus selected-token log-probability.
* Fused selected-token log-probability plus KL.
* Fused linear GRPO/PPO loss.
* Chunked weight-gradient accumulation.
* Mixed-precision loss and reduction path.

Liger’s current fused linear GRPO implementation is a valuable reference: its interface includes selected token IDs, masks, advantages, old/reference log-probabilities, KL weight and clipping parameters. ([GitHub][17])

### Skill tree C: Rollout–learner consistency

Build a harness that runs the **same weights, prompt and forced token sequence** through:

1. rollout engine;
2. learner forward;
3. standalone reference model;
4. your custom low-precision kernel path.

Record per position:

* max logit error;
* selected-token log-probability error;
* KL divergence;
* top-1 and top-k disagreement;
* saturation counts;
* scale values;
* importance ratio;
* cumulative divergence versus sequence length.

Repeat across:

* batch sizes;
* tensor-parallel degrees;
* prompt lengths;
* generation lengths;
* easy and ambiguous token distributions;
* weight updates of different magnitudes.

**Boss: Quantized Policy Contract**

Complete a small but genuine RL run in which:

* rollout uses a low-precision path;
* learner forward is explicitly aligned or explicitly corrected;
* high-precision master weights are retained;
* your fused loss avoids materializing complete logits;
* reward and KL remain stable against a BF16 control;
* rollout throughput or total step time improves;
* every difference from the BF16 baseline is attributable.

**Unlock:** You can explain low-precision RL simultaneously at the Tensor Core, autograd, probability-distribution and distributed-systems levels.

---

# Endgame raid — Weeks 53–60

## Final project: SGLang × FA4 × low-precision RL

Build one integrated system with the following path:

```text
Prompts
  ↓
SGLang scheduler and KV cache
  ↓
custom FA4 or specialized attention backend
  ↓
low-precision rollout
  ↓
rewards / verifiers
  ↓
selected-token log-probability kernel
  ↓
fused GRPO loss
  ↓
low-precision-aligned learner forward
  ↓
weight quantization and publication
  └───────────────────────────────→ SGLang
```

An optional mythic extension replaces a portion of decode with a persistent megakernel.

## Raid phases

### Weeks 53–54: Establish the BF16 oracle

* Small model and dataset.
* Fully reproducible BF16 training.
* Complete timing breakdown.
* Weight-version and log-probability tracing.
* Evaluation and reward baselines.

### Weeks 55–56: Insert the custom serving kernel

* Add FA4 or another custom attention path to SGLang.
* Validate cache and attention outputs.
* Benchmark prefill, decode and mixed traffic.
* Verify generated tokens and forced-token log-probabilities.

### Weeks 57–58: Insert low-precision rollout and training kernels

* Add quantized weight publication.
* Add aligned learner forward.
* Add fused selected-token log-probability and GRPO loss.
* Measure mismatch by sequence length.

### Weeks 59–60: Integrate and defend

Produce:

1. architecture diagram;
2. kernel dataflow diagrams;
3. correctness matrix;
4. precision-flow graph;
5. NCU reports;
6. Nsight Systems timeline;
7. serving benchmark;
8. RL step-time breakdown;
9. learning curves;
10. failure analysis and limitations.

A valid final result does **not** have to beat every production library. It must demonstrate a defensible improvement for a precisely defined workload and show that you understand the remaining bottleneck.

---

# Permanent drill decks

## Layout drill — 15 minutes, three times per week

Take a tensor layout or tile and manually map:

```text
logical coordinate
→ thread/warp ownership
→ register fragment
→ shared-memory address
→ bank
→ global-memory transaction
```

This is the highest-return drill for CuTe and attention work.

## Roofline drill — one kernel per week

Before running the kernel:

* count useful operations;
* count unavoidable bytes at each memory level;
* predict memory- or compute-bound;
* estimate best-case runtime;
* identify the likely secondary bottleneck.

After profiling, write a five-sentence postmortem.

## Pipeline drill — twice per week

Draw a cycle-level conceptual timeline:

```text
TMA load tile 0
TMA load tile 1
MMA tile 0
softmax tile 0
MMA tile 1
rescale tile 0
store tile 0
```

Mark every barrier and buffer reuse. Then ask which operation could be moved without violating dependencies or resource limits.

## Numerical drill — weekly

Generate hostile inputs:

* all equal;
* one huge outlier;
* alternating large positive and negative;
* very long reduction;
* near-underflow probabilities;
* values at quantizer boundaries;
* zero-length or fully masked rows.

Compare output error, gradient error and downstream policy divergence.

## Code-reading drill — weekly

Never read a large repository passively. For each file:

1. state its role before reading;
2. draw its inputs and outputs;
3. locate allocation and synchronization sites;
4. identify the fast path;
5. change or instrument something;
6. write a one-page summary.

---

# Weekly operating rhythm

For a 12-hour week:

| Activity                      |      Time |
| ----------------------------- | --------: |
| Theory and derivation         |   2 hours |
| Implementation                |   5 hours |
| Testing and adversarial cases | 1.5 hours |
| Profiling and benchmarking    |   2 hours |
| Code reading                  |    1 hour |
| Written postmortem            |  0.5 hour |

Each week should end with exactly four artifacts:

```text
one derivation
one working implementation
one benchmark or profiler capture
one written conclusion
```

Every fourth week is a boss week: reduce new reading and finish an integrated, defensible artifact.

---

# Benchmarking laws

1. **Never benchmark only one shape.** Use a distribution reflecting the target workload.
2. **Never report speedup without naming the baseline.**
3. **Never compare against an intentionally poor baseline alone.**
4. **Warm up, synchronize and report a latency distribution.**
5. **Record GPU, driver, CUDA, PyTorch, compiler, repository commit and kernel configuration.**
6. **Separate operator time from end-to-end time.**
7. **Test backward and numerical behavior when training is the goal.**
8. **Profile representative shapes, not only the fastest shape.**
9. **A fusion that increases register pressure or destroys occupancy may lose.**
10. **A quantized kernel that changes the sampled policy is not “correct” merely because its output tensors are close.**

---

# Hardware ladder

| Campaign stage  | Practical hardware                                                         |
| --------------- | -------------------------------------------------------------------------- |
| Levels 0–6      | A modern CUDA GPU; Ampere-class or newer is a convenient baseline          |
| Level 7         | H100/H200 for Hopper TMA/WGMMA and FA3 work                                |
| Level 8         | B200 for the Blackwell-specific FA4 TMEM, TCGEN05 and 2-CTA work           |
| Levels 9–11     | Whatever can host the chosen small model; multi-GPU later                  |
| Level 12        | H100-class hardware for serious FP8 work; B200 for MXFP8/NVFP4 experiments |
| Megakernel raid | H100 or B200, matching the reference projects                              |

Write and unit-test architecture-independent pieces locally. Reserve expensive architecture-specific access for profiler sessions and boss benchmarks. Pin known-good containers and commits; these repositories and compiler stacks move quickly.

---

# Your first 14 days

| Days      | Mission                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------- |
| **1–3**   | Set up a reproducible repository, benchmark harness and PyTorch transformer reference. Study shapes, strides and storage. |
| **4–5**   | Derive stable softmax and online softmax. Implement both and gradient-check them.                                         |
| **6–8**   | Write CUDA vector addition, reduction and transpose. Predict bytes and bottlenecks first.                                 |
| **9–10**  | Profile all three with Nsight Compute. Produce your first roofline and memory-access postmortem.                          |
| **11–12** | Implement Triton vector addition and fused softmax from the official progression.                                         |
| **13**    | Benchmark multiple row lengths and explain configuration transitions.                                                     |
| **14**    | Boss: improve one of your own naive kernels, prove correctness, and defend the speedup with profiler evidence.            |

The first boss is deliberately small. The habit it establishes—**derive, implement, falsify, profile, explain**—is the same loop you will use when the kernel has become FlashAttention-4 or a device-resident RL megakernel.

[1]: https://github.com/dao-ailab/flash-attention "GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention · GitHub"
[2]: https://arxiv.org/html/2512.22219v2 "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs"
[3]: https://arxiv.org/html/2601.14243v1 "Jet-RL: Enabling On-Policy FP8 Reinforcement Learning with Unified Training and Rollout Precision Flow"
[4]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "2. Profiling Guide — NsightCompute 13.3 documentation"
[5]: https://triton-lang.org/main/getting-started/tutorials/ "Tutorials — Triton  documentation"
[6]: https://github.com/ScalingIntelligence/KernelBench "GitHub - ScalingIntelligence/KernelBench: KernelBench: Can LLMs Write GPU Kernels? - Benchmark + Toolkit with Torch -> CUDA (+ more DSLs) · GitHub"
[7]: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html "Introduction — NVIDIA CUTLASS Documentation"
[8]: https://github.com/linkedin/Liger-Kernel "GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training · GitHub"
[9]: https://arxiv.org/html/2603.05451v1 "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"
[10]: https://github.com/sgl-project/mini-sglang "GitHub - sgl-project/mini-sglang: A compact implementation of SGLang, designed to demystify the complexities of modern LLM serving systems. · GitHub"
[11]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py "sglang/python/sglang/srt/model_executor/forward_batch_info.py at main · sgl-project/sglang · GitHub"
[12]: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py "sglang/python/sglang/srt/layers/attention/flashinfer_backend.py at main · sgl-project/sglang · GitHub"
[13]: https://github.com/HazyResearch/ThunderKittens "GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels · GitHub"
[14]: https://arxiv.org/abs/2402.03300 "[2402.03300] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
[15]: https://github.com/verl-project/verl "GitHub - verl-project/verl: verl/HybridFlow: A Flexible and Efficient RL Post-Training Framework · GitHub"
[16]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/ "Transformer Engine documentation — Transformer Engine 2.16.0 documentation"
[17]: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/grpo_loss.py "Liger-Kernel/src/liger_kernel/chunked_loss/grpo_loss.py at main · linkedin/Liger-Kernel · GitHub"
