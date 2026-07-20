# Lab 6: Matrix Multiply – Tensor Cores

## Prologue: Logistics

### Due Dates

For this lab, you’ll be turning in the following deliverables:

- **Checkpoint:** Due Tuesday, October 14, 11:59pm. For this checkpoint, submit your responses to the prelab (Part 0), and tell us how you’re doing in optimizing your implementation for the main lab.

- **Final Submission:** Due Friday, October 17, 11:59pm. Submit your completed code for `matmul_3.cu`, as well as a write-up containing your answers to Questions 1 - 2.


### Starter Code

You can get the starter code for this lab by cloning the [lab repository](https://github.com/accelerated-computing-class/lab6):

```
git clone git@github.com:accelerated-computing-class/lab6.git
```

## Introduction

### Goals for This Lab

So far in our exploration of matrix multiplication, we’ve focused primarily on optimizing **data movement** ( [Lab 4](https://accelerated-computing.academy/fall25/labs/lab4)) and **work partitioning** ( [Lab 5](https://accelerated-computing.academy/fall25/labs/lab5)). As we’ve worked to reduce bottlenecks along those dimensions, the run times of our implementations have increasingly become dominated by the cost of the **floating point computations** in our kernels’ innermost loops. Up until now, we’ve been implementing those core floating point computations using [**fused multiply-add (FMA)**](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma) instructions. However, we can do better: modern NVIDIA GPUs support so-called [“ **tensor core**”](https://www.nvidia.com/en-us/data-center/tensor-cores/) instructions, which are designed specifically to accelerate matrix multiplication workloads. In this third and final matrix multiplication lab, we’ll be looking at how to use those tensor core instructions to speed up our kernels.

The tensor core instructions we’ll be using in this lab aren’t exposed by default in the CUDA C++ language, so we’ll be accessing them via [**inline PTX assembly**](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html), as you’ve done before in this class.[1](https://accelerated-computing.academy/fall25/labs/lab6/#cutlass)

1. In the prelab, we’ll look at how to access **tensor core instructions** in PTX, and how to work with the **data layouts** those tensor core instructions expect.

2. Then, we’ll integrate tensor core instructions into our **full matrix multiplication kernel**, and try to obtain a speedup over what were able to achieve using FMAs.


### Note on Terminology: What is a “Tensor Core?”

Although the phrase “tensor core” might conjure up mental images of something similar to a “CPU core” – perhaps something with its own register file and program counter, decoding and executing a programmable stream of instructions in sequence – a tensor core is **not** actually that kind of “core” in the traditional computer architecture sense.

The phrase “tensor core” is just NVIDIA’s name for a particular kind of **functional unit** which exists on recent generations of NVIDIA GPUs. Tensor cores are not fundamentally different from ALUs or FPUs – each tensor core is attached to a warp scheduler, and solely executes math operations.

![](https://accelerated-computing.academy/fall25/labs/lab6/images/tensor_core_schematic.svg)

From a software point of view, tensor cores simply provide **another kind of math instruction** which your code is able to invoke. As we’ll see, these tensor core instructions have some interesting and unusual properties, but their existence doesn’t radically alter anything about the CUDA programming model.

## Part 0: Prelab – Invoking Tensor Core Instructions

First, let’s start by using our background in inline PTX to use it to interact with the tensor cores on our GPU!

The RTX 4000 Ada GPU we’re using belongs to NVIDIA’s [Ada Lovelace](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)) generation (specifically, “Compute Capability 8.9”). On Ada, there are tensor core instructions available in four different floating-point flavors:

- [**`f16`**](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) – The 16-bit floating point format [defined by the IEEE 754 standard](https://en.wikipedia.org/wiki/IEEE_754), with 5 exponent bits and 10 mantissa bits.

- [**`bf16`**](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) – “ [Brain](https://en.wikipedia.org/wiki/Google_Brain) float 16,” with 8 exponent bits and 7 mantissa bits. Popular in deep learning.

- [**`tf32`**](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) – “TensorFloat-32,” which is basically ordinary [32-bit floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format), but with compromises made in the accuracy of the multiplications performed by the tensor core. The mantissa of each input value is implicitly truncated to 10 bits (down from the ordinary 23) before participating in the multiplication.

- [**`e4m3`, `e5m2`**](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) – 8-bit floating point, which is configurable to either use 4 or 5 exponent bits (`e4m3` vs. `e5m2`), which was added in the Hopper/Ada generations.


Additionally, for all four of these formats, tensor cores support **accumulating** results in full 32-bit precision, effectively casting the results of the tensor core’s lower-precision multiplications up to FP32 before adding them together or to an existing partial sum.

Because the kernels we developed in Lab 4 and Lab 5 work in 32-bit precision, we’ll be focusing on **TF32** precision in this lab for the sake of compatibility with our existing code. Given that TF32 tensor core instructions perform multiplications in lower precision than the FP32 FMAs we’ve been using until now, we can expect to see some unavoidable accuracy loss when we adapt our kernel to use tensor cores.

So – what kind of TF32-precision tensor core functionality do we actually have on our 4000 Ada GPU? The answer is simple – we have exactly two instructions:[2](https://accelerated-computing.academy/fall25/labs/lab6/#wmma)

- **`mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`** (`HMMA.1684.F32.TF32` in SASS)
- **`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`** (`HMMA.1688.F32.TF32` in SASS)

(You can find the PTX documentation for these instructions here: [MMA PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).)

Both of these instructions are “ **matrix-multiply-accumulate**” ( **MMA**) instructions; conceptually, they each implement an operation like:

D←AB+CD \\leftarrow A B + CD←AB+C

where AAA, BBB, CCC, and DDD are matrices. The two instructions differ only in the dimensions of the matrices they operate on:

| Instruction Dimensions | AAA Dimensions | BBB Dimensions | CCC, DDD Dimensions |
| --- | --- | --- | --- |
| `m16n8k4` | `16 * 4` | `4 * 8` | `16 * 8` |
| `m16n8k8` | `16 * 8` | `8 * 8` | `16 * 8` |

Empirically, the course staff have observed that these instructions are equivalent in terms of FLOP throughput; the `m16n8k4` variant performs half as much work per instruction as `m16n8k8`, but twice as many `m16n8k4` instructions can execute per cycle on average as `m16n8k8`.

In this part of the lab, we’ll look at how we can use the **`m16n8k8`** TF32 MMA instruction to execute a **single `16 * 8 * 8` matrix multiplication**. As we’ll see, this isn’t actually trivial – in particular, it will require understanding the unusual way in which tensor core instructions expect their operands to be laid out in registers.

### Warp-Level Semantics

To understand how the `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` instruction works, the most important fact to establish is that tensor core instructions fundamentally operate **at the warp level**.

As we saw in Labs 1 and 2, the GPU’s hardware always executes instructions in a 32-wide SIMD fashion, with every 32 consecutive CUDA threads grouped together as 32 lanes of a vector. Viewing the GPU as a SIMD machine, virtually all the instructions we’ve seen our GPU execute so far in this course have been **element-wise vector operations**, with each instruction applying an **identical, independent** operation in each lane (modulo masking). When every instruction we execute is element-wise, we can often get away with ignoring the fact that the GPU is a SIMD machine at all, and simply pretend like every CUDA thread its executing its own independent stream of instructions. However, tensor core instructions **break this illusion**, because they are **not element-wise**.[3](https://accelerated-computing.academy/fall25/labs/lab6/#illusion)

When a warp executes a tensor core operation like our `m16n8k8` instruction, it is not executing a separate, independent matrix multiplication for each CUDA thread in the warp; rather, it is executing a **single `16 * 8 * 8` matrix multiplication** cooperatively across the **entire warp**, with the input and output data for the instruction **distributed** across the registers of all the CUDA threads in the warp. When thinking about tensor core instructions, it’s most helpful to think of each “register” in your program as a **32-word-wide vector register**, rather than as a single scalar register per CUDA thread.

With all of that in mind, let’s take a look at how the `m16n8k8` instruction we’re using actually expects data to be laid out in registers. First, a bit of math:

- The AAA matrix is `16 * 8` words, so we need `16 * 8 / 32 = 4` registers to store it.
- The BBB matrix is `8 * 8` words, so we need `8 * 8 / 32 = 2` registers to store it.
- The CCC/DDD matrix is `16 * 8` words, so we need `16 * 8 / 32 = 4` registers to store it.

Accordingly, the PTX syntax for invoking our tensor core instruction looks like this:

```
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
    {%0, %1, %2, %3},     /* 'D' matrix */
    {%4, %5, %6, %7},     /* 'A' matrix */
    {%8, %9},             /* 'B' matrix */
    {%10, %11, %12, %13}; /* 'C' matrix */
```

(PTX syntax supports `/* ... */` comments.)

From the perspective of each CUDA thread, each of these `%0`, `%1`, etc operands is a **1-word scalar register**. Collectively across the entire warp, each operand is a **32-word vector register**.

How does the `m16n8k8` instruction expect data to be packed into these registers? We present the layouts below.

![](https://accelerated-computing.academy/fall25/labs/lab6/images/tf32_16x8x8_a.png)

![](https://accelerated-computing.academy/fall25/labs/lab6/images/tf32_16x8x8_b.png)

![](https://accelerated-computing.academy/fall25/labs/lab6/images/tf32_16x8x8_c.png)

You can flip through interactive versions of each matrix here:

Ampere Matrix Layout Visualizations

Matrix AMatrix BMatrix C

# Ampere TF32 16x8x8 Tensor Core 'A' Matrix Layout

## Logical Layout

0

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

## Register Layout

Lanes

Reg 0

Reg 1

Reg 2

Reg 3

0

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

0

1

2

3

8

9

10

11

16

17

18

19

24

25

26

27

32

33

34

35

40

41

42

43

48

49

50

51

56

57

58

59

64

65

66

67

72

73

74

75

80

81

82

83

88

89

90

91

96

97

98

99

104

105

106

107

112

113

114

115

120

121

122

123

4

5

6

7

12

13

14

15

20

21

22

23

28

29

30

31

36

37

38

39

44

45

46

47

52

53

54

55

60

61

62

63

68

69

70

71

76

77

78

79

84

85

86

87

92

93

94

95

100

101

102

103

108

109

110

111

116

117

118

119

124

125

126

127

(The [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688) also contains its own versions of these diagrams.)

Essentially:

- The ‘A’ matrix is split into 4 quadrants. Each quadrant is mapped to a separate register, first top-to-bottom then left-to-right. Within each quadrant, the data is laid out in the corresponding register in row-major order.

- The ‘B’ matrix is split vertically into 2 halves. Each half is mapped to a separate register, first top then bottom. Within each half, the data is laid out in the corresponding register in column-major order.

- The ‘C’ matrix (and ‘D’ matrix) is split vertically into 2 halves, and each half is sliced into alternating vertical stripes, with all the even stripes grouped together and all the odd stripes grouped together. Each stripe-set in each half is mapped to a separate register. Within each stripe-set, the data is laid out in the corresponding register in row-major order.


Recall that for the ‘A’ matrix, the “vertical” and “horizontal” dimensions correspond to the `i` and `k` indices in the matrix multiply computation, whereas for ‘B’ they correspond to the `k` and `j` indices, and for ‘C’ they correspond to `i` and `j`.

### A Note on Register Types

You now have almost everything you need in order to invoke the `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` instruction to perform a `16 * 8 * 8` matrix-multiply-accumulate. There is, however, one remaining quirk of the PTX interface to be aware of: this PTX instruction expects every operand to be a 32-bit **integer** register. Of course, the bits these integer values carry will actually encode 32-bit floating-point data, but it expects them to be integer registers nonetheless. To cope with this, you can use the built-in [`__float_as_uint`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__float_as_uintf) and [`__uint_as_float`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__uint_as_floatj) functions to reinterpret the bits of a `float` as a `uint32_t`, and vice-versa. (These conversion functions are purely a compile-time formality and should ultimately have zero cost at run time.)

### Implementation

> **Deliverable:** In the file `exercise_mma.cu`, implement the function `mma_16x8x8_kernel` to perform a single `16 * 8 * 8` matrix multiplication on the matrices stored in `a` and `b`, and accumulate the results of that matrix multiplication into `c`, using the tensor core instruction `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. In addition to invoking this tensor core instruction, your kernel can use whatever additional CUDA logic you like to compute indices, move data around, etc. The data in `a`, `b`, and `c` is stored in row-major layout in global memory. Note that the kernel `mma_16x8x8_kernel` will be launched with exactly 32 CUDA threads (one warp).

> **Prelab Question:** Look at the assembly code generated for your `exercise_mma.cu` file. What does the generated SASS look like? Can you find the tensor core instruction? Paste the relevant chunk of SASS in your answer and the output when running `exercise_mma.cu`.

## Part 1: Accelerating Matrix Multiply

Now that we’ve seen how to invoke tensor core instructions on our GPU, we’re ready to integrate them into our full matrix multiply kernel!

For this lab, we’ll be focusing on just a subset of the problem sizes from Lab 5. Here they are:

| `size_i` | `size_j` | `size_k` |
| --- | --- | --- |
| `3072` | `3072` | `3072` |
| `2048` | `3072` | `3072` |
| `1024` | `3072` | `3072` |
| ` 512` | `3072` | `3072` |
| ` 256` | `3072` | `3072` |
| ` 128` | `3072` | `3072` |
| `  64` | `3072` | `3072` |
| `  32` | `3072` | `3072` |
| `  16` | `3072` | `3072` |

### Analysis

To understand the maximum performance we can achieve with our tensor core implementation on each of these problem sizes, we can repeat a similar analysis to the one we carried out for the previous lab. For this analysis, you can assume that the theoretical peak TF32 tensor core throughput on our 4000 Ada GPU is given by:

```
  (128 FLOP / tensor core / cycle)
* (4 tensor cores / SM)
* (48 SMs)
* (2.175 GHz)

= 53.45 TFLOP/s
```

> **Question 1 for final write-up:** For **each of the problem sizes** in this lab, walk through the following analysis (you may find it helpful to reuse some of your calculations from [Lab 5](https://accelerated-computing.academy/fall25/labs/lab5) Question 2):
>
> 1. Considering the total number of FLOPs required to process this problem size, what is the fastest we could process this problem size if **tensor core throughput** were the only constraint?
>
> 2. Considering (1) as well as the minimum time required to access each unique matrix element in DRAM, what **lower bound** does this imply for the run time of our algorithm? Is this workload compute-bound or bandwidth-bound?
>
> 3. Considering (2), what is the **maximum TFLOP/s** we could achieve on this problem size? (This is just (2) divided by the total FLOPs.)
>
> 4. How does (3) **compare** to the maximum throughput achievable if we were to use **FMAs** rather than tensor cores (as we calculated in Lab 5 Question 2.6)? Is the workload constrained by the same resource (either compute or bandwidth) in both cases, or is one scenario compute-bound while the other is bandwidth-bound?

### Implementation

Our goal for the final part of this lab will be to write a matrix multiply kernel which uses **tensor cores** to run **faster than any FMA-based matrix multiply realistically could** on our largest problem sizes.

To calibrate our expectations for how fast an FMA-based kernel could realistically run, we’ve measured the performance of NVIDIA’s highly-optimized [cuBLAS library](https://developer.nvidia.com/cublas) on each problem size **when running without tensor cores**:[4](https://accelerated-computing.academy/fall25/labs/lab6/#cublas_fp32)

**cuBLAS Performance Without Tensor Cores:**

| `size_i` | `size_j` | `size_k` | Time (ms) | Throughput (TFLOP/s) |
| --- | --- | --- | --- | --- |
| `3072` | `3072` | `3072` | `3.152` | `18.396` |
| `2048` | `3072` | `3072` | `2.174` | `17.781` |
| `1024` | `3072` | `3072` | `1.090` | `17.726` |
| `512` | `3072` | `3072` | `0.559` | `17.295` |
| `256` | `3072` | `3072` | `0.356` | `13.566` |
| `128` | `3072` | `3072` | `0.256` | `9.437` |
| `64` | `3072` | `3072` | `0.194` | `6.243` |
| `32` | `3072` | `3072` | `0.181` | `3.332` |
| `16` | `3072` | `3072` | `0.181` | `1.666` |

Our goal will be to write an implementation which beats these non-tensor-core cuBLAS numbers on the **following problem sizes:**

- `3072 * 3072 * 3072`
- `2048 * 3072 * 3072`
- `1024 * 3072 * 3072`
- `512 * 3072 * 3072`
- `256 * 3072 * 3072`
- `128 * 3072 * 3072`

For the other problem sizes, your implementation should be correct, but it’s okay if it achieves worse performance than cuBLAS.

> **Deliverable:** In the file [`matmul_3.cu`](https://github.com/accelerated-computing-class/lab6/blob/main/matmul_3.cu), implement the function `launch_matmul_tensor`, and any associated kernels, so that when `size_i` is `3072`, `2048`, `1024`, `512`, `256`, or `128`, it achieves a higher throughput than our FMA-based cuBLAS baseline. To do this, you will (almost certainly) need to use tensor cores.

To hit this performance target, you don’t need any techniques other than what we’ve already discussed in Lab 4, Lab 5, and this lab. All the suggestions from the previous labs continue to apply; a few especially important ones which you may find it helpful to keep in mind are:

- **Avoid register spills**, by not exceeding the register file capacity and by not accessing arrays using dynamic indices.

- **Use microtiles for register-level reuse.** This is still relevant when using tensor cores! (But how should you structure the microtiles now?)

- **Overlap data movement with computation**, using e.g. asynchronous copy instructions.

- **Avoid many-way bank conflicts**, either by adjusting your data layouts or by adjusting your access patterns.


Good luck! Once you’ve implemented your optimized kernel, you can answer the final question of the lab:

> **Question 2 for final write-up:** How does the performance of your implementation compare to the cuBLAS FMA baseline for each problem size? What fraction of theoretical peak throughput (calculated in Question 3.3) were you able to achieve for each problem size? What did you need to change about your kernel design in order to make use of tensor core instructions? What RRMSE numbers do you observe for your implementation, and how do they compare to the RRMSE numbers for your non-tensor-core implementation from Lab 5? Did you encounter any interesting bugs along the way? Finally, optionally: do you have any ideas for how it might be possible to develop an implementation which runs even faster?

Congratulations – you’ve reached the end of the matrix multiplication labs for 6.S894! We hope you’ve had as much fun working through them as we’ve had creating them.

You’re now well on your way to being able to implement the kinds of high-performance matrix multiplication kernels which power the world’s most computationally demanding deep learning applications, as well as important applications in many other domains.

## Further Reading

If you want to learn more about matrix multiplication, there are a huge number of additional topics you may find it interesting to look into, including:

- **Reduced precisions:**
  - The tensor cores on our GPU support 16-bit floating point ( [PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688)), 8-bit integer ( [PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-i8-f8)), 8-bit floating point ( [PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832)) and 4-bit integer ( [PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832)) precisions. Handling these precisions efficiently poses new and interesting challenges, like needing to pack multiple values into each 32-bit word ( [CUDA docs](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__bfloat162.html#struct____nv__bfloat162)).

  - NVIDIA’s newest Blackwell GPUs support exotic formats like 4-bit floating point and 6-bit (!) floating point ( [link](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-valid-comb-detail)).
- **Fused and variant kernels:**
  - It’s very common for high-performance machine learning applications to employ kernels which implement matrix multiplication **combined with some other operation**. This can look like applying an element-wise operation or a reduction on the final output of a matrix multiplication before writing it back to DRAM ( [relevant paper](https://dl.acm.org/doi/pdf/10.1145/3620666.3651369)), but it also includes even more complicated kernel designs like FlashAttention ( [link](https://arxiv.org/abs/2205.14135), and [state-of-the art version](https://arxiv.org/abs/2505.14201v1)) which embed matrix-multiply-like computations inside other kinds of workloads. An interesting recent example is FLUX ( [link](https://arxiv.org/abs/2406.06858v4)), which combines matrix multiplication with cross-GPU communication.
- **Higher-level tools:**
  - In this lab we’ve been programming the GPU at a very low level of abstraction, but it’s also possible, and often preferable, to write matrix multiplications using higher-level CUDA libraries like CUTLASS ( [link](https://github.com/NVIDIA/cutlass)), or domain-specific languages like Triton ( [link](https://triton-lang.org/main/index.html)). Many of the performance optimizations used in the implementations of those tools are techniques that you now know how to implement by hand!
- **Improved scheduling and partitioning:**
  - NVIDIA’s best-performing matrix multiply kernels use techniques slightly more advanced than (but very similar to!) the ideas we’ve discussed in these past three labs. Some of the key differences from the designs we’ve covered include overlapping data movement with computation using software pipelines with more than two stages (i.e. staging data into a ring buffer rather than double-buffering it) ( [link](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)), and using partitioning strategies which achieve better load-balancing than split-k, such as “stream-k” ( [link](https://arxiv.org/abs/2301.03598)).
- **Newer GPUs:**
  - NVIDIA’s Hopper generation of GPUs includes several new features which are relevant when writing matrix multiply kernels. Notably, these include an evolved version of asynchronous copy instructions using something called the “tensor memory accelerator” (TMA) ( [link](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)), as well as new extremely high-throughput tensor core instructions which run asynchronously, can load their inputs directly from shared memory, and work at a granularity of four warps at a time ( [PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)). Even though these features are different than the ones we’ve covered in this course, we hope you feel that you’re now very well-prepared to figure out how to use them (we think you are!).
- **Alternative platforms:**
  - Because deep learning is such an active space, there’s now a veritable zoo of different hardware accelerator platforms all designed to multiply matrices. Spanning various levels of readiness, performance, and public availability, these include: [Google’s TPUs](https://cloud.google.com/tpu), [AMD’s GPUs](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html), [Apple’s M-series chips](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/), [Intel’s Gaudi](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi-overview.html?cid=sem&source=sa360&campid=2024_ao_cbu_us_gmocoma_gmocrbu_awa_text-link_brand_exact_cd_HQ-ai-gaudi_3500268603_google_b2b_is_non-pbm_intel&ad_group=AI_Brand-Gaudi_Gaudi_Exact&intel_term=intel+gaudi&sa360id=43700079829610445&gad_source=1&gclid=CjwKCAjwjsi4BhB5EiwAFAL0YMUtHoQrSRnN4bQqFSg3SvR9uBaTz8x4oA3q6AbsQUgtmOb8VOlOkxoCc9IQAvD_BwE&gclsrc=aw.ds), [Graphcore’s IPUs](https://www.graphcore.ai/bow-processors), [Amazon’s Trainium](https://aws.amazon.com/machine-learning/trainium/), [Tenstorrent’s various accelerators](https://tenstorrent.com/), [Cerebras’s Wafer-Scale Engine](https://cerebras.ai/condor-galaxy), [Groq’s TSPs](https://groq.com/groq-tensor-streaming-processor-architecture-is-radically-different/), [the chips in Tesla’s Dojo supercomputer](https://en.wikipedia.org/wiki/Tesla_Dojo), [Etched’s Sohu](https://www.etched.com/), [SambaNova’s RDU](https://sambanova.ai/technology/sn40l-rdu-ai-chip), [Microsoft’s MAIA](https://azure.microsoft.com/en-us/blog/azure-maia-for-the-era-of-ai-from-silicon-to-software-to-systems/), and [Meta’s MTIA](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/) – and we’re sure we forgot some! You may get a chance to program some of these in the future – and although they’re not NVIDIA GPUs, many of the fundamental ideas we’ve covered should be relevant as you learn to program **any** of them!

In a few weeks, we’ll start discussing ideas for final projects or the final labs (whichever you’re interested in). If any of the matrix-multiplication-related topics above sound interesting to you, keep them in mind when you’re thinking about what you might want to work on for your final project, and keep all these ideas in mind for the optional final labs!

* * *

1

NVIDIA has developed an external library called [“CUTLASS”](https://github.com/NVIDIA/cutlass) which provides a higher-level C++ interface for interacting with tensor cores. However, CUTLASS is built on top of many layers of complicated C++ template metaprogramming machinery, and in the course staff’s experience, accessing tensor cores directly via PTX provides better clarity about what’s actually going on. Libraries like CUTLASS can be convenient in practice, but they’re never necessary; anything you can do using CUTLASS, you can also do yourself using inline PTX.

5

In practice, thanks to compiler optimizations in the PTX-to-SASS translation step, if you write CUDA code which implements three-way bit-wise operations in terms of normal two-way bit-wise operators like `(a & b) | c`, the compiler will sometimes end up generating fast `LOP3` instructions for you at the SASS level anyway. However, explicitly invoking the `lop3` instruction via inline PTX provides more control.

2

In PTX there is also an API called [`wmma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-wmma-mma), which superficially appears to offer yet another way to use the machine’s tensor cores. However, inspecting the SASS generated for `wmma.mma` instructions reveals that, on our GPU, it ultimately compiles to the same `HMMA` instructions which are already exposed through the `mma` API we’re using for this lab. As far as we can tell, this alternate `wmma` API exists mostly for historical reasons.

3

Tensor core instructions aren’t the only instructions on the GPU with warp-level semantics; there are also [warp-level reductions and warp-level permutations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions), among others. [This blog post](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) has some interesting commentary on such warp-level functions and their history.

4

We make sure the cuBLAS kernels we’re calling won’t use tensor cores by explicitly requesting matrix multiplies in full FP32 precision as opposed to TF32 precision.