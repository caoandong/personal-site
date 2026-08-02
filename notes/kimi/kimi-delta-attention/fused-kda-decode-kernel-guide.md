# Kimi Delta Attention CUDA Decode Guide

This guide explains [`fused_kda_decode_kernel.cu`](./fused_kda_decode_kernel.cu)
from the outside in.

The shortest useful description is:

> One kernel call consumes one new token, turns it into $q$, $k$, and $v$,
> edits the head's persistent KDA memory, reads the edited memory with $q$, and
> saves the result and caches for the next token.

This is a **decode kernel**, not the chunkwise training kernel. There is no loop
over sequence length here. The caller launches it again for every generated
token.

---

## 1. Keep the reference algorithm beside the CUDA

For one batch item and one head, the readable recurrence is:

```python
# q, k: [K]
# v:    [V]
# S:    [K, V]

q = normalize(q) / sqrt(K)
k = normalize(k)

S = decay[:, None] * S
prediction = S.T @ k
correction = beta * (v - prediction)
S = S + outer(k, correction)
output = S.T @ q
```

In equations:

$$
\bar S = \operatorname{Diag}(d)S
$$

$$
\hat v = \bar S^\top k
$$

$$
c = \beta(v-\hat v)
$$

$$
S' = \bar S + kc^\top
$$

$$
o = {S'}^\top q
$$

The meanings are:

```text
d       how much old memory survives
k       where to write
v       what the current token wants that address to contain
v_hat   what memory already returns from that address
c       the error that still needs to be written
beta    how strongly to write the error
q       where to read
```

Everything else in the file exists to calculate these lines quickly and to
maintain the surrounding convolution and output caches.

---

## 2. The state is transposed in memory

The mathematical state is:

$$
S \in \mathbb{R}^{K\times V}
$$

A key-space vector enters on the left, and a value-space vector comes out:

$$
S^\top k \in \mathbb{R}^{V}
$$

The CUDA kernel physically stores the transpose:

```text
state[slot, head, v, k]
```

or:

$$
H = S^\top \in \mathbb{R}^{V\times K}
$$

The last two dimensions both happen to be 128, so the public shape
`[slots, H, 128, 128]` does not reveal this distinction. The indexing in the
main loop does:

```cpp
(head * V + v) * K + k
```

With the stored orientation, the core recurrence becomes:

```python
# H is the stored state, with shape [V, K].
H = H * decay[None, :]
prediction = H @ k
correction = beta * (v - prediction)
H = H + outer(correction, k)
output = H @ q
```

This is exactly the same algorithm. Storing one complete `K` row contiguously
just makes the GPU implementation convenient.

---

## 3. Inputs, outputs, and persistent memory

The public function starts at
[`fused_kda_decode`](./fused_kda_decode_kernel.cu#L946).

| Tensor | Shape | Purpose |
| --- | --- | --- |
| `x` | `[B, 3 * H * 128]` | Current token's packed Q/K/V inputs |
| `weight` | `[3, 4, H * 128]` | Width-4 convolution weights |
| `conv_state` | `[slots, 3 * H * 128, 3]` | Previous three Q/K/V inputs |
| `raw_g` | `[1, B, H, 128]` | Per-key-channel decay controls |
| `raw_beta` | `[1, B, H]` | Per-head write strength |
| `state` | `[slots, H, 128, 128]` | Persistent KDA memory |
| `state_indices` | `[B]` | Maps a batch row to its cache slot |
| `output_gate` | `[B, H, 128]` or `[1, B, H, 128]` | Optional output gate |
| `out` | `[1, B, H, 128]` | Current token's attention result |

The important data types are:

```text
token inputs and output:  BF16
convolution weights:      FP32
KDA state:                FP32
arithmetic accumulators:  FP32
```

The state stays in FP32 because it is repeatedly decayed, corrected, and reused
across many tokens. BF16 is used at the model boundary to reduce bandwidth.

---

## 4. Host-side workflow

### Step 1: validate the contract

The wrapper checks devices, data types, shapes, strides, and supported head
counts. This kernel is intentionally specialized:

```text
K = 128
V = 128
convolution width = 4
H = 12, 24, 48, or 96
```

It is not a general-purpose KDA implementation.

### Step 2: split packed storage without copying

`x`, the convolution weights, optional biases, and convolution cache each hold
three adjacent regions:

```text
[Q region | K region | V region]
```

The wrapper creates Q/K/V pointers by adding byte offsets. No new tensors are
allocated and no data is copied.

### Step 3: choose a compiled kernel variant

The dispatch code selects compile-time variants for:

- optional output RMSNorm and gate;
- the decay parameterization;
- whether `beta` needs a sigmoid;
- whether the convolution cache is updated;
- the supported number of heads.

Compile-time choices remove those branches from the hot device loop.

### Step 4: launch one block per batch item and head

The selected path launches:

```cpp
grid  = dim3(B, H)
block = dim3(256)
```

Therefore:

```text
blockIdx.x -> batch row
blockIdx.y -> head
```

One block owns one head's complete $128\times128$ state update for one token.

---

## 5. Device-side workflow

The device kernel starts at
[`kda_decode_fusion_many_heads_kernel`](./fused_kda_decode_kernel.cu#L274).

### Step 1: find the sequence, head, and cache slot

The block derives:

```text
i_n   batch row
i_h   query/key head
i_hv  value/state head
slot  persistent cache slot
```

In the public decode path, query/key heads and value heads are one-to-one, so
`i_h == i_hv`.

`state_indices[i_n]` matters during serving because batch row 2 does not
necessarily own cache slot 2. Requests can be reordered while their persistent
states stay in a cache pool.

### Step 2: request the first state chunk early

The full state contains:

```text
128 value rows * 128 key columns
```

The kernel processes 32 value rows at a time:

```text
chunk 0: v = 0..31
chunk 1: v = 32..63
chunk 2: v = 64..95
chunk 3: v = 96..127
```

It starts an asynchronous global-to-shared-memory copy for chunk 0 before
calculating Q, K, and V. The memory transfer can then overlap with useful
arithmetic.

Shared memory has two state buffers:

```cpp
s_state[2][32][128]
```

While the kernel computes on one buffer, it can fill the other with a future
chunk. This is ordinary double buffering.

### Step 3: run the width-4 Q/K/V convolutions

For each channel, the kernel combines:

```text
three cached inputs + current input + optional bias
```

Then it applies SiLU:

$$
q_{raw} = \operatorname{SiLU}(\operatorname{Conv4}(x_q))
$$

$$
k_{raw} = \operatorname{SiLU}(\operatorname{Conv4}(x_k))
$$

$$
v = \operatorname{SiLU}(\operatorname{Conv4}(x_v))
$$

Intuitively, the current token's Q/K/V features include a small amount of local
history before they interact with the long-lived KDA state.

The same code shifts the convolution cache:

```text
[oldest, middle, newest] -> [middle, newest, current]
```

That prepares the cache for the next token.

### Step 4: turn the raw gate into retention

Each key channel gets its own decay value. The standard path calculates:

$$
a_h = e^{A_{log,h}}
$$

$$
z_k = g_k + dt\_bias_k
$$

$$
d_k = \exp\left(-a_h\operatorname{softplus}(z_k)\right)
$$

Because the exponent is non-positive:

$$
0 < d_k \le 1
$$

The optional lower-bounded form is:

$$
d_k = \exp\left(g_{min}\operatorname{sigmoid}(a_hz_k)\right)
$$

when `lower_bound` represents the negative bound $g_{min}$.

Geometrically, KDA can fade different directions of key space at different
rates. It does not have to forget the entire matrix uniformly.

### Step 5: turn `raw_beta` into write strength

The normal public path applies:

$$
\beta = \operatorname{sigmoid}(\text{raw\_beta})
$$

so $0 < \beta < 1$.

```text
beta near 0 -> preserve the existing association
beta near 1 -> strongly correct the association
```

### Step 6: normalize Q and K

The block cooperatively computes the two squared norms:

$$
\|q_{raw}\|^2 = \sum_k q_{raw,k}^2
$$

$$
\|k_{raw}\|^2 = \sum_k k_{raw,k}^2
$$

It then applies:

$$
q = \frac{q_{raw}}{\sqrt{\|q_{raw}\|^2+\epsilon}}\frac{1}{\sqrt{128}}
$$

$$
k = \frac{k_{raw}}{\sqrt{\|k_{raw}\|^2+\epsilon}}
$$

Normalizing $k$ makes its dot products similarity-like. Scaling $q$ keeps
the readout magnitude under control.

### Step 7: divide each state row across a warp

There are 256 threads, or eight warps.

Within every warp:

```text
lane 0  owns k = 0..3
lane 1  owns k = 4..7
...
lane 31 owns k = 124..127
```

So one warp collectively owns all 128 key coordinates of a value row:

```text
one state row H[v, :]

lane 0       lane 1                         lane 31
k 0..3       k 4..7          ...            k 124..127
   \            |                              /
    ----------- warp-wide reduction -----------
```

Each warp handles four of the chunk's 32 value rows. The inner loop processes
two rows together to reuse the same Q, K, and decay values.

### Step 8: decay one piece of the state

Each lane loads four FP32 state values and multiplies them by four decay values:

```cpp
h[k] = old_state[v, k] * decay[k];
```

Across the whole state, this is:

$$
\bar H_{v,k} = H_{v,k}d_k
$$

or, in the mathematical orientation:

$$
\bar S = \operatorname{Diag}(d)S
$$

This is the “fade old memory” step.

### Step 9: ask what memory already predicts

For each value row, every lane calculates four terms of:

$$
\hat v_v = \sum_k \bar H_{v,k}k_k
$$

A warp shuffle reduction adds the 32 lanes' partial sums. Across all rows:

$$
\hat v = \bar Hk = \bar S^\top k
$$

This asks:

> If the current key reads the old memory, what value does it already return?

This CUDA operation corresponds to the readable PyTorch expression:

```python
prediction = (k[..., None] * S).sum(-2)
```

### Step 10: calculate only the missing information

The code computes:

```cpp
correction[v] = (v[v] - prediction[v]) * beta;
```

or:

$$
c = \beta(v-\hat v)
$$

This is the delta rule:

```text
memory already correct -> correction is near zero
memory is wrong         -> write the prediction error
```

The kernel does not blindly add $v$ every time. It writes what is missing.

### Step 11: apply the outer-product update

Every lane updates its four state elements:

```cpp
new_state[v, k] = decayed_state[v, k] + correction[v] * k[k];
```

In stored orientation:

$$
H' = \bar H + ck^\top
$$

In the paper's orientation:

$$
S' = \bar S + kc^\top
$$

The outer product means:

> Attach the correction vector to the direction identified by $k$.

A future query aligned with $k$ will retrieve that correction. Queries
orthogonal to $k$ will barely see it.

The updated FP32 values are written directly back to persistent state memory.

### Step 12: read the newly updated state with Q

Before discarding the values in registers, the kernel also computes:

$$
o_v = \sum_k H'_{v,k}q_k
$$

Across all value rows:

$$
o = H'q = {S'}^\top q
$$

The update and read share the same state load. The kernel does not make a
second pass through the $128\times128$ matrix.

The roles are now easy to separate:

```text
k decides where to write
v - prediction decides what correction to write
q decides where to read
```

### Step 13: optionally normalize and gate the output

When output normalization is enabled, the block computes an RMS value:

$$
r = \sqrt{\frac{1}{128}\sum_v o_v^2 + \epsilon}
$$

and writes:

$$
y_v = \frac{o_v}{r}\,w_v\,\operatorname{sigmoid}(gate_v)
$$

Otherwise it writes $o$ directly.

The final result is converted from FP32 to BF16 and stored in `out`.

---

## 6. What changes after one call

One invocation has three observable results:

```text
out         contains this token's KDA output
state       contains the corrected long-lived memory
conv_state  contains the latest three local Q/K/V inputs
```

On the next decoding step, the kernel repeats the same process with those two
updated caches.

```text
token t
  -> local convolution cache update
  -> KDA state update
  -> output
  -> token t+1 uses the new caches
```

---

## 7. Why the kernel is fused

Without fusion, decoding could require separate launches for:

```text
Q convolution
K convolution
V convolution
SiLU
Q/K normalization
decay generation
KDA state prediction
KDA state update
KDA read
RMSNorm
output gate
cache updates
```

This implementation performs them in one kernel. Fusion avoids writing and
rereading intermediate Q/K/V, decay, prediction, correction, and raw output
tensors from global memory.

For one-token decoding, reducing memory traffic and launch latency is usually
more important than expressing the work as large matrix multiplications.

---

## 8. Does it use Tensor Cores?

No. This decode implementation contains no `mma.sync`, `wgmma`, WMMA, CUTLASS,
or cuBLAS operations.

It uses:

- ordinary FP32 multiply-add arithmetic;
- warp shuffle reductions;
- `float4` vectorized state loads and stores;
- `cp.async` global-to-shared-memory copies;
- BF16-to-FP32 conversion at input boundaries.

`float4` and `cp.async` are performance features, but they are not Tensor Core
operations.

This is sensible for a single-token decode step. The recurrence consists of
matrix-vector products and a rank-one update, which do not naturally fill large
Tensor Core matrix tiles. Chunkwise training and prefill kernels have much more
matrix-matrix parallelism and are a different optimization problem.

---

## 9. Reading map for the source file

Read the file in this order:

1. [`fused_kda_decode`](./fused_kda_decode_kernel.cu#L946): tensor contract and
   pointer splitting.
2. [`launch_kda_decode_many_heads_raw`](./fused_kda_decode_kernel.cu#L727): grid,
   block, and selected specialization.
3. [Block identity and shared memory](./fused_kda_decode_kernel.cu#L290): which
   sequence, head, and state slot the block owns.
4. [Q/K convolution and decay](./fused_kda_decode_kernel.cu#L355): local feature
   construction.
5. [V convolution and beta](./fused_kda_decode_kernel.cu#L445): value and update
   strength.
6. [Q/K normalization](./fused_kda_decode_kernel.cu#L518): block reductions.
7. [Main KDA loop](./fused_kda_decode_kernel.cu#L541): decay, predict, correct,
   update, and read.
8. [Output normalization](./fused_kda_decode_kernel.cu#L654): RMSNorm, gate, and
   BF16 store.

The helpers above the kernel are implementation machinery for asynchronous
copies, reductions, conversions, and stores. They are useful after the core
workflow is clear, not before.

---

## 10. One-page mental model

```text
CURRENT TOKEN
    |
    +-> width-4 convolution using three cached tokens
    |       |
    |       +-> q: where should I read?
    |       +-> k: where should I write?
    |       +-> v: what should be stored there?
    |
    +-> decay: which old key directions should fade?
    +-> beta: how strongly should memory be corrected?

PERSISTENT STATE H[V, K]
    |
    +-> fade:       H_bar = H * decay
    +-> predict:    v_hat = H_bar @ k
    +-> error:      c = beta * (v - v_hat)
    +-> correct:    H = H_bar + outer(c, k)
    +-> read:       o = H @ q
    |
    +-> optional RMSNorm and gate
    |
    +-> BF16 output

NEXT TOKEN REUSES
    - corrected FP32 KDA state
    - shifted BF16 convolution cache
```

The central insight is that KDA is not storing every past token. It is training
and updating a small linear map from key space to value space. Each new token
checks that map, writes only its prediction error, and immediately queries the
corrected map.
