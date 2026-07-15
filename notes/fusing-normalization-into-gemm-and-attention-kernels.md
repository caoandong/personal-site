Search

[Close Search](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#)

[Blog](https://pytorch.org/blog/category/blog/)

# Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels

By [Jacky (Junqing) Zhou, Hongtao Yu, Jackie (Jiaqi) Xu, Menglu Yu, Ethan Che, Han Xu, Darren Liu, Peng Chen (Dev Infra), Daohang Shi, Max Leung](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#)July 10, 2026[No Comments](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#respond)

### Featured projects

- [![Helion logo](https://pytorch.org/wp-content/uploads/2026/05/Helion-medium2-300x82.png)](https://pytorch.org/projects/helion/ "Helion")

![](https://pytorch.org/wp-content/uploads/2026/07/All-PyTorch-Blog-Social-Images-12.png)

Code available at: [https://github.com/facebookresearch/ads\_model\_kernel\_library/tree/main/multi\_cta\_norm\_fusion](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion) and [https://github.com/facebookresearch/ads\_model\_kernel\_library/tree/main/gdpa\_megakernel](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel)

## **TL;DR**

In this blog post, we present various novel kernel fusion techniques for common normalization ops like LayerNorm and RMSNorm, which provide significant speedup by reducing the memory-IO overhead of these highly memory-bound kernels. We start with a brief overview of the modeling importance as well as performance challenges of normalization ops common in both LLMs and ads recommendation models, then present the novel strategies in tackling the performance bottlenecks, including Lazy Pre-Norm and Multi-CTA Norm Fusion. We show that such techniques can hide as much as **90%** of a normalization kernel’s latency by fusing with GEMMs. In the end, we present the FlashNormAttention algorithm where we apply fusion for multiple normalizations around an attention kernel like GDPA \[1\], achieving up to **35%** kernel speedup.

This work is done with primarily two kernel DSLs: [TLX](https://arxiv.org/abs/2605.10905), a set of Triton DSL extensions with lower-level, hardware-aware support for GPU execution control; and [Helion](https://pytorch.org/blog/helion/), a high-level DSL that excels at developer velocity, portability, and comprehensive autotuning. Benchmarks are performed with data type bfloat16, on NVIDIA B200 GPUs in Meta’s data centers with a 750 W power cap.

## Introduction

Normalization techniques have become indispensable in most deep learning architectures due to their excellent effectiveness in stabilizing training and accelerating convergence. In particular, traditional normalization across the innermost embedding dimension (e.g. LayerNorm, RMSNorm) has been the most common and ubiquitous type in modern Large Language Models as well as recsys models like Meta’s ads models. For example, in the [Kunlun](https://arxiv.org/abs/2602.10016) \[2\] architecture deployed on Meta’s largest Recsys training foundation model, the [Generative Ads Model (GEM)](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/) \[3\], LayerNorm/RMSNorm exists in nearly all key components, such as Multi-Head Attention, Hierarchical Seed Pooling, and GDPA-enhanced \[1\] PFFN.

However, the ubiquity of normalization also brings a difficult performance challenge: it’s highly memory-bound with no TensorCore utilization. This hinders us from saturating hardware compute capabilities in model training. Using Kunlun \[2\] as an example, normalization takes up roughly 20% of the total tr

aining latency there. This means we immediately lose 20% of our hardware’s compute throughput without optimization. In a typical LLM which is more compute-bound, normalization could still take roughly 10% of total latency.

To address this, we must design our normalization kernels in an IO-aware way, carefully saving memory IO costs without compromising computation accuracy via kernel fusion, and also overlapping these memory-/CUDACore-heavy ops with TensorCore-intensive ops. Since most normalization operations follow or precede matmul operations (e.g. MLP, Attention), our work focuses on how to efficiently fuse norms with matmuls. We start by describing multiple strategies in efficiently fusing norms with single GEMMs, and in the end present the FlashNormAttention algorithm, fusing both a LayerNorm and an RMSNorm into attention.

**Note**: In the following benchmark results, we disable [elementwise affines](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html) unless otherwise specified, as we find those to incur significant performance overhead with marginal model quality effects in our models. Also, unless otherwise specified, the core optimization and algorithmic ideas presented are applicable regardless of whether elementwise affines exist, although the performance results may differ.

## 1\. Challenges of Normalization Fusion

**Overview:** In this section we discuss the challenges of fusing normalization with compute-intensive kernels like GEMMs in the typical way, which fundamentally stem from different tiling strategies. We then present a “naive” fusion solution that forces the GEMM algorithm to obey the same tiling as the normalization algorithm, and observe that it performs well for super small N, but becomes suboptimal or even infeasible as N increases due to tiling constraints and inefficiencies.

Compared with standard activation fusion (e.g. GEMM+ReLU), the fundamental challenge with normalization fusion is the difference in tiling. Normalization is by nature a reduction operation that requires access to data along an entire dimension to compute the correct result. In particular, for LayerNorm and RMSNorm, a typical kernel tiles the input along the outer dimension(s), but not the inner, meaning each CTA always needs to load entire rows of data. By comparison, a typical GEMM is tiled in both dimensions, meaning each tile does not span an entire row, making a following row-wise normalization impossible.

![](https://pytorch.org/wp-content/uploads/2026/07/1.png)

The most straightforward workaround for this would be to stretch the tile size of the GEMM so that each tile spans the entire inner dimension. For a typical (MxK) @ (KxN) GEMM, this means that the tile size along the N dimension must be larger than N (usually the next power of 2 from N). The high-level algorithm is illustrated in the following diagram:

![](https://pytorch.org/wp-content/uploads/2026/07/2-1-scaled.png)

There are two primary issues with this approach:

- It deviates from the would-be optimal tiling strategy for a pure GEMM, which would degrade the performance of the GEMM itself due to suboptimal cache behavior, pipelining behavior, etc.
- It places a hard limit on the input shapes; specifically, on how large N could be. Too large an N would not be able to fit into shared memory.

Here’s some napkin math to get a sense of how large N can be. Assume a Blackwell GPU with 228KB shared memory size, a dtype of bfloat16, and a minimum number of 2 pipelining stages for efficient pipelining/overlap. Further assume a minimum tile size of 32 for M dimension and 32 for K dimensions. Then we have:

```py
2 stages x 2 bytes / element x (tile_m x tile_k + tile_k x tile_n + tile_m x tile_n) < 228KB
=> 32 x 32 + 32 x tile_n + 32 x tile_n < 228KB / 4
=> 512 < tile_n < 1024
```

Copy

Since tile size should usually be a power of 2, this restricts tile\_n, and therefore N, to being at most 512 for this kernel to even be able to run.

We will discuss ways to work around these limitations in the following sections. But despite them, we’ve found that such fusion strategies can still produce significant gains for small values of N. For this experiment, we used [Helion](https://pytorch.org/blog/helion/) because of its high developer efficiency and exhaustive autotuning which helps in uncanonical cases like this where one of the tile sizes is hard-constrained.

Below are the benchmark results on the typical input shapes observed in ads models. Note that the latency saving is calculated as the percentage of the torch inductor’s normalization kernel’s latency. We do this so that the metric becomes independent of the latency of the base GEMM kernel (and how it compares to that of the normalization kernel), and inherently captures the headroom of fusion attempts of this sort (i.e. 100% is the best we could do and would require completely overlapping the normalization with the GEMM).

![](https://pytorch.org/wp-content/uploads/2026/07/3.png)

For small shapes like 64 and 128, this fusion strategy can yield a significant 17%-32% latency saving for the LayerNorm kernel. However, as K/N grows larger beyond 128, the gain starts disappearing and even turns into huge regression. This is because as N grows, the enforcement that tile\_n = N becomes a larger deviation from the would-be optimal tile size for an unfused GEMM kernel, and the benefit of saving memory IO gradually becomes overshadowed by the harm of distorting the base GEMM algorithm.

## 2\. Lazy Pre-Norm: A Novel Technique of Fusing Pre-Norm with Linear Layers

**Overview:** In this section we introduce a novel prologue fusion technique for fusing pre-RMSNorm into a GEMM kernel. We discuss the motivations as well as challenges of such prologue fusion, and present a novel algorithm named **Lazy Pre-Norm** that tackles these challenges via strategically delaying part of the pre-norm computation until after completing the GEMM using a mathematical trick, and yields good performance speedup.

The first idea we present gets around the aforementioned issue by fusing pre-norms with following GEMMs, as a prologue fusion. Although prologue fusion should generally be avoided, there are a few reasons it’s still worth exploring:

1. Prologue fusion bypasses the tiling issue encountered in epilogue fusion, where each CTA in a GEMM kernel doesn’t have access to entire rows of the output tensor. In contrast, each CTA does scan through entire rows of the input tensor A just by the algorithm!
2. Realistically, pre-norm has become much more prevalent in post-norm, especially in Large Language Models.

For prologue fusion to work well, we devised an optimization technique called **Lazy Pre-Norm** for a special case of pre-norm: RMSNorm without elementwise affines. Specifically we are looking to fuse:

`C = rmsnorm(A) @ B`

`where rmsnorm(A) = A * rstd(A)[:, None]`

`and rstd(A) = rsqrt((A ** 2).sum(dim=-1) / A.shape[-1] + 1e-5)`

Here’s the key difficulty of this fusion that Lazy Pre-Norm aims to resolve: in a typical tiled GEMM, although we eventually have access to entire rows (which is what allows us to compute the reduction result rstd), we access them tile by tile, and we actually need rstd in order to process each tile! This creates a cyclic dependency: we need to wait until the end of the k-loop to be able to compute rstd, but we need rstd to even start working on the loop!

To resolve this, the first key observation here is that the two inter-dependent parts are in essence different types of computation: reduction and elementwise application.

1. The rstd computation part is a **reduction** over the entire rows.
2. Using rstd to apply normalization is an **elementwise** computation on each individual element of A

Let’s tackle these two components separately. For the reduction part, the first thing to notice is that it’s not blocking anything itself, which is a nice property because it means we can compute it in parallel to the TensorCore computation. Since each CTA naturally scans along the inner dimension of A, we can just accumulate the square sum of A alongside, and in parallel to, the matmuls.

The elementwise part is more problematic as it depends on the result of the reduction, which causes cyclical dependency. The rescue here is a mathematical trick based on the key observation that the elementwise multiplication in an affine-free RMSNorm is actually **row-wise multiplication**, where all elements in the same row of A are multiplied by the same rstd. This implies the following key property:

`(A * rstd[:, None]) @ B = (A @ B) * rstd[:, None]`

`Proof:
Row-wise multiplication is equivalent to M @ A where for some diagonal matrix M. So we have (A * rstd) @ B = (M @ A) @ B = M @ (A @ B) = (A @ B) * rstd`

This is great because it means the elementwise computation can be “lazily computed” and delayed until after the whole k-loop is done, and effectively becomes an epilogue! Putting it all together, below is the kernel pseudocode for the Lazy Pre-Norm algorithm:

```py
def GEMM_norm_fusion_kernel(A, B, C):
	compute the m_tile and n_tile of this CTA
	square_sum = zeros(m_tile)
	acc = zeros(m_tile, n_tile)
	for each k_tile:
		tile_A = A[m_tile][k_tile]
		tile_B = B[k_tile][n_tile]
		acc += tile_A @ tile_B
		square_sum += (tile_A * tile_A).sum(-1) # computed in parallel to the GEMM!
	rstd = rsqrt(square_sum / A.shape[-1] + 1e-5)
	acc *= rstd[:, None]
	C[m_tile][n_tile] = acc
```

Copy

Note that while some additional computation is still incurred in each k iteration, it can be overlapped with the matmul computation. With warp specialization, the kernel’s warp partitioning and execution would look like:

![](https://pytorch.org/wp-content/uploads/2026/07/4.png)

Notice that this algorithm still features a key disadvantage of prologue fusion: the RMSNorm computation is redundantly done across many CTAs (think about all CTAs computing the same rows but different columns of the output tensor; more on this in Section 3). However, since Lazy Pre-Norm makes sure that most of this computation is fully overlapped with TensorCore, the redundancy is acceptable and still yields good performance gains.

![](https://pytorch.org/wp-content/uploads/2026/07/5.png)

A few limitations to note about the Lazy Pre-Norm algorithm:

1. It cannot easily support elementwise affines, which work as a column-wise multiplication. It would break our precondition that the elementwise operation must be a row-wise multiplication.
2. It does not work with LayerNorm, because the elementwise part of LayerNorm involves subtraction and is not a simple row-wise multiplication.
3. The backward implementation for this fusion would be tricky, because we never materialize rmsnorm(A) anywhere in forward. As such, we’d need to reconstruct rmsnorm(A) from A and rstd on the fly in computing both dA and dB.

## 3\. Multi-CTA Norm: Fusing Post-Norm with Linears as Epilogue

**Overview:** Despite the good speedup, the Lazy Pre-Norm prologue fusion still has its limitations and cannot be generalized to most norm use cases. In this section we discuss a more general technique for fusing post-norms with GEMMs, and come back to the realm of epilogue fusion, directly tackling the tiling mismatch issue presented in Section 1, using **CTA clusters** and **Distributed Shared Memory**.

We borrow an idea from [Quack](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md) and extend it beyond standalone norm kernels to the fused kernels. The Quack norm kernels leverage [CTA clusters](https://docs.nvidia.com/cuda/parallel-thread-execution/#cluster-of-cooperative-thread-arrays) to partition large N among different CTAs in the same cluster, and let them collaborate on a single reduction across N by communicating necessary data with each other via **distributed shared memory**. This allows us to have multiple CTAs collaboratively divide and work on the same rows of data, and communicate with each other as needed by normalization, without incurring the cost of global memory IO.

![](https://pytorch.org/wp-content/uploads/2026/07/6.png)

As mentioned above, most normalization ops can be decomposed into a reduction part (e.g. rstd for RMSNorm, mean and variance for LayerNorm), and a following elementwise part that utilizes the reduction result. Only the reduction part requires scanning through the entire N dimension, which we divide and conquer within a CTA cluster. Because the reduction result is usually small (because, well, it’s a reduction), a quite minimal DSMEM communication overhead is needed to send/receive it to/from other CTAs.

![](https://pytorch.org/wp-content/uploads/2026/07/7.png)

Note that this idea tackles the exact same problem that we are facing with norm fusion – simply that N is too large! (although the reason and threshold for N being too large differ). This means that we can simply take this multi-CTA algorithm and put it in the epilogue of our GEMM, and the fusion is done!

```py
def GEMM_norm_fusion_kernel(A, B, C):
	compute the m_tile and n_tile of this CTA
	acc = zeros(m_tile, n_tile)
	for each k_tile:
		tile_A = A[m_tile][k_tile]
		tile_B = B[k_tile][n_tile]
		acc += tile_A @ tile_B
	acc = multi_cta_norm(acc) # where DSMEM communication happens
	C[m_tile][n_tile] = acc
```

Copy

Note that this fusion is by no means free and places a few constraints on the kernel other than introducing DSMEM overhead, which could potentially cause regression on the base GEMM kernel we are fusing norm into:

1. It puts strong restrictions on CTA scheduling. In particular, adjacent CTAs in a cluster must share the same m\_tile but different n\_tile’s.
2. Because of #1, [paired-CTA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair) matmul is difficult
3. Similarly, because of #1, [tile super-grouping](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations) \[5\] cannot be done
4. This still doesn’t unlock N from being indefinitely large. We are still bounded by the single-CTA limit (around 512) multiplied by the max cluster size. On Blackwell, the portable max cluster size is [8](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#thread-block-clusters). This limits N to be at most 4096.

Nonetheless, the benefit of saving significant memory IO still far outweighs the limitations. We chose [TLX](https://pytorch.org/blog/enabling-cluster-launch-control-with-tlx/) for this kernel which strikes a good balance between flexibility / dev efficiency and lower-level hardware control, both of which are critical in this case study. We built the fused kernel on top of the [TLX GEMM kernel](https://github.com/facebookexperimental/triton/blob/main/third_party/tlx/tutorials/blackwell_gemm_ws.py) with warp specialization. We benchmarked it against some common shapes in ads modelling (M = 256k, K = O(512), N = O(512)), and achieved the following performance result.

![](https://pytorch.org/wp-content/uploads/2026/07/8.png)

Note that we capped K and N at 2048, because as they reached 4096, the latency became completely dominated by the GEMM and the normalization takes less than 5% of the total latency.

### What about backward? Fusion regrouping.

**Overview:** In this subsection, we discuss the additional challenge with implementing the same fusion idea for backward: **forward epilogue fusion naturally becomes prologue fusion in backward**. We discuss the key issues with prologue fusion, and present a novel workaround solution which fuses norm with different GEMMs in forward v.s. in backward, **resulting in efficient epilogue fusion in both**.

The backward of LayerNorm and RMSNorm also involves reduction, which wouldn’t be a big issue as we can resolve it in a similar fashion as in forward. Efficient backward computation would also need the intermediate reduction result stored from forward, which also wouldn’t be a big issue since the reduction result is 1-dimensional and results in minimal IO. The real issue is that epilogue fusion in forward becomes prologue fusion in backward.

```py
# forward formula
C = norm(A @ B)

# backward formula
dA = norm_backward(dC) @ B.T
dB = A.T @ norm_backward(dC)
```

Copy

Notice how in backward, the norm\_backward computation happens before the GEMM, making a potential fusion prologue fusion. We’ve discussed some general drawbacks of prologue fusion in Section 2, but in this specific case, let’s look at a potential prologue fusion solution to understand more why it’s an issue.

```py
def GEMM_norm_bwd_fusion_kernel(dC, BT, dA):
	compute the m_tile and n_tile of this CTA
	acc = zeros(m_tile, n_tile)
	for each k_tile:
		tile_dC = dC[m_tile][k_tile]
		tile_dC = multi_cta_norm_bwd(tile_dC) # where DSMEM communication happens
		tile_BT = BT[k_tile][n_tile]
		acc += tile_dC @ tile_BT
	dA[m_tile][n_tile] = acc
```

Copy

There are several performance issues with this approach:

1. The norm bwd computation is on the critical path blocking the GEMM computation for every iteration! Compare this to epilogue fusion where a single computation is done at the end of the main loop.
2. Redundant computation is being done for the norm backward. Remember that each row of dC is loaded separately by different CTAs that compute different tiles of dA that share the same m\_tile but different n\_tile’s. Each of those CTAs would need to do the same norm backward computation on the same dC tiles.
3. This fusion kernel only computes dA, but we’d also need to compute dB as well, where we’d compute norm\_backward(dC) again, leading to more redundant computation.

What’s the solution here? Well, there isn’t much we can do unless it’s some specialized cases like the one discussed in Section 2. So just avoid prologue fusion. To do that we have to be a bit more flexible in our fusion strategy: since normalization layers usually stand between linear layers, what if we fuse the normalization op with different linears in forward v.s. in backward?

![](https://pytorch.org/wp-content/uploads/2026/07/9.png)

It’s easy to see that now backward is also epilogue fusion, and that there’s no longer any redundant computation for the norm backward. Also more importantly, the fusion becomes structurally identical to the forward fusion! Just replace `multi_cta_norm` with `multi_cta_norm_bwd` in the forward kernel, and you’ve got your backward kernel. Below are the benchmark results on the same shapes as forward.

![](https://pytorch.org/wp-content/uploads/2026/07/10.png)

Note that for this idea to work, the linears do not have to be linears exactly. For example, in LLM architectures, we might see a pattern like attention -> norm -> linear, or vice versa. These could still adopt the same optimization technique, as long as what sandwiches the norm is compute-intensive ops which are beneficial (and feasible) to fuse norms into.

The next section discusses an example of fusing norms into attention.

## 4\. FlashNormAttention: Fusing both Pre-Norm and Post-Norm into FlashAttention-variant kernels

**Overview:** In this section we look at a case study of applying the aforementioned fusion ideas to the [GDPA kernel](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/) \[1\], and present the **FlashNormAttention** algorithm. The GDPA kernel is heavily used in Meta ads models and specifically the [Kunlun](https://arxiv.org/abs/2602.10016) \[2\] architecture, and is a generalized attention kernel redesigned from [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute) \[6\]. As such, most of the optimization ideas discussed below are generalizable to other attention kernels like FlashAttention. The algorithm uses the exact same algorithmic trick as the multi-CTA GEMM+norm fusion from above, but is at a higher level of complexity (attention v.s. GEMM, fusing two norms v.s. one). Many optimization techniques are adopted to make it performant and discussed below, including:

- SMEM / TMEM reuse to reduce memory pressure
- Register subtiling to reduce register pressure
- Fine-tuned warp specialization to parallelize heavy CUDACore operations
- Norm recomputation in backward to avoid IO costs of saving additional tensors in forward
- Using advanced hardware features like [TMA\_REDUCE\_ADD](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278) and TensorCore Accumulate for better pipeline efficiency

The typical PFFN block in Kunlun \[2\] uses a GDPA kernel as the backbone, but also surrounds it with a couple of normalization and residual connections. The diagram below shows how data flows inside a PFFN block.

![](https://pytorch.org/wp-content/uploads/2026/07/11-scaled.png)

Notice how IO-heavy this is with reduction and elementwise kernels scattered both before and after the GDPA kernel. Our goal is to fuse all of these operations into one single kernel, that we call the **FlashNormAttention**. This is like a “megakernel” performing all operations in a module, but it differs, in intention and meaning, from the original [megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) \[7\] in that it aims not just to save kernel launch costs, but to **save the total amount of data transfer to/from HBM**, which is usually more of a bottleneck.

The overall fusion plan here stays the same: leveraging CTA clusters and distributed shared memory to collaboratively compute normalization. Note that the GDPA here is multi-head, and the norms are done across all heads. So even though in the typical GDPA/FA kernel, a single CTA has access to the full head dimension, it still needs data from the other heads for normalization, necessitating the multi-CTA norm algorithm.

Let’s start with the original GDPA kernel algorithm which we describe in pseudocode below. For simplicity only the bare bones relevant to fusion are included. For the detailed, optimized algorithm, refer to the original [GDPA blog](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/).

```py
# input: Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]
# metaparam: BLOCK_M (tile size of m_tile on seq_len_q), BLOCK_N (tile size of n_tile on seq_len_kv)
# grid on (batch_size, seq_len_q // BLOCK_M, H)
def gdpa_fwd_kernel(Q, K, V, output):
	compute the batch_idx, m_tile and head_idx for this CTA
	q = Q[batch_idx, m_tile, head_idx, :] # [BLOCK_M, head_dim], B and H dimensions are indexed
	acc = zeros(BLOCK_M, head_dim)
	for n_tile over entire seq_len_kv:
		k = K[batch_idx, n_tile, head_idx, :] # [BLOCK_N, head_dim], B and H dimensions are indexed
		v = V[batch_idx, n_tile, head_idx, :] # [BLOCK_N, head_dim]
		p = elementwise_activation(q @ k.T) # [BLOCK_M, BLOCK_N]
		acc += p @ v # [BLOCK_M, head_dim]
	output[batch_idx, m_tile, head_idx, :] = acc
```

Copy

Even before getting into fusion work, let’s start with a key tweak on this algorithm. In Kunlun’s \[2\] use case, we observed that seq\_len\_q is usually large (O(1k)) while seq\_len\_kv is usually small (O(128)). This makes the inner loop’s pipelining very shallow and exposes prologue and epilogue overhead. To improve the performance for this case, we swap the role of Q and K/V in the kernel, gridding on KV and looping over Q. Notice that this algorithm is only numerically correct when we do not tile on seq\_len\_kv.

```py
# input: Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]
# metaparam: BLOCK_M (tile size of m_tile on seq_len_q)
# grid on (batch_size, H)
def gdpa_fwd_kernel_short_kv(Q, K, V, output):
	compute the batch_idx and head_idx for this CTA
	k = K[batch_idx, :, head_idx, :] # [seq_len_kv, head_dim], B and H dimensions are indexed
	v = V[batch_idx, :, head_idx, :] # [seq_len_kv, head_dim]
	for m_tile over entire seq_len_q:
		q = Q[batch_idx, m_tile, head_idx, :] # [BLOCK_M, head_dim], B and H dimensions are indexed
		p = elementwise_activation(q @ k.T) # [BLOCK_M, BLOCK_N]
		output[batch_idx, m_tile, head_idx, :] = p @ v # [BLOCK_M, head_dim]
```

Copy

Although this optimization is not directly related to our topic, we still include it here because we built our fusion kernel on top of this version, and benchmarks were done against this version. We include it for completeness as it was not mentioned in the original GDPA blog.

Now we fuse the norms and residuals into the kernel. The idea is the same as above with multi-CTA reduction. The only caveat is that here CTAs in the same cluster should share the same batch\_idx and process different head\_idx. Also, notice that the layernorm here is a prologue fusion. While the pipelining and the matmuls’’ dependency on it are still an issue (which we address below), fortunately we don’t have the problem of redundant layernorm computation thanks to K/V being short. Since we don’t tile over K/V’s length, every tile of Q will only be loaded and processed by one single CTA.

```py
# input: Q: [batch_size, seq_len_q, H, head_dim], K/V: [batch_size, seq_len_kv, H, head_dim]
# metaparam: BLOCK_M (tile size of m_tile on seq_len_q)
# grid on (batch_size, H)
def gdpa_fwd_fusion_kernel_short_kv(Q, K, V, output):
	compute the batch_idx and head_idx for this CTA
	k = K[batch_idx, :, head_idx, :]
	v = V[batch_idx, :, head_idx, :]
	for m_tile over entire seq_len_q:
		q = Q[batch_idx, m_tile, head_idx, :]
		ln_q = multi_cta_layernorm(q) # multi-CTA norm 1
		p = elementwise_activation(ln_q @ k.T)
		gdpa_out = p @ v
		gdpa_out += ln_q # residual connection 1
		out = multi_cta_rmsnorm(gdpa_out) # multi-CTA norm 2
		out += q # residual connection 2
		output[batch_idx, m_tile, head_idx, :] = out
```

Copy

Although this all looks nice and great, there are two critical issues hidden behind this pseudocode:

1. **Memory pressure:** This massive fusion brings high pressure on registers and shared memory usage. We need to keep many more things that didn’t exist before, such as ln\_q and rmsnorm(gdpa\_out). If the execution is purely sequential, this would be fine because as we produce the output for an op, its input can immediately be freed. But this is not the case here due to residual connections. Notice how the lifetime of q and ln\_q spans across a large region because we need to keep them for later residual connection computation. This means we most definitely have to dedicate some shared memory for these variables, increasing the total memory needed. In fact, with this naive algorithm version, we observed the shared memory usage to double, significantly exceeding the limit.
2. **CUDA Core dominance & pipeline stalls:** Even though we eliminated most of the memory IO with the fusion, the CUDA core computation for the norms and residual connections still remains, and blocks Tensor Core utilization. Fine-tuned warp specialization and pipelining are needed to hide as much CUDA core latency as possible.

For memory pressure, we applied 3 main optimization ideas

- **Memory buffer reuse:** A key technique for saving memory usage is to let non-overlapping data share the same memory buffers. A good example in our case is the shared memory buffer for out. In the last line of code above, it looks like we are directly storing out from registers to HBM, but in reality what happens is we first put it in a SMEM buffer, and then invoke TMA to asynchronously perform the store from SMEM to HBM. It’s obvious that this buffer has a short lifespan, and we reuse this buffer to temporarily store q after computing ln\_q but before needing it for the second residual connection.
- **Leverage Tensor Memory and TensorCore’s accumulate:** Notice how ln\_q is added immediately to the result of a matmul p @ v. Instead of keeping ln\_q in SMEM and reading it out after the matmul is done, notice how this is exactly the [MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma) semantic supported by tcgen05 in TMEM. Therefore, we can directly keep ln\_q in the TMEM buffer allocated for p @ v, and offload the ln\_q addition to TensorCore! This helps save both SMEM footprint and computation time.
- **Register subtiling:** Especially in a CUDA-core-heavy kernel like ours, registers are a scarcer resource than SMEM/TMEM. Besides careful register allocation tuning, we use register subtiling to mitigate the register pressure. We cut the tensor in SMEM/TMEM into chunks and load to registers one chunk at a time for normalization computation (both reduction and elementwise). This helps prevent register spilling which would cause significant performance degradation.

For pipeline stalls, we used the following techniques to improve pipeline efficiency:

- **Warp specialization:** In the original GDPA design, we have 4 main specialized warp partitions (load, mma, activation, and epilogue). In FlashNormAttention, we put the RMSNorm computation on the activation warp, while adding a fifth partition dedicated to prologue Layernorm computation, in order to better overlap it with TensorCore as well as other CUDA Core operations (e.g. RMSNorm computation for the previous iteration). The execution pipeline looks like the following. We use 8 warps (0-7) for the activation partition and 4 warps (8-11) for Layernorm. We maximize register allocation for the activation warps while limiting that for the Layernorm warps, using register subtiling to strike an optimal register-latency tradeoff. **![](https://pytorch.org/wp-content/uploads/2026/07/12-scaled.png)**

- **Register pre-loading:** A key factor stalling the execution pipeline is the complex data dependency introduced by residual connections. Both q and ln\_q need to be held in memory for a long time, blocking the prefetching and precomputing of these tensors for the next iteration. Since prefetching with TMA utilizes SMEM, Optimization #2 for memory pressure discussed above effectively moves ln\_q off the critical path. As for q, we let the activation warps preload q from SMEM as soon as it’s ready, and hold on to it until the very end where the second residual connection happens. By doing this, we can immediately free up the SMEM for q so that the next iteration’s q can be loaded as we work on the current iteration. As we allocate a maximum number of registers to the activation warps, the register pressure is still fine.

Below we present the benchmark results for the typical GDPA shapes we use. K/V are dense sequences with length exactly 128. Q is a sparse sequence with average sparsity 0.5 and varied max lengths. The batch size is 768. Head dimension is set to 128, and the number of heads is also varied to reflect the different performance behaviors under different normalization dimensions and different CTA cluster size. Due to the complex fusion, we present the latency saving as a percentage of the total baseline latency instead of just the normalization / elementwise kernels’ latency. The baseline is taken with inductor compilation.

### ![](https://pytorch.org/wp-content/uploads/2026/07/13.png)

### Backward implementation

As for backward, we skip the detailed algorithm for brevity, and only note the interesting optimizations and similarities/differences compared to forward.

- **Algorithm:** First notice that the backward fusion very much resembles forward, because the backward of a reduction op like normalization is also a reduction op, and the backward of a residual connection is also a residual connection
- **Recompute:** The original GDPA backward kernel recomputes q@k to save memory IO for forward. We follow the same idea for our fusion. First, the q@k becomes ln(q)@k in our kernel, so we recompute ln(q) first. We avoid incurring DSMEM overhead here by storing mean and variance in the forward (which is cheap because they are 1D) so that backward can use them to derive ln(q) easily. Second, we need the output of rmsnorm for its backward computation, which we recompute by rmsnorm\_out = kernel\_out – q. Lastly, we need the input of rmsnorm as well for residual connection backward, which we recompute by rmsnorm\_in = rmsnorm\_out / rstd.
- **Warp specialization:** The original GDPA backward kernel has 4 warp partitions: mma, activation, load, and reduction (for atomic-adding dQ). We use the same structure and put all new computation except Layernorm backward in the activation partition. We put the Layernorm backward computation in the reduction partition to facilitate better pipelining (similar to how we put the prologue Layernorm in a separate partition in forward).
- **Memory pressure:** Backward faces an even more severe memory pressure than forward, simply due to how much more data it needs to store and compute. For mitigation we applied aggressive SMEM/TMEM reuse similar to forward, but also had to shrink the tile sizes.
- **Pipeline efficiency:** Similar to forward, the long lifecycle of the residuals blocks the pipelining. For the inner residual, we can still use TMEM and TensorCore accumulate to break the lifecycle. But for the outer residual, we can’t afford register preloading in backward due to the higher pressure. Instead our approach is to use [TMA\_REDUCE\_ADD](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp#L1278) to directly add the residual from SMEM to HBM as soon as it’s ready.

  - It may seem counterintuitive at first to do this because the whole point of FlashNormAttention is to reduce memory IO, but this actually makes good sense. With the fusion the memory IO is no longer the bottleneck of the kernel, so we are happy to go back and trade a little more memory IO for mitigating the current bottleneck – the computation and pipeline stalls.

The chart below shows the performance improvement of the backward fusion. The exact same shapes as in forward are used.

## ![](https://pytorch.org/wp-content/uploads/2026/07/14.png)

## Acknowledgements

We thank Tri Dao, Markus Hoehnerbach, Jay Shah, Ted Zadouri, Vijay Thakkar, Wentao Guo for their open-source work in [Flash Attention](https://github.com/Dao-AILab/flash-attention) and [Quack](https://github.com/Dao-AILab/quack), which laid the foundation and provided inspiration for many optimization techniques discussed in this blog. We thank the Pytorch and Triton teams for their development and maintenance of [Helion](https://github.com/pytorch/helion) and [TLX](https://github.com/facebookexperimental/triton/tree/main), which made the exploration of the presented ideas efficient and fruitful.

## References

\[1\] Generalized Dot-Product Attention: Tackling Real-World Challenges in GPU Training Kernels. [https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/](https://pytorch.org/blog/generalized-dot-product-attention-tackling-real-world-challenges-in-gpu-training-kernels/)

\[2\] Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design. [https://arxiv.org/abs/2602.10016](https://arxiv.org/abs/2602.10016)

\[3\] Meta’s Generative Ads Model (GEM): The Central Brain Accelerating Ads Recommendation AI Innovation. [https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)

\[4\] Quack: Getting Memory-bound Kernels to Speed-of-Light. [https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)

\[5\] Triton Tutorials: 03 Matrix Multiplication. [https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

\[6\] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. [https://arxiv.org/pdf/2407.08608](https://arxiv.org/pdf/2407.08608)

\[7\] Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B. [https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)

### Docs

Access comprehensive developer documentation for PyTorch

[View Docs ›](https://pytorch.org/docs)

### Tutorials

Get in-depth tutorials for beginners and advanced developers

[View Tutorials ›](https://pytorch.org/tutorials)

### Resources

Find development resources and get your questions answered

[View Resources ›](https://pytorch.org/resources)

## **Stay in touch** for updates, event info, and the latest news

Select Country\*AfghanistanÅland IslandsAlbaniaAlgeriaAmerican SamoaAndorraAngolaAnguillaAntarcticaAntigua and BarbudaArgentinaArmeniaArubaAsia/Pacific RegionAustraliaAustriaAzerbaijanBahamasBahrainBangladeshBarbadosBelarusBelgiumBelizeBeninBermudaBhutanBoliviaBosnia and HerzegovinaBotswanaBouvet IslandBrazilBritish Indian Ocean TerritoryBritish Virgin IslandsBruneiBulgariaBurkina FasoBurundiCambodiaCameroonCanadaCanary IslandsCape VerdeCaribbean NetherlandsCayman IslandsCentral African RepublicChadChileChinaChristmas IslandCocos (Keeling) IslandsColombiaComorosCongoCook IslandsCosta RicaCote d'IvoireCroatiaCubaCuraçaoCyprusCzech RepublicDemocratic Republic of the CongoDenmarkDjiboutiDominicaDominican RepublicEast TimorEcuadorEgyptEl SalvadorEquatorial GuineaEritreaEstoniaEthiopiaEuropeFalkland IslandsFaroe IslandsFijiFinlandFranceFrench GuianaFrench PolynesiaFrench Southern and Antarctic LandsGabonGambiaGeorgiaGermanyGhanaGibraltarGreeceGreenlandGrenadaGuadeloupeGuamGuatemalaGuernseyGuineaGuinea-BissauGuyanaHaitiHeard Island and McDonald IslandsHondurasHong KongHungaryIcelandIndiaIndonesiaIranIraqIrelandIsle of ManIsraelItalyJamaicaJapanJerseyJordanKazakhstanKenyaKiribatiKosovoKuwaitKyrgyzstanLaosLatviaLebanonLesothoLiberiaLibyaLiechtensteinLithuaniaLuxembourgMacauMacedonia (FYROM)MadagascarMalawiMalaysiaMaldivesMaliMaltaMarshall IslandsMartiniqueMauritaniaMauritiusMayotteMexicoMicronesiaMoldovaMonacoMongoliaMontenegroMontserratMoroccoMozambiqueMyanmar (Burma)NamibiaNauruNepalNetherlandsNetherlands AntillesNew CaledoniaNew ZealandNicaraguaNigerNigeriaNiueNorfolk IslandNorth KoreaNorthern Mariana IslandsNorwayOmanPakistanPalauPalestinePanamaPapua New GuineaParaguayPeruPhilippinesPitcairn IslandsPolandPortugalPuerto RicoQatarRéunionRomaniaRussiaRwandaSaint BarthélemySaint HelenaSaint Kitts and NevisSaint LuciaSaint MartinSaint Pierre and MiquelonSaint Vincent and the GrenadinesSamoaSan MarinoSao Tome and PrincipeSaudi ArabiaSenegalSerbiaSeychellesSierra LeoneSingaporeSint MaartenSlovakiaSloveniaSolomon IslandsSomaliaSouth AfricaSouth Georgia and the South Sandwich IslandsSouth KoreaSouth SudanSpainSri LankaSudanSurinameSvalbard and Jan MayenSwazilandSwedenSwitzerlandSyriaTaiwanTajikistanTanzaniaThailandTogoTokelauTongaTrinidad and TobagoTunisiaTürkiyeTurkmenistanTurks and Caicos IslandsTuvaluU.S. Virgin IslandsUgandaUkraineUnited Arab EmiratesUnited KingdomUnited StatesUnited States Minor Outlying IslandsUruguayUzbekistanVanuatuVatican CityVenezuelaVietnamWallis and FutunaWestern SaharaYemenZambiaZimbabwe

By submitting this form, I consent to receive marketing emails from the LF and its projects regarding their events, training, research, developments, and related announcements. I understand that I can unsubscribe at any time using the links in the footers of the emails I receive. [Privacy Policy](https://www.linuxfoundation.org/legal/privacy-policy)

By submitting this form, I consent to receive marketing emails from the LF and its projects regarding their events, training, research, developments, and related announcements. I understand that I can unsubscribe at any time using the links in the footers of the emails I receive. [Privacy Policy](https://www.linuxfoundation.org/privacy/).

© 2026 PyTorch. Copyright © The Linux Foundation®. All rights reserved. The Linux Foundation has registered trademarks and uses trademarks. For more information, including terms of use, privacy policy, and trademark usage, please see our [Policies](https://www.linuxfoundation.org/legal/policies) page. [Trademark Usage](https://www.linuxfoundation.org/trademark-usage). [Privacy Policy](http://www.linuxfoundation.org/privacy).

[Close Menu](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#)

[Menu](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#slide-out-widget-area)

[Back to top](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/#)