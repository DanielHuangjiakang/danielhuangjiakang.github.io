---
title: "Deep Dive into speedup_by_fusion in PyTorch Inductor"
date: 2026-03-31
permalink: /blog/speedup-by-fusion-pytorch-inductor/
excerpt: "A benchmark-driven analysis of PyTorch Inductor's speedup_by_fusion config, its runtime logs, and why register spilling can reject fusions that still help."
author_line: "Jiakang Huang"
translation_key: "speedup-by-fusion-pytorch-inductor"
lang: en
related: false
show_post_navigation: false
header:
  teaser: speedup-by-fusion-cover.png
---

**Author:** Jiakang Huang

<figure class="post-feature-image">
  <img src="/images/speedup-by-fusion-cover.png" alt="Cover illustration for speedup_by_fusion in PyTorch Inductor">
</figure>

In the previous post, we walked through the overall architecture of `fuse_nodes` in PyTorch Inductor (see: [Deep Dive into fuse_nodes in PyTorch Inductor](/blog/fuse-nodes-pytorch-inductor/)). Today we zoom in on a particularly interesting configuration within the fusion pipeline: `speedup_by_fusion` (exposed as `benchmark_fusion`). We will cover how to enable it, what it does under the hood, what the logs look like, and a critical limitation around register spilling that may lead to suboptimal fusion decisions.

## 1. Enabling benchmark_fusion

`benchmark_fusion` is a config flag in `torch._inductor.config`. When turned on, Inductor uses actual GPU benchmarks—rather than heuristics alone—to decide whether fusing two operators is worthwhile.

You can enable it in two ways:

```python
import torch._inductor.config as config
config.benchmark_fusion = True
```

Or via the `options` dict passed to `torch.compile`:

```python
compiled_model = torch.compile(model, backend="inductor", options={"benchmark_fusion": True})
```

## 2. What Happens When It Is Enabled

In default mode, Inductor's fusion decisions are purely heuristic: `can_fuse` checks legality, `score_fusion` ranks candidates, and fusions are applied greedily.

With `benchmark_fusion` enabled, an additional step is inserted: **each candidate fusion pair is actually benchmarked on the GPU**. The system times:

- The **separate execution** of both operators
- The **fused execution** as a single kernel

A fusion is only committed if the fused kernel is measurably faster.

## 3. What the Logs Look Like

With benchmark fusion enabled, the Inductor fusion log emits entries like:

```
V0312 02:40:20.816000 3795204 scheduler.py:4396] [0/0] [__fusion]
  can fuse (benchmark): fusing OrderedSet(['buf17']) with OrderedSet(['buf18'])
  cause 2.462x speedup
```

This tells us that `buf17` and `buf18` were actually benchmarked, and the fused kernel ran **2.462x faster**, so the fusion was accepted.

## 4. Limitations and the Register Spilling Problem

While benchmark-driven fusion sounds strictly better than heuristics, there are two issues worth examining.

### 4.1 Greedy Fusion Is Not Globally Optimal

The benchmark evaluates a **single pair** of operators in isolation. Even if fusing A and B is locally faster, it does not guarantee that the resulting full graph is globally optimal. This is an inherent limitation of greedy algorithms: a sequence of locally optimal decisions may not compose into a globally optimal solution.

### 4.2 Register Spilling Causes Premature Rejection

During benchmarking, the fused kernel may trigger **register spilling**, at which point the current implementation immediately rejects the fusion without measuring the actual performance impact. Here is an example from the logs:

```
V0312 02:40:31.500000 3795204 scheduler.py:1776] [0/0] [__fusion]
  cannot fuse op1_op6_op11_op2_op7_op12 with op16_op17_op18:
  register spilling of the fused kernel
```

**What is register spilling?** Each GPU thread has a limited number of registers. When a kernel requires more registers than the hardware provides per thread, the excess variables are "spilled" to local memory, which resides in much slower off-chip storage. This increases memory traffic and can degrade performance significantly.

The current implementation treats register spilling as a hard rejection signal. But this raises an important question:

> **Could the reduction in kernel launch overhead from fusion outweigh the performance cost of register spilling?**

In other words, the current policy may be too conservative, rejecting fusions that would still be net beneficial despite some spilling.

## 5. Experimental Results

To investigate, I ran a controlled experiment on an **RTX 5090** with PyTorch 2.10.0+cu128, comparing two settings on one synthetic workload:

- **benchmark_fusion_0**: benchmark fusion **off** (heuristic-only)
- **benchmark_fusion_1**: benchmark fusion **on**

### Model 20: HubConflictRoundOpt

A synthetic model with a shared hub tensor feeding six competing branches, mixing reductions across different axes and transcendental operations (`tanh`, `sin*cos`, `relu`).

| Metric | Off (fusion_0) | On (fusion_1) | Change |
|--------|---------------|--------------|--------|
| Compiled runtime (ms) | 0.817 | 0.964 | +17.9% (slower) |
| Eager runtime (ms) | 79.36 | 60.10 | -24.3% |
| Compiled speedup vs Eager | 97.1x | 62.4x | -35.8% |
| FX compile time (s) | 7.22 | 20.00 | +176.8% |
| Fusion rounds | 3 | 2 | -1 |
| Net node reduction | 67 | 62 | -5 |
| Benchmark decisions | 0 | 62 | +62 |

### Analysis

After reviewing the full logs for this workload, I can confirm that **every fusion rejected in the benchmark-on run was rejected due to register spilling**—not because the benchmark showed a slowdown.

For this model, turning on `benchmark_fusion` reduced the number of fusion rounds, reduced net node elimination, and made the compiled runtime **17.9% slower**. That pattern suggests that **the extra launch overhead from keeping more kernels separate (due to spilling-based rejections) outweighed the cost of the spilling that the fused kernels might have incurred.**

The compile-time cost is also substantial: FX compile time increased from **7.22s to 20.00s** (**+176.8%**), since each candidate pair has to be compiled and profiled on the GPU.

## 6. Discussion

Using real benchmarks to validate fusion decisions is a smart idea—it replaces speculation with measurement. However, the current handling of register spilling is arguably too blunt: **spilling is treated as a hard veto, bypassing the benchmark entirely.**

This single workload already suggests that the policy may be overly conservative. A more nuanced approach would be to let the benchmark run even when spilling is detected, and let the actual timing data determine whether the fusion is worthwhile. After all, the severity of register spilling depends heavily on the amount of spilling and memory access patterns—not all spilling leads to unacceptable performance degradation.

I am not fully familiar with the internals of the benchmark implementation, and there may well be better ways to evaluate pre- and post-fusion performance. If you have thoughts or ideas on this topic, I would love to hear from you—feel free to reach out by email.
