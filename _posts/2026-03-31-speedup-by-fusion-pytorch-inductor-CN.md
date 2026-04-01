---
title: "PyTorch Inductor 中 speedup_by_fusion 深度解析"
date: 2026-03-31
permalink: /zh/blog/speedup-by-fusion-pytorch-inductor/
excerpt: "详解 PyTorch Inductor 的 speedup_by_fusion 配置：开启方式、工作原理、benchmark 日志示例，以及 register spilling 带来的融合决策争议。"
author_line: "Jiakang Huang"
translation_key: "speedup-by-fusion-pytorch-inductor"
lang: zh
related: false
show_post_navigation: false
header:
  teaser: speedup-by-fusion-cover.png
---

**作者：** Jiakang Huang

<figure class="post-feature-image">
  <img src="/images/speedup-by-fusion-cover.png" alt="PyTorch Inductor 中 speedup_by_fusion 的封面图">
</figure>

上篇文章我们分析了 Inductor 中 `fuse_nodes` 的整体架构和工作流程（详见：[PyTorch Inductor 中 fuse_nodes 融合流程深度解析](/zh/blog/fuse-nodes-pytorch-inductor/)）。本篇我们将聚焦其中一个有趣的配置项 `speedup_by_fusion`，从开启方式、运行机制、实际日志到局限性展开讨论。

## 1. 如何开启 speedup_by_fusion

`speedup_by_fusion` 是 `torch._inductor.config` 中的一个配置项。开启后，Inductor 在融合决策阶段会通过实际 benchmark 来判断两个算子融合后是否真的更快，而不仅仅依赖启发式打分。

可以通过以下方式开启：

```python
import torch._inductor.config as config
config.benchmark_fusion = True
```

或者通过 `torch.compile` 的 `options` 参数传入：

```python
compiled_model = torch.compile(model, backend="inductor", options={"benchmark_fusion": True})
```

## 2. 开启后做了什么

在默认模式下，Inductor 的融合决策完全基于启发式规则——通过 `can_fuse` 检查合法性，通过 `score_fusion` 打分排序，然后贪心地执行融合。

开启 `benchmark_fusion` 后，流程增加了一个关键步骤：**对候选融合对进行实际 GPU benchmark**。具体来说，系统会分别计时：

- 两个算子**独立运行**的总耗时
- 两个算子**融合后**作为一个 kernel 的耗时

只有当融合后确实更快时，才执行该融合。

## 3. 日志中的 Speedup 示例

开启后，在 Inductor 的 fusion 日志中可以看到类似如下的输出：

```
V0312 02:40:20.816000 3795204 scheduler.py:4396] [0/0] [__fusion]
  can fuse (benchmark): fusing OrderedSet(['buf17']) with OrderedSet(['buf18'])
  cause 2.462x speedup
```

这条日志表明 `buf17` 和 `buf18` 经过实际 benchmark 测试后，融合带来了 **2.462 倍**的加速，因此决定执行融合。

## 4. 局限性与 Register Spilling 问题

开启 `speedup_by_fusion` 虽然看起来更加"科学"，但实际使用中存在两个值得讨论的问题。

### 4.1 贪心融合的全局最优性问题

benchmark 测试的是**两个算子**融合前后的性能对比。但这个局部最优并不一定意味着全图在 GPU 上运行时也是最优的。贪心算法的固有缺陷在于：局部最优决策的累积不一定导向全局最优。

### 4.2 Register Spilling 导致的融合拒绝

在实际 benchmark 过程中，可能出现融合后的 kernel 因为 **register spilling** 而被拒绝融合的情况。日志示例如下：

```
V0312 02:40:31.500000 3795204 scheduler.py:1776] [0/0] [__fusion]
  cannot fuse op1_op6_op11_op2_op7_op12 with op16_op17_op18:
  register spilling of the fused kernel
```

**什么是 Register Spilling？** GPU 的每个线程有有限数量的寄存器。当一个 kernel 需要的寄存器数量超出硬件限制时，多余的变量会被"溢出"到较慢的 local memory 中。这就是 register spilling。它会导致显著的性能下降，因为 local memory 的访问延迟远高于寄存器访问。

当前实现中，一旦检测到 register spilling，就**直接拒绝该融合**，不再进一步评估。这带来了一个重要疑问：

> **即使发生了 register spilling，融合带来的 launch overhead 减少是否有可能超过 spilling 的性能损失？**

换句话说，当前的实现可能因为 register spilling 而过于保守地拒绝了一些实际上有益的融合。

## 5. 实验数据

为验证上述假设，我在 **RTX 5090** 上基于一个合成 workload 做了对比实验。实验环境为 PyTorch 2.10.0+cu128。

- **benchmark_fusion_0**：关闭 `benchmark_fusion`（纯启发式）
- **benchmark_fusion_1**：开启 `benchmark_fusion`

### 模型 20：HubConflictRoundOpt

该模型具有共享 hub tensor 和多分支竞争结构，包含多种 reduction 和 transcendental 运算。

| 指标                | 关闭 (fusion_0) | 开启 (fusion_1) | 变化          |
| ------------------- | --------------- | --------------- | ------------- |
| 编译后运行时间 (ms) | 0.817           | 0.964           | +17.9% (变慢) |
| Eager 运行时间 (ms) | 79.36           | 60.10           | -24.3%        |
| 编译加速比 vs Eager | 97.1x           | 62.4x           | -35.8%        |
| FX 编译耗时 (s)     | 7.22            | 20.00           | +176.8%       |
| 融合轮数            | 3               | 2               | -1            |
| 节点缩减数          | 67              | 62              | -5            |
| Benchmark 决策次数  | 0               | 62              | +62           |

### 数据分析

查看该 workload 的完整日志后可以确认：**所有少融合的节点，都是因为开启 benchmark 后检测到 register spilling 而被拒绝的。**

在这个模型上，开启 `benchmark_fusion` 后，融合轮数减少、节点缩减数减少，最终编译后运行时间反而**变慢了 17.9%**。这说明在这个 workload 中，**因 register spilling 而少融合节点所带来的额外 launch overhead，很可能比融合后可能出现的 spilling 成本更大。**

更值得注意的是，开启 benchmark 后 FX 编译时间从 **7.22s** 增加到 **20.00s**，增幅约 **176.8%**，因为每个候选对都需要实际在 GPU 上跑一遍。

## 6. 思考

使用真实 benchmark 来决定两个节点是否应该融合，这无疑是一个聪明的做法——它直接用数据说话，避免了启发式规则可能的误判。

但当前对 register spilling 的处理方式过于简单粗暴：**一旦检测到 spilling，直接拒绝融合，不再进行 benchmark 评估。** 即使只看这一个 workload，这种策略也可能过于保守。

个人认为，即使出现了 register spilling，也应该继续运行 benchmark，让实际的运行数据来决定是否融合。毕竟 register spilling 的影响程度取决于溢出量和访问模式，并非所有 spilling 都会导致不可接受的性能下降。

当然，我对 benchmark 的具体实现方式了解有限，也许存在更好的方法来判断融合前后的性能差异。欢迎大家通过邮件与我讨论。
