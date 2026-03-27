---
title: "Syntactic Prediction through Reinforcement Learning"
collection: publications
category: manuscripts
permalink: /publication/2026-03-10-syntactic-prediction-through-reinforcement-learning
excerpt: "Under review at ACL ARR 2026 March Submission. This is my first paper and it studies reinforcement learning for hierarchical syntactic prediction in large language models."
date: 2026-03-10
modified: 2026-03-17
venue: "ACL ARR 2026 March Submission (under review)"
citation: 'Jinman Zhao, Yining Wang, Jiahe Liu, Xueyan Zhang, Linbo Cao, Jiakang Huang, Yiren Zhao, Renyi Cai, YiTian Ding, Gerald Penn. &quot;Syntactic Prediction through Reinforcement Learning.&quot; Under review at <i>ACL ARR 2026 March Submission</i>, preferred venue EMNLP.'
---

This is my first paper and it is currently under review.

**Authors:** Jinman Zhao, Yining Wang, Jiahe Liu, Xueyan Zhang, Linbo Cao, Jiakang Huang, Yiren Zhao, Renyi Cai, YiTian Ding, Gerald Penn

**Status:** Under review at ACL ARR 2026 March Submission  
**Preferred venue:** EMNLP  
**Paper type:** Long paper

**Abstract:** Syntactic prediction requires models to generate outputs that are globally consistent and satisfy hierarchical structural constraints. Although large language models have been applied to such tasks, existing approaches still rely mainly on prompting or supervised fine-tuning, and their performance generally remains below that of specialized neural syntactic models. Our experiments show that supervised fine-tuning reaches a plateau, and additional supervised training does not further improve prediction quality. We therefore propose a two-stage post-training framework for syntactic prediction. The first cold-start stage adapts the model to a unified syntax generation format, and then applies reinforcement learning on a disjoint data subset to directly optimize structure-level objectives. Experiments on constituency parsing and CCG supertagging show that Syntax-R1 consistently outperforms supervised LLM baselines and achieves state-of-the-art results. Further analyses suggest that its gains come primarily from the reinforcement learning stage. These findings identify reinforcement learning as an effective post-training mechanism for improving hierarchical syntactic prediction in large language models.

**Keywords:** LLM, syntactic prediction, parsing, supertag, syntax
