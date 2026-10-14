---
title: "Syntactic Prediction through Reinforcement Learning"
collection: publications
category: conferences
permalink: /publication/2026-03-10-syntactic-prediction-through-reinforcement-learning
excerpt: "Accepted to EMNLP 2026 (Main Conference). Syntax-R1 is a two-stage post-training framework that uses reinforcement learning to improve hierarchical syntactic prediction in large language models, achieving state-of-the-art results on constituency parsing and CCG supertagging."
date: 2026-08-21
modified: 2026-08-21
venue: "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026), Main Conference"
venue_short: "EMNLP 2026 Main"
status: accepted
authors: "Jinman Zhao, Yining Wang, Jiahe Liu, Xueyan Zhang, Linbo Cao, Jiakang Huang, Yiren Zhao, Renyi Cai, YiTian Ding, Gerald Penn"
header:
  teaser: syntactic-prediction-rl-cover.svg
paperurl: "https://drive.google.com/file/d/1WCfqVSMVibQns_b9hOwuzeHq80Yi9SM2/view?usp=sharing"
---

**Paper:** [PDF](https://drive.google.com/file/d/1WCfqVSMVibQns_b9hOwuzeHq80Yi9SM2/view?usp=sharing)

**Accepted to EMNLP 2026 Main Conference.** This is my first paper.

**Authors:** Jinman Zhao, Yining Wang, Jiahe Liu, Xueyan Zhang, Linbo Cao, Jiakang Huang, Yiren Zhao, Renyi Cai, YiTian Ding, Gerald Penn

**Status:** Accepted (EMNLP 2026, Main Conference)  
**Venue:** EMNLP 2026 Main Conference  
**Paper type:** Long paper

**Abstract:** Syntactic prediction requires models to generate outputs that are globally consistent and satisfy hierarchical structural constraints. Although large language models have been applied to such tasks, existing approaches still rely mainly on prompting or supervised fine-tuning, and their performance generally remains below that of specialized neural syntactic models. Our experiments show that supervised fine-tuning reaches a plateau, and additional supervised training does not further improve prediction quality. We therefore propose a two-stage post-training framework for syntactic prediction. The first cold-start stage adapts the model to a unified syntax generation format, and then applies reinforcement learning on a disjoint data subset to directly optimize structure-level objectives. Experiments on constituency parsing and CCG supertagging show that Syntax-R1 consistently outperforms supervised LLM baselines and achieves state-of-the-art results. Further analyses suggest that its gains come primarily from the reinforcement learning stage. These findings identify reinforcement learning as an effective post-training mechanism for improving hierarchical syntactic prediction in large language models.

**Keywords:** LLM, syntactic prediction, parsing, supertag, syntax
