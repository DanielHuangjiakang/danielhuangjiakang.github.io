---
published: false
title: "IntentEval: Evaluating Whether Large Language Models Answer What Users Actually Mean"
collection: publications
category: manuscripts
permalink: /publication/2026-05-19-intenteval
excerpt: "Under review at ACL ARR 2026 May Submission. IntentEval is a benchmark for evaluating whether LLMs answer the user's intended question, using human-derived probability distributions over interpretations and an intent-aware judge. It achieves 0.90 correlation with LMArena rankings."
date: 2026-05-19
venue: "ACL ARR 2026 May Submission (under review)"
venue_short: "ACL ARR 2026"
authors: "Yining Wang, Linquan Yuan, Chuqiao Lin, Renyi Cai, Xueyan Zhang, Jiakang Huang, Yiren Zhao, YiTian Ding, Jinman Zhao, Lei Li, Shinan Liu, Gerald Penn"
header:
  teaser: intenteval-cover.png
---

**Authors:** Yining Wang, Linquan Yuan, Chuqiao Lin, Renyi Cai, Xueyan Zhang, Jiakang Huang, Yiren Zhao, YiTian Ding, Jinman Zhao, Lei Li, Shinan Liu, Gerald Penn

**Status:** Under review at ACL ARR 2026 May Submission  
**Preferred venue:** EMNLP  
**Paper type:** Long paper  
**Keywords:** Benchmarking, LLM Evaluation

**Abstract:** Current chat benchmarks often use clear, well-specified questions with a single intended meaning. Real open-ended conversations are messier: user questions are often ambiguous, underspecified, or poorly posed, so a model must infer what the asker most likely meant before giving a useful answer. We introduce IntentEval, a benchmark for evaluating whether language models answer the user's intended question. Each instance contains a real user question, asker-provided context, possible interpretations, and a human-derived probability distribution over those interpretations. Responses are scored by an intent-aware judge that emphasizes dominant-intent alignment, answer quality, and length calibration, while penalizing verbose multi-interpretation hedging. Across closely matched frontier models, rerunning MT-Bench and AlpacaEval yields only moderate correlation with LMArena leaderboard rankings, around 0.6–0.7, suggesting reduced discriminative power among modern high-performing models. By contrast, IntentEval reaches a 0.90 correlation with the current rankings. Removing interpretation-space information and judging only general answer quality lowers the correlation to 0.83, showing that explicit intent modeling provides an evaluation signal beyond generic answer quality.
