<div align="center">


# Awesome-LM-Architecture

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Last Commit](https://img.shields.io/github/last-commit/Superposition09m/Awesome-LM-Architecture)](https://github.com/Superposition09m/Awesome-LM-Architecture)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()
[![GitHub stars](https://img.shields.io/github/stars/Superposition09m/Awesome-LM-Architecture?style=social)](https://github.com/Superposition09m/Awesome-LM-Architecture/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Superposition09m/Awesome-LM-Architecture?style=social)](https://github.com/Superposition09m/Awesome-LM-Architecture/watchers)
</div>

<div align="center">
<h1>🥳Introduction</h1>
</div>

This is a curated collection of LLM architectures. Our approach is **Architecture-First**, but we believe a model is more than its primary layers.

We systematically collect and deconstruct *not only the main model architecture* but also *the crucial components*, such as optimizers, positional encodings, and normalization schemes, that define a model's success. 

Designing new generations of LLMs needs a comprehensive view of the landscape and insights from other domains. So we also gather cross-domain research that inspires the next generation of LM design.

<!-- <div align="center">
<h1>📙Table of Contents</h1>
</div> -->

<div align="center">
<h1>📃Collection</h1>
</div>

# Main Arch

## Full Attention Improvements
### Efficiency

> FlashAttention-1,2,3

![](https://img.shields.io/badge/arXiv-2022.05-red) [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

![](https://img.shields.io/badge/arXiv-2023.07-red) [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)


![](https://img.shields.io/badge/arXiv-2024.07-red) [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)

>MHA->MQA->GQA->MLA

![](https://img.shields.io/badge/arXiv-2017.06-red) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

![](https://img.shields.io/badge/arXiv-2019.11-red) [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

![](https://img.shields.io/badge/arXiv-2023.05-red) [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

![](https://img.shields.io/badge/arXiv-2024.05-red) [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)


### Stability

![](https://img.shields.io/badge/arXiv-2025.05-red) [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)

## Sparse Attention

![](https://img.shields.io/badge/arXiv-2025.02-red) [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)

![](https://img.shields.io/badge/arXiv-2025.02-red) [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)

![](https://img.shields.io/badge/arXiv-2025.12-red) [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)

## Linear Model

![](https://img.shields.io/badge/arXiv-2020.06-red) [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)

![](https://img.shields.io/badge/arXiv-2022.10-red) [The Devil in Linear Transformer](https://arxiv.org/abs/2210.10340)

![](https://img.shields.io/badge/arXiv-2023.12-red) [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)

![](https://img.shields.io/badge/arXiv-2024.06-red) [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)

![](https://img.shields.io/badge/arXiv-2024.12-red) [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)

![](https://img.shields.io/badge/arXiv-2025.10-red) [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)



Mamba Family

![](https://img.shields.io/badge/arXiv-2023.12-red) [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752)

![](https://img.shields.io/badge/arXiv-2024.05-red) [Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality](https://arxiv.org/pdf/2405.21060)

![](https://img.shields.io/badge/openreview-2025.09-yellow) [MAMBA-3: IMPROVED SEQUENCE MODELING USING
STATE SPACE PRINCIPLES](https://openreview.net/pdf?id=HwCvaJOiCj)


## Test Time Learning Family

![](https://img.shields.io/badge/arXiv-2025.01-red) [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

![](https://img.shields.io/badge/arXiv-2025.05-red) [ATLAS: Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735)

![](https://img.shields.io/badge/arXiv-2025.12-red) [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)


## MoE

![](https://img.shields.io/badge/arXiv-2024.01-red) [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)

![](https://img.shields.io/badge/arXiv-2025.01-red) [Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models](https://arxiv.org/abs/2501.11873)

### Other New Architectures

![](https://img.shields.io/badge/arXiv-2025.06-red) [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)

![](https://img.shields.io/badge/arXiv-2026.01-red) [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/pdf/2601.07372v1)


# Other Critical Components
## Optimizer
> We believe **The optimizer is the dual of the architecture**. They are not independent.

![](https://img.shields.io/badge/arXiv-2014.12-red) [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)  

![](https://img.shields.io/badge/arXiv-2017.11-red) [Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101)  

![](https://img.shields.io/badge/blog-2024.12-yellow) [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)  

![](https://img.shields.io/badge/arXiv-2025.02-red) [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)  

![](https://img.shields.io/badge/arXiv-2022.03-red) [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)

## Position Embedding

![](https://img.shields.io/badge/arXiv-2023.05-red) [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/pdf/2305.19466)

![](https://img.shields.io/badge/arXiv-2021.04-red) [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)  


![](https://img.shields.io/badge/arXiv-2024.05-red) [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/abs/2405.18719)  

![](https://img.shields.io/badge/arXiv-2021.08-red) [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)

![](https://img.shields.io/badge/arXiv-2025.09-red) [Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings](https://arxiv.org/abs/2509.10534)

![](https://img.shields.io/badge/arXiv-2023.09-red) [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)

## Normalization

![](https://img.shields.io/badge/arXiv-2016.07-red) [Layer Normalization](https://arxiv.org/abs/1607.06450)

![](https://img.shields.io/badge/arXiv-2020.03-red) [Root Mean Square Layer Normalization](https://arxiv.org/abs/2003.02439)

![](https://img.shields.io/badge/arXiv-2025.03-red) [Transformers without Normalization](https://arxiv.org/abs/2503.10622)

## Residual Connection and its Improvements

![](https://img.shields.io/badge/arXiv-2015.12-red) [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

![](https://img.shields.io/badge/arXiv-2024.09-red) [Hyper-Connections](https://arxiv.org/abs/2409.19606)

![](https://img.shields.io/badge/arXiv-2025.12-red) [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)

## Tokenizer

![](https://img.shields.io/badge/arXiv-2015.08-red) [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

## Convolution
![](https://img.shields.io/badge/arXiv-2022.12-red) [Hungry Hungry Hippos: Towards Language Modeling with State
Space Models](https://arxiv.org/pdf/2212.14052)

![](https://img.shields.io/badge/arXiv-2023.02-red) [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866)

![](https://img.shields.io/badge/blog-2025-yellow) [Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality](https://physics.allen-zhu.com/part-4-architecture-design/part-4-2)


## Scaling & Pretraining

![](https://img.shields.io/badge/arXiv-2001.08361-red) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

![](https://img.shields.io/badge/arXiv-2022.03-red) [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)


# Awesome Technical Reports of Models

![](https://img.shields.io/badge/arXiv-2025.01-red) [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

![](https://img.shields.io/badge/arXiv-2025.07-red) [Kimi K2: Open Agentic Intelligence](https://arxiv.org/pdf/2507.20534)  

![](https://img.shields.io/badge/blog-2025.09-yellow) [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d&from=home.latest-research-list)


![](https://img.shields.io/badge/arXiv-2025.12-red) [Olmo 3](https://arxiv.org/abs/2512.13961)

# Related Research in other Domains/sub-domains that are related/inspiring to LM Architecture

![](https://img.shields.io/badge/blog-2024.7-yellow) [Physics of Language Models](https://physics.allen-zhu.com/)

![](https://img.shields.io/badge/arXiv-2025.11-red) [Weight-sparse transformers have interpretable circuits](https://arxiv.org/abs/2511.13653)


<div align="center"> 
<h1>📈 Star History</h1>
</div>



[![Star History Chart](https://api.star-history.com/svg?repos=Superposition09m/Awesome-LM-Architecture&type=Date)](https://star-history.com/#Superposition09m/Awesome-LM-Architecture&Date)

