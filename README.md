# Code for HE_MLPO

PyTorch implementation on: Multilevel Prototype Constraints Based on Hyperbolic Space for EEG Auditory
Attention Decoding

## Introduction
This paper proposes a **Hyperbolic Embedding-based Multi-Level Prototype Optimization** strategy (**HE-MLPO**), which maps multi-level features of EEG signals into a hyperbolic space and leverages its hierarchical geometric properties to model neural activity patterns. By introducing a learnable hierarchical prototype parameterization mechanism, the proposed strategy dynamically adjusts the positions of prototypes and feature embeddings, reducing the explicit reliance on mean statistics of labeled data while fusing spatio-temporal features. As a result, HE-MLPO effectively captures latent hierarchical structural differences in EEG signals and significantly improves auditory attention decoding performance in noisy environments.

<p>
  <image src="figure/main_figure.png">
</p >
