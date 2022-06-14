# DREAM: Domain Invariant and Contrastive Representation for Sleep Dynamics
This repository contains the official PyTorch implementation of the following paper: \\

abstract:
Sleep staging is a key challenge in diagnosing and treating sleep-related diseases due to its labor-intensive, time-consuming, costly, and error-prone. With the availability of large-scale sleep signal data, many deep learning methods are proposed for automatic sleep staging. However, these existing methods face several challenges including the heterogeneity of patients' underlying health conditions; the infeasibility of extracting meaningful information from unlabeled sleep signal data to improve prediction performances; the difficulty modeling complex interactions between sleep stages; and the lack of an effective mechanism in clinical contexts that allows for human intervention when needed. In this paper, we propose a neural network architecture named DREAM to tackle these issues for automatic sleep staging. DREAM consists of (i) a feature representation network that generates robust representations for sleep signals via the variational auto-encoder framework and contrastive learning and (ii) a sleep stage classification network that explicitly models the interactions between sleep stages in the sequential context at both feature representation and label classification levels via Transformer and conditional random field architectures. Our experimental results indicate that DREAM significantly outperforms existing methods for automatic sleep staging on three sleep signal datasets. Further, DREAM provides an effective mechanism for quantifying uncertainty measures for its predictions, thereby helping sleep specialists intervene in cases of highly uncertain predictions, resulting in better diagnoses and treatments for patients in real clinical settings.

# Framework

![DREAM](https://user-images.githubusercontent.com/107287907/173477720-540c4f92-54c5-42a5-a4ae-ff2d1cc53e93.png)
Figure 1: Overall architecture of DREAM


# Installation


# Usage

