# DREAM: Domain Invariant and Contrastive Representation for Sleep Dynamics
This repository contains the official PyTorch implementation of the following paper: 

> **DREAM: Domain Invariant and Contrastive Representation for Sleep Dynamics**
> 
> **abstract:** _Sleep staging is a key challenge in diagnosing and treating sleep-related diseases due to its labor-intensive, time-consuming, costly, and error-prone. With the availability of large-scale sleep signal data, many deep learning methods are proposed for automatic sleep staging. However, these existing methods face several challenges including the heterogeneity of patients' underlying health conditions; the infeasibility of extracting meaningful information from unlabeled sleep signal data to improve prediction performances; the difficulty modeling complex interactions between sleep stages; and the lack of an effective mechanism in clinical contexts that allows for human intervention when needed. In this paper, we propose a neural network architecture named DREAM to tackle these issues for automatic sleep staging. DREAM consists of (i) a feature representation network that generates robust representations for sleep signals via the variational auto-encoder framework and contrastive learning and (ii) a sleep stage classification network that explicitly models the interactions between sleep stages in the sequential context at both feature representation and label classification levels via Transformer and conditional random field architectures. Our experimental results indicate that DREAM significantly outperforms existing methods for automatic sleep staging on three sleep signal datasets. Further, DREAM provides an effective mechanism for quantifying uncertainty measures for its predictions, thereby helping sleep specialists intervene in cases of highly uncertain predictions, resulting in better diagnoses and treatments for patients in real clinical settings._

# Framework

The proposed framework **DREAM** consists of two main networks: **(i) Feature representation network** that transforms the input sleep segment into sleep-relevant and subject-invariant feature representations based on the VAE framework and **(ii) Sleep Stage Classification network** that captures the dependencies between sleep segments in the sequential context to find the best corresponding sleep stage sequence. 

In the first stage, the feature representation network is trained to capture the sleep-relevant and subject-invariant representation from one input sleep segment. In the second stage, the trained feature representation network with fixed weights is employed to generate the sequence of the representations for all segments of the input sequence, which is then used as input for the classification network to identify the corresponding sleep stage sequence.

![DREAM](https://user-images.githubusercontent.com/107287907/173477720-540c4f92-54c5-42a5-a4ae-ff2d1cc53e93.png)

# Installation
Used modules: numpy, scipy, pandas, scikit-learn, TorchCRF, pytorch_metric_learning, and PyTorch (CUDA toolkit if use GPU). 

    $ conda create -n DREAM python=3.9.7
    $ conda activate DREAM
    $ conda install scipy pandas scikit-learn TorchCRF pytorch_metric_learning
    $ conda install numpy=1.21.2
    $ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Usage

## Data
Run the following scripts to download and preprocess SleepEDF-20 dataset
    $ cd data/edf_20
    $ bash download_edf20.sh
    $ cd ../..
    $ python preprocess_edf.py
    
##  Training DREAM
Run the following script to train DREAM

    $ batch job_batch.txt 

