# 3D Bayes: Bayesian Optimisation with Equivariant Graph Neural Networks

**[Paper](TBD)** | **[OpenReview](TBD)** | **[Poster](TBD)**

This repository contains the PyTorch implementation of the work ...


## Content ##
0. [Environment Setup](#environment-setup)
0. [Training](#training)
0. [File Structure](#file-structure)
0. [Checkpoints](#checkpoints)
0. [Citation](#citation)
0. [Acknowledgement](#acknowledgement)



## Environment Setup ##


### Environment 

Set up environment instructions ...


## Training ##


### OC20

1. We train on the QM9 dataset by running:
    
    ```bash
        sh SimpleBayes.sh
    ```
    The above script uses X nodes with  GPUs on each node.


## File Structure ##

1. [`nets`](nets) includes code of different network architectures for the Equiformer_v2 architecture and its variational implementation.
2. [`main_QM9.py`](main_QM9.py) is the code for training, evaluating and running relaxation.


## Checkpoints ##

We provide the checkpoints of the model ...

## Citation ##

...

## Acknowledgement ##

...