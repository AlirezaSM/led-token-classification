# Token Classification Label Error Detection Benchmark

## Overview

This project is an implementation inspired by the paper ["Detecting Label Errors in Token Classification Data."](https://arxiv.org/abs/2210.03920) The original paper introduced methods for detecting label errors in the [CoNLL2003](https://paperswithcode.com/dataset/conll-2003) dataset with the help of the verified labels version [CoNLL++](https://github.com/ZihanWangKi/CrossWeigh/tree/master/data), designed for Named Entity Recognition tasks. In this implementation, the repository from the paper was forked, and four new methods named `median_quality`, `Exponential Moving Average`, `Culmulative Average of Bottom Scores` and `Exponential Distribution` were added to enhance label error detection. While some of the new methods outperformed existing ones, the strongest method from the original paper remained a challenging benchmark, although earnest attempts were made to surpass it.

## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Methods

The project includes implementations of various methods for aggregating token quality scores as a sentence quality score:
- **Median Quality:** 
$$
s_i^*=\underset{k}{\operatorname{median}}\left\{s_i^k\right\}
$$

- **Exponential Moving Average (EMA):** 

  $$
  s_i^* = S_i^K, \text{ where } S_i^t = \begin{cases}
    \check{s}_i^1 & \text{ for } t=1 \\
    \alpha \cdot \breve{s}_i^t + (1-\alpha) \cdot S_i^{t-1} & \text{ for } t>1
  \end{cases}$$

- **Cumulative Average of Bottom Scores (CABS):**
$$
s_i^* = \frac{1}{J} \sum_{k=1}^J \hat{s}_i^k
$$

- **Exponential Distribution:**
$$
s_i^* = \frac{\sum_{k=1}^K s_i^k \cdot \exp(- \lambda s_i^k)}{\sum_{k=1}^K \exp(- \lambda s_i^k)}
$$


## Usage
Most of the implementation is found in the Jupyter notebook named `token-classification-benchmark.ipynb`. The notebook is designed to run seamlessly on Google Colab, providing a convenient platform for users to reproduce experiments and explore the label error detection methods.

## Results

The results of the label error detection methods are presented in the notebook, including `Lift score`, `AUROC` and `AUPRC`. A comparative analysis is provided, highlighting the strengths and limitations of each method, with a focus on the efforts to outperform the strongest method introduced in the original paper.

## Dependencies

The project ran with the following dependencies:

- Python 3.10.12
- NumPy 1.23.5
- Scikit-Learn 1.2.2
- transformers 4.35.2
- cleanlab 2.1.1

Ensure that install the cleanlab library, provided locally in the repository by:
```
pip install ./cleanlab
```

## Acknowledgments

This implementation is based on the paper ["Detecting Label Errors in Token Classification Data."](https://arxiv.org/abs/2210.03920) We appreciate the contributions of the paper's authors and the foundation provided by their work. Special thanks to the original repository, which served as the starting point for this project.

---
