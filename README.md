# Multi-spectral template matching based object detection in a few-shot learning manner

NOTE: DATA ARE BEING UPLOADED...

## Introduction
This repository is released for our code and dataset in [our Inf. Sci. 2023 paper](https://www.sciencedirect.com/science/article/pii/S0020025522015626?via%3Dihub). The dataset is contained within the folder  `data`, which includes the two visible-infrared test sets `1-vs-nir` and `2-vs-lwir`  and the negative pool `pascal-voc`. The main codes are written and run under `matlab` platform.

## Preliminary

* Compile and setup matconvnet.

  ```
  run matlab/vl_compilenn;
  run matlab/vl_setupnn;
  ```

* Compile esvm.

  ```
  run esvm/esvm_compile;
  ```

## Training and Test

* run mstm_fs_main.m. Please refer to the code for the setting of optional arguments.

  ```
  run mstm_fs_main
  ```

## Acknowledgements

Thank Tomasz Malisiewicz et al. for their proposition and implementation of Exemplar-SVM. They help and inspire this work.

## Bibtex

```
@article{FENG202320,
title = {Multi-spectral template matching based object detection in a few-shot learning manner},
journal = {Information Sciences},
volume = {624},
pages = {20-36},
year = {2023},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2022.12.067},
url = {https://www.sciencedirect.com/science/article/pii/S0020025522015626},
author = {Chen Feng and Zhiguo Cao and Yang Xiao and Zhiwen Fang and Joey Tianyi Zhou}
}
```



