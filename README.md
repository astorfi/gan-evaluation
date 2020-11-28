# On the Evaluation of Generative Adversarial Networks By Discriminative Models

<!-- [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/astorfi/cor-gan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/astorfi/cor-gan/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/astorfi/cor-gan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/astorfi/cor-gan/alerts/) -->

[![Name](https://img.shields.io/github/license/astorfi/gan-evaluation)](https://github.com/astorfi/gan-evaluation/blob/master/LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-2010.03549-b31b1b.svg)](https://arxiv.org/abs/2010.03549)






This repository contains an implementation of "On the Evaluation of Generative Adversarial Networks By Discriminative Models".


For a detailed description of the architecture please read [our paper](https://arxiv.org/abs/2010.03549). Using the code of this repository is allowed with **proper attribution**: Please cite the paper if you use the code from this repository in your work.

## Bibtex

    @article{torfi2020evaluation,
      title={On the Evaluation of Generative Adversarial Networks By Discriminative Models},
      author={Torfi, Amirsina and Beyki, Mohammadreza and Fox, Edward A},
      journal={arXiv preprint arXiv:2010.03549},
      year={2020}
    }



Table of contents
=================

<!--ts-->
   * [Paper Summary](#paper-summary)
   * [Running the Code](#Running-the-Code)
      * [Prerequisites](#Prerequisites)
      * [Datasets](#Datasets)
   * [Collaborators](#Collaborators)
<!--te-->


## Paper Summary

<details>
<summary>Abstract</summary>

 *Generative Adversarial Networks (GANs) can accurately model complex multi-dimensional data and generate realistic samples. However, due to their implicit estimation of data distributions, their evaluation is a challenging task. The majority of research efforts associated with tackling this issue were validated by qualitative visual evaluation. Such approaches do not generalize well beyond the image domain. Since many of those evaluation metrics are proposed and bound to the vision domain, they are difficult to apply to other domains. Quantitative measures are necessary to better guide the training and comparison of different GANs models. In this work, we leverage Siamese neural networks to propose a domain-agnostic evaluation metric: (1) with a qualitative evaluation that is consistent with human evaluation, (2) that is robust relative to common GAN issues such as mode dropping and invention, and (3) does not require any pretrained classifier. The empirical results in this paper demonstrate the superiority of this method compared to the popular Inception Score and are competitive with the FID score.*

</details>


## Running the Code

### Prerequisites

* Pytorch
* CUDA [strongly recommended]

**NOTE:** PyTorch does a pretty good job in installing required packages but you should have installed CUDA according to PyTorch requirements.
Please refer to [this link](https://pytorch.org/) for further information.

### Datasets

You need to download and process the datasets mentioned in the paper. **The code in this repository is for MNIST and [CELEB A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets only**.


## Collaborators

| [<img src="https://github.com/astorfi.png" width="100px;"/>](https://github.com/astorfi)<br/> [<sub>Amirsina Torfi</sub>](https://github.com/astorfi) | [<img src="https://github.com/mohibeyki.png" width="100px;"/>](https://github.com/mohibeyki)<br/> [<sub>Mohammadreza Beyki</sub>](https://github.com/mohibeyki) |
| --- | --- |

<!-- ## Credit

This research conducted at [Virginia Tech](https://vt.edu/) under the supervision of [Dr. Edward A. Fox](http://fox.cs.vt.edu/foxinfo.html). -->
