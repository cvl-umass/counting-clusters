
<h1 align="center">Visual Re-ID for Population Size Estimation with Nested Importance Sampling</h1>


PyTorch implementation for [Human-in-the-Loop Visual Re-ID for Population Size Estimation](https://arxiv.org/abs/2312.05287).


## Getting started

1. **Install Anaconda:** We recommend using the free [Anaconda Python
distribution](https://www.anaconda.com/download/), which provides an
easy way for you to handle package dependencies. Please be sure to
download the Python 3 version.

3. **Anaconda virtual environment:** To set up and activate the virtual environment,
run:
```
$ conda create -n nestedis python=3.*
$ conda activate nestedis
```

To install requirements, run:
```
$ conda install --yes --file requirements.txt
```

4. **PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).


5. Install [wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets).

-------
## Pre-compute MegaDescriptor Features

In this repository we use [MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384) features, but the image features can be computed with any model (e.g., CLIP, DINO, etc.)

To pre-compute the image features run:

```
$ bash compute_features.sh
```

## Run Nested-IS estimator

Here we run Nested-IS for a variable number of samples and plot the predictions with confidence intervals, and compare with Nested Monte Carlo (Nested-MC). The script plots the estimation for both methods after 100 runs.

To run the estimator run:
```
bash run_estimator.sh
```

### Cite

If you find this code useful in your research, please consider citing:
```
@inproceedings{psvm_eccv2024,
      	       year = 2024,
      	       author = {Perez, Gustavo and Sheldon, Daniel and Van Horn, Grant and Maji, Subhransu},
      	       title = {Human-in-the-Loop Visual Re-ID for Population Size Estimation},
	       booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)}
```

## Acknowledgements

This work is supported by the [National Science Foundation (NSF)](https://nsf.gov/index.jsp) of the United States under grants \#1749854, \#1749833, and \#2329927.
