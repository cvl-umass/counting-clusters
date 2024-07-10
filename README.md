<h1 align="center">Visual Re-ID for Population Size Estimation with Nested Importance Sampling</h1>

![title_image](main.png)

Python implementation for [Human-in-the-Loop Visual Re-ID for Population Size Estimation](https://arxiv.org/abs/2312.05287). 
This project uses [wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets) which has a simple API to download more than 35 animal re-ID datasets (you can check an overview of the datasets [here](https://wildlifedatasets.github.io/wildlife-datasets/datasets/)). 
We also use [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384) to compute image features. For more information check their [paper](https://wildlifedatasets.github.io/wildlife-datasets/datasets/).

## Getting started

1. **Install Anaconda:** We recommend using the free [Anaconda Python
distribution](https://www.anaconda.com/download/), which provides an
easy way for you to handle package dependencies. Please be sure to
download the Python 3 version.

3. **Anaconda virtual environment:** To set up and activate the virtual environment,
run:
```
conda create -n nestedis python=3.*
conda activate nestedis
```

To install requirements, run:
```
conda install --yes --file requirements.txt
```

4. **PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).


5. Install wildlife-datasets with `pip install wildlife-datasets` or refer to their repository [here](https://github.com/WildlifeDatasets/wildlife-datasets).

-------
## Pre-compute MegaDescriptor Features

In this repository we use [MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384) features, but the image features can be computed with any model (e.g., CLIP, DINO, etc.). 
Additionally, we use wildlife-datasets API to download the datasets automatically. These are downloaded into `data/` folder by default but can be changed in `src/config.py`. 
To pre-compute the image features for MacaqueFaces dataset run:

```
CUDA_VISIBLE_DEVICES=0 python compute_features.py --dataset MacaqueFaces --cuda 
```

Computing the features for the entire MacaqueFaces dataset with a pretrained MegaDescriptor-L-380 takes 2-3 minutes on a RTX 2080 Ti, and around 1 hour without GPU support. 
This repository is ready to run with any of the wildlife-datasets included in our paper (i.e., CTai, CZoo, MacaqueFaces, WhaleSharkID, GiraffeZebraID, OpenCows2020, and IPanda50). 
Modify `load_dataset()` in `src/utils_.py` to add additional datasets.

The pre-computed features will be saved into `features/<dataset name>/` as `<model>_<pretraining>.npy` and the metadata (labels, filenames, etc.) as `<model>_<pretraining>_meta.json`. 
You can load the precomputed features using: 

```
from utils_ import load_features

args = lambda:0
args.dataset = 'MacaqueFaces'
args.feats_dir = 'features'
args.model = 'L-384'
args.pretraining = 'megad' 
features, meta = load_features(args)

print(features.shape)
print(len(meta))
print(meta[0])
```

The code above  should display:
```
>>> (6280, 1536)
>>> 6280
>>> {'id': 0, 'label': 0, 'filename': 'MacaqueFaces/Random/Teal/Macaque_Face_5185.jpg'}
```

## Run Nested-IS estimator

To run our nested-IS estimator no GPU is required. 
As default the script will run nested-IS and nested-MC baselines 10 times with ~70 increasing values of number of samples, and plot the mean results with confidence intervals for every number of samples (as presented in our paper). 
The script will take around 1 minute for the MacaqueFaces dataset. You can run the estimator with:
```
python run_estimator.py --dataset MacaqueFaces --runs 10
```

The previous command line should display the following:
```
>>> [loading MacaqueFaces-L-384-megad features...] done (6280, 1536) [0.0s]
>>> [creating GT matrix...] done (6280, 6280) [4.4s]
>>> [calculating similarity matrix (cosine)...] done (6280, 6280) [18.8s]
>>> [running nested-MC and nested-NIS (10 runs)...] done [0.8m]
```

The script will save the results as `results_<dataset>_<model>_<pretraining>.pdf`.

## Cite

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
