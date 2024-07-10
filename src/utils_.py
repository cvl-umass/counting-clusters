import sys
import json
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

sys.path.insert(0, '../wildlife-datasets')
from wildlife_datasets import datasets
from wildlife_datasets.analysis import display_statistics

def load_features(args):
    fname = '%s_%s'%(args.pretraining, args.model)
    features = np.load(join(args.feats_dir, args.dataset, '%s.npy'%fname))
    with open(join(args.feats_dir, args.dataset, '%s_meta.json'%fname), encoding='utf-8') as f:
        meta = json.load(f)
    return features, meta

def plot_results(f_hat_nis_all, ci_nis_all, f_hat_mc_all, ci_mc_all, x, args, n_cats, lw=1.5, fs=9):
    fig, ax = plt.subplots(1,1, figsize=(2.7,2.3))
    plt.plot(x, f_hat_nis_all, color='maroon', linestyle='--', linewidth=lw, label='Nested-IS') 
    plt.fill_between(x, np.array(f_hat_nis_all)-np.array(ci_nis_all),
                                                 np.array(f_hat_nis_all)+np.array(ci_nis_all),
                                                 alpha=0.1, edgecolor='maroon', facecolor='maroon')
    plt.plot(x, f_hat_mc_all, color='teal', linestyle='--', linewidth=lw, label='Nested-MC')
    plt.fill_between(x, np.array(f_hat_mc_all)-np.array(ci_mc_all),
                                                 np.array(f_hat_mc_all)+np.array(ci_mc_all),
                                                 alpha=0.1, edgecolor='teal', facecolor='teal')
    plt.hlines(y=n_cats, xmin=-1, xmax=x[-1], color='black', linewidth=2, alpha=0.5, label='GT')
    plt.title('%s (%s) %d runs'%(args.dataset, args.model, args.runs), fontsize=fs)
    plt.ylabel('Estimated count', fontsize=fs)
    plt.xlabel('# sampled pairs / # total edges', fontsize=fs)
    plt.xticks([0, 0.004, 0.008], [0, '0.004%', '0.008%'], rotation=0, fontsize=fs)
    plt.xlim(-0.001, 0.012)
    plt.grid(True, color='gray', linestyle=':', linewidth=0.5)
    plt.legend(fontsize=fs)
    fig.tight_layout()
    plt.savefig('results_%s_%s_%s.pdf'%(args.dataset, args.model, args.pretraining), dpi=150)
    plt.show()

def load_dataset(dataset, root_dir, small=False, small_n=1000, classes=200, seed_=0):
    '''
    load dataset file names and labels (used for the dataloader)

    inputs:
    - dataset (str): dataset name ['cub', 'cars', 'aircraft', 'flowers', 'inaturalist']
    - root_dir (str)
    - small (bool): True to get reduced version of the dataset
    - small_n (int): small version size (e.g., total 1000 images)
    - classes (int): number of categories in the dataset
    - seed (int): seed for random sampling images to get small version
    '''
    files = []
    labels = []

    if dataset == 'MacaqueFaces':
        root = 'data/MacaqueFaces'
        datasets.MacaqueFaces.get_data(root)
        datasets.MacaqueFaces.download(root)
        d = datasets.MacaqueFaces(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    elif dataset == 'WhaleSharkID':
        root = 'data/WhaleSharkID'
        datasets.WhaleSharkID.get_data(root)
        datasets.WhaleSharkID.download(root)
        d = datasets.WhaleSharkID(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    elif dataset == 'GiraffeZebraID':
        root = 'data/GiraffeZebraID'
        datasets.GiraffeZebraID.get_data(root)
        datasets.GiraffeZebraID.download(root)
        d = datasets.GiraffeZebraID(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    elif dataset == 'CTai':
        root = 'data/CTai'
        datasets.CTai.get_data(root)
        datasets.CTai.download(root)
        d = datasets.CTai(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    elif dataset == 'CZoo':
        root = 'data/CZoo'
        datasets.CZoo.get_data(root)
        datasets.CZoo.download(root)
        d = datasets.CZoo(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    elif dataset == 'OpenCows2020':
        root = 'data/OpenCows2020'
        datasets.OpenCows2020.get_data(root)
        datasets.OpenCows2020.download(root)
        d = datasets.OpenCows2020(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    elif dataset == 'IPanda50':
        root = 'data/IPanda50'
        datasets.IPanda50.get_data(root)
        datasets.IPanda50.download(root)
        d = datasets.IPanda50(root)
        df = d.df
        display_statistics(df)
        identities = []
        for i, filename in enumerate(df.path):
            files.append(filename)
            identities.append(df.identity[i])
        unique_identities = list(set(identities))
        for identity in identities:
            labels.append(unique_identities.index(identity))

    else:
        sys.exit('ERROR: dataset not supported in utils > load_dataset()')

    return files, labels
