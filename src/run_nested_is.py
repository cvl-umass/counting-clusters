import os
import time
import argparse
import numpy as np
from os.path import join

from methods import nested_is, nested_mc, calculate_similarity
from utils_ import load_features, plot_results

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Estimate population size in a dataset')
    parser.add_argument('--feats_dir', type=str, default='features', help='for features')
    parser.add_argument('--dataset', type=str, default='MacaqueFaces', help='for features')
    parser.add_argument('--model', type=str, default='megad', help='for features')
    parser.add_argument('--pretraining', type=str, default='L-384', help='for features')
    parser.add_argument('--metric', type=str, default='cosine', help='for similarity')
    parser.add_argument('--ratio', type=int, default=1, help='Nn/Nv ratio (use 7 for GiraffeZebraID)')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature for similarity softmax')
    parser.add_argument('--runs', type=int, default=100, help='for error calculation')
    parser.add_argument('--seed', type=int, default=0, help='seed for features')
    parser.add_argument('--verbose', action='store_true', help='print intermediate progress')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    ti = time.time()
    print('[loading %s-%s-%s features...]'%(args.dataset, args.model, args.pretraining), end=" ")
    features, meta = load_features(args)
    print('done %s [%.1fs]'%(str(features.shape), time.time() - ti))

    # caculate GT matrix (Used for evaluation ONLY)
    ti = time.time()
    print('[creating GT matrix...]', end=" ", flush=True)
    gt_s_ij = np.zeros((features.shape[0], features.shape[0]))
    cats = []
    for i, sample1 in enumerate(meta):
        if sample1['label'] not in cats:
            cats.append(sample1['label'])
        for j, sample2 in enumerate(meta):
            if sample1['label'] == sample2['label']:
                gt_s_ij[i,j] = 1
    n_cats = len(cats)
    print('done %s [%.1fs]'%(str(gt_s_ij.shape), time.time() - ti))

    # calculate similarity matrix (from features)
    # create a list of lists with feats. per class (C, (N, D))
    ti = time.time()
    print('[calculating similarity matrix (%s)...]'%args.metric, end=" ", flush=True)
    feats = [ [] for _ in range(n_cats)] # create list with empty lists
    current_label = meta[0]['label']
    cont = 0
    for i, feat in enumerate(features):
        if current_label != meta[i]['label']:
            cont += 1
            current_label = meta[i]['label']
        feats[cont].append(feat)
    s_ij = calculate_similarity(feats, metric=args.metric, tau=args.tau)
    print('done %s [%.1fs]'%(str(s_ij.shape), time.time() - ti))

    # Calculate Nested-IS and Nested-MC estimates
    ti = time.time()
    print('[running nested-MC and nested-NIS (%d runs)...]'%args.runs, end=" ", flush=True)
    N = [1] + list(range(1,201,3))
    
    f_hat_mc_all, f_hat_nis_all, ci_mc_all, ci_nis_all = [], [], [], []
    for ii, n in enumerate(N): 
        tii = time.time()
        f_hat_nis_, ci_nis_, f_hat_mc_, ci_mc_ = [], [], [], []
        n_hat = None
        Nv = max(1, round(n/np.sqrt(args.ratio)))
        Nn = max(1, round(args.ratio/np.sqrt(args.ratio)*n))
        for _ in range(args.runs):
            f_hat_nis, ci_nis, n_hat = nested_is(gt_s_ij, s_ij, Nv, Nn, ci=True, n_hat=n_hat)
            f_hat_mc, ci_mc = nested_mc(gt_s_ij, Nv, Nn, ci=True)
            f_hat_nis_.append(f_hat_nis)
            f_hat_mc_.append(f_hat_mc)
            ci_nis_.append(ci_nis)
            ci_mc_.append(ci_mc)
        if args.verbose:
            print('%d/%d [%.2fs]'%(ii+1, len(N), time.time() - tii))
        f_hat_mc_all.append(np.mean(f_hat_mc_))
        f_hat_nis_all.append(np.mean(f_hat_nis_))
        ci_mc_all.append(np.mean(ci_mc_))
        ci_nis_all.append(np.mean(ci_nis_))
    print('done [%.1fm]'%((time.time() - ti)/60.))

    # Plot results
    tn = len(features)
    x = [n**2/((tn*(tn-1))/2)*100. for n in N]
    plot_results(f_hat_nis_all, ci_nis_all, f_hat_mc_all, ci_mc_all, x, args, n_cats)
