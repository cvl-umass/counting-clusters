import os
import sys
import json
import argparse
import numpy as np
from os.path import join

sys.path.insert(0, 'src')
from datasets import Datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
import torchvision.models as models

from config import dset_root, dset_classes

import timm

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compute features and save into disk')
    parser.add_argument('--gpu', default='', type=str, help='CUDA visible device')
    parser.add_argument('--cuda', action='store_true', help='enables CUDA training')
    parser.add_argument('--save_dir', type=str, default='features', help='for features')
    parser.add_argument('--dataset', type=str, default='MacaqueFaces', help='for features')
    parser.add_argument('--model', type=str, default='L-384', help='for features')
    args = parser.parse_args()
    return args

def compute_features(data_loader):
    model.eval()
    features = np.array([], dtype=np.float32).reshape(0, 1536) # for megadescriptor-L-384
    meta = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, sample in enumerate(data_loader):
            data, label, filename = sample['image'], sample['label'], sample['filename']
            if args.cuda:
                data = data.cuda()
            image_features = model(data)
            feats = image_features.cpu().numpy()
            features = np.concatenate((features, feats))
            meta.append({
                         'id': int(batch_idx),
                         'label': int(label.cpu().numpy()[0]),
                         'filename': filename[0]})

            if (batch_idx + 1)%1000 == 0:
                print('%d/%d'%(batch_idx + 1, len(data_loader)))

    return features, meta

if __name__ == '__main__':

    args = parse_args()
    save_dir = join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    args.cuda = args.cuda and torch.cuda.is_available()

    if args.model == 'L-384':
        #model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        model = timm.create_model("hf_hub:BVRA/MegaDescriptor-L-384", pretrained=True) # depends on Timm version
    else:
        sys.exit('ERROR: Model not supported')

    if args.cuda:
        model.cuda()

    print('[loading dataset...]', end=" ")
    dset = Datasets(dset_root[args.dataset],
                    dataset=args.dataset,
                    classes=dset_classes[args.dataset])
    data_loader = torch_du.DataLoader(dset, batch_size=1, shuffle=False)
    print('dataset images: %d'%(len(data_loader.dataset)))

    try:
        features, meta = compute_features(data_loader)

        fname = 'megad_%s'%(args.model)
        np.save(join(save_dir, '%s.npy'%fname), features)
        with open(join(save_dir, '%s_meta.json'%fname), 'w') as fout:
            json.dump(meta, fout)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

