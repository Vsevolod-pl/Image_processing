import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import colour
import cv2
import random
import os
import random
import torch
import argparse
import XYZ_to_SRGB

import menon
device = 'cuda' if torch.cuda.is_available else 'cpu'


def load_denoising_model(path2model='./KAIR/model_zoo/dncnn_15.pth'):
    path2folder = '/'.join(path2model.split('/')[:-2])
    import sys
    sys.path.append(path2folder)
    from KAIR.models.network_dncnn import DnCNN

    model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
    model.load_state_dict(torch.load(path2model))
    return model.to(device)

# ## Denoising
def denoise(img_numpy, model):
    imgT = torch.transpose(torch.tensor(img_numpy, dtype=torch.float32), 0, 2)
    imgT = imgT.to(device)
    with torch.no_grad():
        denoised = model(imgT.unsqueeze(1))
        denoised = denoised.squeeze_()
    return denoised

def load_color_space_transform(path2model):
    from torch import nn

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding='same'),
        nn.LeakyReLU(),
        nn.Conv2d(16, 16, 3, padding='same'),
        nn.LeakyReLU(),
        nn.Conv2d(16, 16, 3, padding='same'),
        nn.LeakyReLU(),
        nn.Conv2d(16, 3, 3, padding='same')
    ).to(device)

    model.load_state_dict(torch.load(path2model))
    return model.to(device)
    

def full_pipeline(img_path, out_path, denoise_model, cst_model):
    sample = np.load(img_path, allow_pickle=True).item()
    sample_img   = sample['image']
    sample_cmfs  = sample['cmfs']
    sample_light = sample['light']
    sample_bayer = sample['bayer']
    sample_mean  = sample['mean']
    sample_sigma = sample['sigma']

    img = menon.bayer2rgb(sample_img/255, pattern=sample_bayer)

    denoised = denoise(img, denoise_model)

    with torch.no_grad():
        img_xyz = cst_model(denoised.to(device))

    light = img_xyz.max(dim=2).values.max(dim=1).values
    res = torch.transpose(img_xyz, 0, 2).cpu()/light.cpu()

    img_srgb = XYZ_to_SRGB.XYZ_TO_SRGB().XYZ_to_sRGB(res.numpy())

    plt.imsave(out_path, img_srgb)
    return img_srgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Image processing pipeline', description='converts all files inside --inf folder to pngs in --out folder, --denoise-weights to specify path to weights for KAIR denoising DnCNN, --cst-weights to specify path to cst model weights',)
    parser.add_argument('-i', '--inf')
    parser.add_argument('-o', '--out')
    parser.add_argument('-d', '--denoise-weights')
    parser.add_argument('-c', '--cst-weights')
    args = parser.parse_args()

    denoise_model = load_denoising_model(args.denoise_weights)
    cst_model = load_color_space_transform(args.cst_weights)

    for file in os.listdir(args.inf):
        out_name = f'{args.out}/{file}.png'
        full_pipeline(f'{args.inf}/{file}', out_name, denoise_model, cst_model)