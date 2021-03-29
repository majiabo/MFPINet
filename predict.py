"""
author:majiabo
time:2019.2.8
function : 3dSR predict
"""

import sys
sys.path.append('../')
from lib.model import Generator
from lib.dataset import D3Dataset
from lib.dataset import wrap_multi_channel_img
import os
import cv2
import torch
import numpy as np
# modified this line
from configs import mae_v0 as args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
weight_path = './assets/ckpt/G_19_1519.pth'
img_path = './assets/images/sample_1.jpg'
if __name__ == '__main__':

    G = Generator(args.in_c, args.out_c)
    G.load_state_dict(torch.load(weight_path))
    G.eval()
    # preprocess
    img = cv2.imread(img_path)[..., ::-1]
    img = D3Dataset.degrade_img(img, True)
    img = img.astype('float32')/255
    img = torch.from_numpy(np.transpose(img[None, ...], axes=(0, 3, 1, 2)))

    prefix, postfix = os.path.split(img_path)
    save_path = os.path.join(prefix, 'result_{}'.format(postfix))
    with torch.no_grad():
        gen = G(img)
        tensor_list = [gen[0]]
        wraped = wrap_multi_channel_img(tensor_list)
        cv2.imwrite(save_path, wraped)

