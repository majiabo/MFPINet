"""
author:majiabo
time:2019.2.8
function : 3dSR
"""

import sys
sys.path.append('../')
from lib.model import Generator, GeneratorHR
from lib.model import Discrimator, SingleDiscriminator
from lib.dataset import D3Dataset
from lib.dataset import wrap_multi_channel_img
from lib.utilis import path_checker
from torch.utils.data import DataLoader

import os
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

from configs import mae_v0 as args
torch.manual_seed(args.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_num = len(args.gpu_id.split(','))
device = torch.device('cuda')

batch_size = args.batch_size*device_num
num_workers = device_num*args.num_workers
img_save_step = int(200/batch_size)
model_save_step = int(3000/batch_size)
lr = args.lr

path_checker(args.checkpoints_path)
path_checker(args.img_log_path)

Writer = SummaryWriter(args.tensorboard_path)
#===============================
# set model, loss, dataset,optimizer
#===============================
# train_set = D3Dataset(args.data_root, 'train', input_layer=args.in_img, out_layer=args.out_img, seed=0)
train_set = D3Dataset(args.data_root, 'train', input_layer=args.in_img, pattern='all',
                      out_layer=args.out_img, seed=0, lr2hr=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
all_d = {
    'single_discriminator': SingleDiscriminator,
    'discriminator': Discrimator
}

G = Generator(args.in_c, args.out_c).to(device)
# G = GeneratorHR(args.in_c, args.out_c).to(device)
D = all_d[args.D](args.out_c).to(device)
G.train()
D.train()

if args.G_path:
    G.load_state_dict(torch.load(args.G_path), strict=True)
if args.D_path:
    D.load_state_dict(torch.load(args.D_path), strict=True)

optimizerG = torch.optim.Adam(G.parameters(), lr=lr)
optimizerD = torch.optim.Adam(D.parameters(), lr=lr)

if device_num>1:
    G = torch.nn.DataParallel(G).to(device)
    D = torch.nn.DataParallel(D).to(device)
main_loss = {
    'mae': torch.nn.L1Loss(),
    'mse': torch.nn.MSELoss()
}

bce_loss = torch.nn.BCELoss()
mse_loss = main_loss[args.loss]

img_save_counter = 0

#=============================
#   train model
#=============================
for epoch in range(args.start_epoch, args.stop_epoch):
    for index,(lr_img,hr_img) in enumerate(train_loader):

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        #=============
        # update D
        #=============
        D.zero_grad()
        gen = G(lr_img)
        gen_label = D(gen)
        real_label = D(hr_img)

        valid = torch.cuda.FloatTensor(gen_label.shape).fill_(1.)
        invalid = torch.cuda.FloatTensor(gen_label.shape).fill_(0.0)

        real_loss = bce_loss(real_label,valid)
        fake_loss = bce_loss(gen_label,invalid)
        dloss = real_loss + fake_loss
        dloss.backward(retain_graph = True)
        optimizerD.step()
        #===========
        # update G
        #===========
        G.zero_grad()
        mse = mse_loss(gen,hr_img)
        adver = bce_loss(gen_label,valid)*args.adver_decay

        gloss = mse + adver
        gloss.backward()
        optimizerG.step()

        # process log
        if (index) % img_save_step == 0:
            tensor_list = [gen[0,:,:,:],hr_img[0,:,:,:]]
            wraped = wrap_multi_channel_img(tensor_list)

            Writer.add_scalar('scalar/dloss',dloss,img_save_counter)
            Writer.add_scalar('scalar/fake_loss', fake_loss, img_save_counter)
            Writer.add_scalar('scalar/real_loss', real_loss, img_save_counter)
            Writer.add_scalar('scalar/gloss',gloss,img_save_counter)
            Writer.add_scalar('scalar/adver',adver,img_save_counter)
            Writer.add_scalar('scalar/{}'.format(args.loss),mse,img_save_counter)
            
            img_path = os.path.join(args.img_log_path, '{}_{}.jpg'.format(epoch, img_save_counter))
            cv2.imwrite(img_path,wraped)
            img_save_counter += 1

        if (index+1) % model_save_step == 0 and index != 0:
            gg_path = os.path.join(args.checkpoints_path, 'G_{}_{}.pth'.format(epoch, img_save_counter))
            dd_path = os.path.join(args.checkpoints_path, 'D_{}_{}.pth'.format(epoch, img_save_counter))
            if device_num>1:
                torch.save(G.module.state_dict(), gg_path)
                torch.save(D.module.state_dict(), dd_path)
            else:
                torch.save(G.state_dict(), gg_path)
                torch.save(D.state_dict(), dd_path)
        sys.stdout.write("\r[Epoch {}/{}] [Batch {}/{}] [d_fake:{:.5f}, d_real:{:.5f}] [g_{}:{:.5f}, g_adver:{:.5f}]".format(
            epoch,args.stop_epoch,index,len(train_loader),fake_loss.item(), real_loss.item(),args.loss, mse.item(), adver.item()))
        sys.stdout.flush()
    print()
    if epoch%4 ==0 and epoch!=0:
        lr = lr*args.decay_factor
        for param in optimizerD.param_groups:
            param['lr'] = lr
        for param in optimizerG.param_groups:
            param['lr'] = lr
        print('learning rate is :',lr)

