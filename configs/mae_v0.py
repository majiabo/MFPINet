"""
using MAE ERROR
"""
gpu_id = '0'
random_seed = 110
title = 'mae_v0'


data_root = '/mnt/disk_8t/kara/3DSR/data'
checkpoints_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/{}'.format(title)
tensorboard_path = '/mnt/disk_8t/kara/3DSR/log/tensorboards/{}'.format(title)
img_log_path = '/mnt/disk_8t/kara/3DSR/log/imgs/train/{}'.format(title)

#for pretrain_model
D_path = None#'/mnt/disk_8t/kara/3DSR/log/checkpoints/exp_0/D_18_1365.pth'
G_path = '/mnt/disk_8t/kara/3DSR/log/checkpoints/exp_0/G_3_240.pth'
#=================================
#       model config
#=================================
adver_decay = 0.01
start_epoch = 0
stop_epoch = 20
lr = 1e-4
decay_factor = 0.5
loss = 'mae'
pattern = 'all'
D = 'single_discriminator'
batch_size = 1
num_workers = 2
in_img = [0]           #输入图像的通道数
out_img = list(range(-5, 6))
in_c = len(in_img)*3
out_c = len(out_img)*3

# -------predict config ------
weight_id = ['G_19_1519', 'G_18_1443', 'G_17_1367']
test_img_log_path = '/mnt/disk_8t/kara/3DSR/log/imgs/test/{}__'.format(title)
