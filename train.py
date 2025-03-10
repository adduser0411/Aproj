import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
# import sys
import sys
import os
sys.path.append(os.getcwd())
# import numpy as np
import tqdm
import pdb, os, argparse
from datetime import datetime

from utils.loader import get_loader
from utils.utils import clip_gradient, adjust_lr

import utils.pytorch_iou as pytorch_iou

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Overwriting.*")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = project_root +'/datasets/RS-SOD/'
formatted_time =datetime.now().strftime('%y%m%d_%H%M')

# ===============================================================
torch.cuda.set_device(0)
data_type='ORSSD' #['ORSSD','EORSSD','ors-4199','RSISOD']
from models.DBANet_SimAM_ShuffleAttn import DBANet_SimAM_ShuffleAttn as Net
save_name = formatted_time + '_DBANet+SimAM+ShuffleAttn2_' + data_type
# ===============================================================


model = Net()
save_path = './weights/'+ save_name

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=45, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args() #


model.cuda() 
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# 改用 AdamW 优化器
# optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=0.01)

image_root=data_root+data_type+'_aug/train/images/'
gt_root=data_root+data_type+'_aug/train/gt/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)



def train(train_loader, model, optimizer, epoch):
    model.train()
    # 使用 tqdm 包装 train_loader 以显示进度条
    # train_loader = tqdm.tqdm(train_loader, desc=f'{datetime.now()} Epoch {epoch}/{opt.epoch}', total=len(train_loader), mininterval=20)
    # train_loader = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{opt.epoch}', total=len(train_loader),
    #                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        sal, sal_sig = model(images)
        loss = CE(sal, gts) + IOU(sal_sig, gts)

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                            opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))
            # 更新 tqdm 进度条的描述信息 
            # train_loader.set_description(f'Epoch {epoch}/{opt.epoch}, Loss: {loss.data:.4f}, LR: {opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch):.6f}')

    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) > 38:
        torch.save(model.state_dict(), save_path + '/' + save_name + '.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)

print("Let's go!")

for epoch in tqdm.trange(1, opt.epoch, desc="Training Epochs", unit="epoch",mininterval=20):
# for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
