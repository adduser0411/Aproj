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
from models.blocks.EdgeLoss import EdgeLoss

import utils.pytorch_iou as pytorch_iou


import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Overwriting.*")

# 在文件开头的导入部分添加
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = project_root +'/datasets/RS-SOD/'
formatted_time =datetime.now().strftime('%y%m%d_%H%M')

# ===============================================================
gpu=5
data_type='ORSSD' #['ORSSD','EORSSD','ors-4199','RSISOD']
from models.DBANetMambaOut import DBANet as Net
model_name = '_DBANetMambaOut_'
# ===============================================================

# 先定义save_name和save_path
save_name = formatted_time + model_name + data_type
save_path = './weights/'+ save_name

# 然后再初始化SummaryWriter
writer = SummaryWriter(f'./runs/{save_name}')  # TensorBoard日志
best_mae = float('inf')
early_stop_counter = 0

model = Net()
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default=data_type, help='choose dataset')
parser.add_argument('--gpu', type=int, default=gpu, help='set gpu id')
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
# 在导入部分添加
from utils.data import test_dataset
from torch.utils.tensorboard import SummaryWriter

# 在参数解析部分修改
val_path = os.path.join(data_root, data_type+'_aug', 'val')
if not os.path.exists(os.path.join(val_path, 'images')):
    print(f"Warning: Validation path {val_path}/images does not exist!")
    # 可以选择创建目录或使用训练集作为验证集
    val_path = os.path.join(data_root, data_type+'_aug', 'train')  # 回退方案

parser.add_argument('--val_root', type=str, default=val_path, help='validation set path')

opt = parser.parse_args()

# 在模型初始化后添加
writer = SummaryWriter(f'./runs/{save_name}')  # TensorBoard日志
best_mae = float('inf')
early_stop_counter = 0

torch.cuda.set_device(opt.gpu)
model.cuda()
print("## device ## :", opt.gpu)
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# 改用 AdamW 优化器
# optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=0.01)
data_type=opt.data_type
image_root=data_root+data_type+'_aug/train/images/'
gt_root=data_root+data_type+'_aug/train/gt/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
edge_loss = EdgeLoss()


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
        
        # 在train函数中添加边缘感知损失
        # loss = CE(sal, gts) + IOU(sal_sig, gts)
        loss = CE(sal, gts) + IOU(sal_sig, gts) + 0.3*edge_loss(sal_sig, gts)   

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

    
    if (epoch+1) > 35:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + '/' + save_name + '.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)

print("Let's go!")

def validate(model, data_root, data_type, epoch):
    model.eval()
    val_loader = test_dataset(data_root+data_type+'_aug/test/images/',
                             data_root+data_type+'_aug/test/gt/',
                             opt.trainsize)
    mae = 0
    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, _ = val_loader.load_data()
            image = image.cuda()
            res, _ = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            mae += torch.abs(res.sigmoid().cpu() - gt).mean()
    
    mae /= val_loader.size
    writer.add_scalar('MAE/val', mae, epoch)
    return mae


for epoch in tqdm.trange(1, opt.epoch, desc="Training Epochs", unit="epoch"):
    train(train_loader, model, optimizer, epoch)
    
    # 每5个epoch验证一次
    if epoch % 5 == 0:
        val_mae = validate(model, data_root, data_type, epoch)
        
        # 早停机制
        if val_mae < best_mae:
            best_mae = val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= 3:  # 连续3次未提升则停止
                print(f'Early stopping at epoch {epoch}')
                break

writer.close()




# 替换原有的adjust_lr
from torch.optim.lr_scheduler import CosineAnnealingLR

# 在优化器定义后添加
scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

# 在训练循环中替换adjust_lr调用
scheduler.step()