'''不包含深度监督的train脚本'''
import time
import numpy as np
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
# from utils.dataEdge import get_loader
from utils.utils import clip_gradient, adjust_lr
from models.blocks.EdgeLoss import EdgeLoss
# from utils.data import test_dataset
import utils.pytorch_iou as pytorch_iou
from utils.data_gt2tenser import test_dataset
# from utils.data import test_dataset6
import logging
#重构

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Overwriting.*")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = project_root +'/datasets/RS-SOD/'
formatted_time =datetime.now().strftime('%y%m%d_%H%M')

# ===============================================================
gpu=4
data_type='EORSSD' #['ORSSD','EORSSD','ors-4199','RSISOD']
from models.enhancedpvtme import DBANet as Net
model_name = '_enhancedpvtme_'
# ===============================================================


model = Net()

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default=data_type, help='choose dataset')
parser.add_argument('--gpu', type=int, default=gpu, help='set gpu id')
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args() #
torch.cuda.set_device(opt.gpu)
model.cuda()
print("## device ## :", opt.gpu)
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
data_type=opt.data_type
image_root=data_root+data_type+'_aug/train/images/'
gt_root=data_root+data_type+'_aug/train/gt/'
# edge_root=data_root+data_type+'_aug/train/edge/'
# train_loader = get_loader(image_root, gt_root,edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
edge_loss = EdgeLoss()


save_name = formatted_time + model_name + data_type
save_path = './weights/'+ save_name

# 在train_show_figure.py中添加以下全局变量
best_mae = 1
best_Fmax = 0
best_epochM = 0
best_epochF = 0

# 添加测试数据集路径
test_image_root = data_root + data_type + '/test/images/'
test_gt_root = data_root + data_type + '/test/gt/'
test_loader = test_dataset(test_image_root, test_gt_root, testsize=opt.trainsize)

def is_foreground():
    """判断是否在前台运行"""
    return sys.stdin.isatty() and 'nohup' not in os.environ.get('_', '')

def get_progress_wrapper(iterable, desc=None, unit=None):
    """智能返回进度条包装器或原始迭代器（根据是否在前台）"""
    if is_foreground():
        return tqdm.tqdm(iterable, desc=desc, unit=unit)
    return iterable

def train(train_loader, model, optimizer, epoch):
    model.train()
    # 使用 tqdm 包装 train_loader 以显示进度条
    # train_loader = tqdm.tqdm(train_loader, desc=f'{datetime.now()} Epoch {epoch}/{opt.epoch}', total=len(train_loader), mininterval=20)
    # train_loader = tqdm.tqdm(train_loader, desc=f'Epoch {epoch}/{opt.epoch}', total=len(train_loader),
    #                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    wrapped_loader = get_progress_wrapper(train_loader, desc=f'Epoch {epoch}/{opt.epoch}', unit='batch')
    for i, pack in enumerate(wrapped_loader, start=1):
    # for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        
        # #================深度监督损失【
        # # sal, sal_sig = model(images)
        # pred, pred_sig, x1, x1_sig, x23, x23_sig, x4, x4_sig = model(images)

        # # 计算各层损失
        # loss_main = CE(pred, gts) + IOU(pred_sig, gts)
        # loss_x1 = CE(x1, gts) + IOU(x1_sig, gts)
        # loss_x23 = CE(x23, gts) + IOU(x23_sig, gts) 
        # loss_x4 = CE(x4, gts) + IOU(x4_sig, gts)

        # # 加权总损失
        # # loss = loss_main + 0.5*loss_x1 + 0.5*loss_x23 + 0.5*loss_x4
        # # loss = loss_main + 0.5*loss_x1 + 0.5*loss_x23 + 0.5*loss_x4  + 0.3*edge_loss(pred_sig, gts) #带上边缘损失
        # # loss = loss_main + 0.6*loss_x1 + 0.4*loss_x23 + 0.3*loss_x4 # 渐进式权重
        # loss = loss_main + 0.6*loss_x1 + 0.4*loss_x23 + 0.3*loss_x4  + 0.3*edge_loss(pred_sig, gts)  #渐进权重带边缘损失

        # loss.backward()
        # #=================】


        #================CE+IOU损失【
        sal, sal_sig = model(images)
        # 在train函数中添加边缘感知损失
        loss = CE(sal, gts) + IOU(sal_sig, gts)
        # loss = CE(sal, gts) + IOU(sal_sig, gts) + 0.3*edge_loss(sal_sig, gts) 

        loss.backward()
        # #=================】


        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # 修改时间格式化部分
        current_time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        
        if(is_foreground()):
            if hasattr(wrapped_loader, 'set_description'):
                wrapped_loader.set_description(f'{current_time} Epoch [{epoch}/{opt.epoch}], Loss: {loss.data:.4f} ')
        else:
            if i % 20 == 0 or i == total_step:
                print(f'{current_time} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], Learning Rate: {opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch)}, Loss: {loss.data:.4f} ')

        # 更新 tqdm 进度条的描述信息 
        # train_loader.set_description(f'Epoch {epoch}/{opt.epoch}, Loss: {loss.data:.4f}, LR: {opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch):.6f}')


    if (epoch+1) > 30:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + '/' + save_name + '.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)

print("Let's go!")

# 添加Eval_fmeasure函数
def _eval_pr(y_pred, y, num):
    if y_pred.sum() == 0:
        y_pred = 1 - y_pred
        y = 1 - y
    prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
    thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall

def Eval_fmeasure(pred, gt):
    beta2 = 0.3
    pred = pred.cuda()
    gt = gt.cuda()
    with torch.no_grad():
        pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
        prec, recall = _eval_pr(pred, gt, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0
    return f_score

# 添加test函数
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epochM, best_epochF, best_Fmax
    print('\n###TESTING###')
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        F_value = 0.0
        time_sum = 0
        
        # # 使用tqdm包装测试循环
        for i in get_progress_wrapper(range(test_loader.size), desc=f'Testing Epoch {epoch}', unit='img'):
        # for i in range(test_loader.size):
            image, gt, gt2tensor = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            gt2tensor = gt2tensor.cuda()
            
            time_start = time.time()
            res, sal_sig = model(image)
            time_end = time.time()
            time_sum += (time_end - time_start)
            
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            smap = res.sigmoid()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            F_value += Eval_fmeasure(smap, gt2tensor)
            if i == test_loader.size-1:
                print('Running time {:.5f}'.format(time_sum/test_loader.size))
                print('FPS {:.5f}'.format(test_loader.size / time_sum))

        mae = mae_sum / test_loader.size
        maxF = (F_value/test_loader.size).max().item()
        
        if epoch == 1:
            best_mae = mae
            best_Fmax = maxF
            logging.info('FPS {:.5f}'.format(test_loader.size / time_sum))
        else:
            if mae < best_mae:
                best_mae = mae
                best_epochM = epoch
                torch.save(model.state_dict(), save_path + 'M.pth')
            if maxF > best_Fmax:
                best_Fmax = maxF
                best_epochF = epoch
                torch.save(model.state_dict(), save_path + 'F.pth')

        print('###TEST###')
        print('MAE: ', mae)
        print('maxF: {:.4f}\t'.format(maxF))
        print('bestMAE: ', best_mae)
        print('bestMaxF: ', best_Fmax)
        print('best_epochMAE: ', best_epochM)
        print('best_epochmaxF: ', best_epochF)
        logging.info('#TEST#:Epoch:{} MAE:{} maxF:{}'.format(epoch, mae, maxF))
        logging.info('#TEST#:best_epochMAE:{} bestMAE:{} '.format(best_epochM, best_mae))
        logging.info('#TEST#:best_epochmaxF:{} bestmaxF:{} '.format(best_epochF, best_Fmax))

# 修改主训练循环
for epoch in tqdm.trange(1, opt.epoch, desc="Training Epochs", unit="epoch", mininterval=20):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    if epoch > 30 :  # 与原来的保存条件保持一致
        test(test_loader, model, epoch, save_path)