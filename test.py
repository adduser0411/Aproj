import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc
import imageio
import time
import re
import importlib

# a new line
# from models.DBANet import DBANet
from utils.data import test_dataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Overwriting.*")
proj_path = os.path.dirname(os.path.abspath(__file__) ) 

# parser = argparse.ArgumentParser()
# opt = parser.parse_args()

dir_path = os.path.dirname(os.path.abspath(__file__))
dataset_path =os.path.join( os.path.dirname(dir_path),"datasets/RS-SOD")

# ================================================================
device = 4
# m='DBANetARConv'
path="weights/250401_2255_DBANetARConv_EORSSD/250401_2255_DBANetARConv_EORSSD.pth.38"
# from models.DBANetARConv import DBANet as Net
# ================================================================
pattern = r'weights/\d{6}_\d{4}_([^_]+)_'
match = re.search(pattern, path)
if match:
    m = match.group(1)
    print(f"提取的模型名是: {m}")
else:
    print("未匹配到模型名")


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--d', type=int, default=device, help='set device')
parser.add_argument('--m', type=str, default=m, help='set model')
parser.add_argument('--w', type=str, default=path, help='set path')
opt = parser.parse_args() #
if opt:
    device = opt.d
    path=opt.w
    m=opt.m

weight_path = os.path.join(proj_path,path)
torch.cuda.set_device(device)
try:
    module = importlib.import_module(f'models.{m}')
    Net = getattr(module, 'DBANet')
except ImportError:
    print(f"Failed to import module models.{m}")
except AttributeError:
    print(f"Module models.{m} does not have class {'DBANet'}")

model = Net()
model.load_state_dict(torch.load(weight_path))

model.cuda()
model.eval()

filename = os.path.basename(weight_path)
match = re.search(r'(\d{6}_\d{4})_([\w+]+)_(\w+)\.pth\.(\d+)', filename)
MODEL_NAME = match.group(2)
DATASET_NAME = match.group(3)
END_NUMBER = match.group(4)
test_datasets = [DATASET_NAME]
# test_datasets = ['ORSSD']
#test_datasets = ['EORSSD','ORSSD','ors-4199']

for dataset in test_datasets:
    # save_path = './models/DBANet/' + dataset + 'DBABet-38/'
    filename_without_ext = os.path.splitext(filename)[0]
    # 如果还有扩展名，继续去除
    filename_without_ext = os.path.splitext(filename_without_ext)[0]
    save_path = os.path.join(dir_path,"predict", MODEL_NAME, DATASET_NAME, filename_without_ext +"_"+ END_NUMBER + '/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
      
    image_root = os.path.join( dataset_path , dataset , 'test/images/')
    print(dataset)
    
    gt_root = os.path.join( dataset_path, dataset, 'test/gt/')
    
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, sal_sig = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        # 修改此处，使用 nn.functional.interpolate 替代 nn.functional.upsample
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8) 
        imageio.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('FPS {:.5f}'.format(test_loader.size / time_sum))

