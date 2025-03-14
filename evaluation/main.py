import os
import re
# import torch.nn as nn
import argparse
import os.path as osp

from evaluator import Eval_thread
from dataloader import EvalDataset
proj_path = osp.dirname(osp.dirname(osp.abspath(__file__) )) 

# =================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
weight_path = osp.join(proj_path,'weights/250313_2249_DBANet+EVCBlock_ORSSD/250313_2249_DBANet+EVCBlock_ORSSD.pth.39')
# =================================================

filename = os.path.basename(weight_path)
match = re.search(r'(\d{6}_\d{4})_([\w+]+)_(\w+)\.pth\.(\d+)', filename)
TIME = match.group(1)
MODEL_NAME = match.group(2)
DATASET_NAME = match.group(3)
END_NUMBER = match.group(4)

dir_path = os.path.dirname(os.path.abspath(__file__))
data_path =os.path.join( os.path.dirname(os.path.dirname(dir_path)),"datasets")
data_type = DATASET_NAME #['ORSSD','EORSSD','ors-4199','RSISOD']

# from concurrent.futures import ThreadPoolExecutor
def main(cfg):
    # root_dir = cfg.root_dir
    # if cfg.save_dir is not None:
    #     output_dir = cfg.save_dir
    # else:
    #     output_dir = root_dir
    output_dir = cfg.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gt_dir = cfg.gt_root_dir
    pred_dir = cfg.pred_root_dir
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        #method_names = cfg.methods.split(' ')
        method_names = cfg.methods
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        #dataset_names = cfg.datasets.split(' ')
        dataset_names = cfg.datasets
    threads = []
    for dataset in dataset_names:
        for method in method_names:
            # loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
            print("123123",method_names)
            # loader = EvalDataset(osp.join(pred_dir,  method, dataset), osp.join(gt_dir, dataset))
            # thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            loader = EvalDataset(pred_dir, gt_dir)
            # thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            thread = Eval_thread(loader, method+'_'+END_NUMBER, dataset, output_dir, cfg.cuda)
            threads.append(thread)
            print("MODEL_NAME",method,"DATASET_NAME",dataset, "TIME",TIME)
    for thread in threads:
        print(thread.run())

if __name__ == "__main__":
    filename_without_ext = os.path.splitext(filename)[0]
    # 还有扩展名，继续去除
    filename_without_ext = os.path.splitext(filename_without_ext)[0]
    ########################改############################这里是预测图目录
    # pred_root_dir=os.path.join(data_path,"RS-SOD", data_type, 'test/images') # 注意观察该目录下文件夹结构，必须是：方法名/ORSSD/ 
    pred_root_dir=os.path.join(os.path.dirname(dir_path),"predict", MODEL_NAME, DATASET_NAME, ''+filename_without_ext +"_"+ END_NUMBER) # 注意观察该目录下文件夹结构，必须是：方法名/ORSSD/
    # pred_root_dir='/home/xxx/SOD_Evaluation_Metrics/pred_contrast_EORSSD/' # 注意观察该目录下文件夹结构，必须是：方法名/EORSSD/
    # pred_root_dir='/home/xxx/SOD_Evaluation_Metrics/pred_contrast_orsi4199/' # 注意观察该目录下文件夹结构，必须是：方法名/ors-4199/

    #######################改############################这里是选择预测的数据集
    gt_root_dir=os.path.join(data_path,"RS-SOD", data_type, 'test/gt')
    # gt_root_dir='/home/xxx/SOD_Evaluation_Metrics/gt_EORSSD/'
    # gt_root_dir='/home/xxx/SOD_Evaluation_Metrics/gt_RSISOD/'
    # gt_root_dir='/home/xxx/SOD_Evaluation_Metrics/gt_orsi4199/'
    # gt_root_dir='/home/xxx/SOD_Evaluation_Metrics/gt_orsi4199_NoSpace/'
    
    ########################改############################这里是指标结果存放目录，如果目录不存在会自动生成目录
    # score_dir = os.path.join(dir_path,"result", 'score_contrast_'+MODEL_NAME+'_'+data_type)
    # score_dir = os.path.join(dir_path,"result", 'score_contrast_'+MODEL_NAME.split('+')[0]+'_'+data_type)
    score_dir = os.path.join(dir_path,"result", 'score_contrast_'+data_type)
    # score_dir = '/home/xxx/SOD_Evaluation_Metrics/score_contrast_EORSSD/'
    # score_dir = '/home/xxx/SOD_Evaluation_Metrics/score_contrast_4199/'


    if not os.path.exists(score_dir):
        os.makedirs(score_dir)


    # MODEL_NAMES = os.listdir(pred_root_dir)
    # MODEL_NAMES.sort(key=lambda x: x[:])
    # DATA_NAMES = os.listdir(gt_root_dir)
    # DATA_NAMES.sort(key=lambda x: x[:])
    MODEL_NAMES = [MODEL_NAME]
    DATA_NAMES = [DATASET_NAME]
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=MODEL_NAMES)
    parser.add_argument('--datasets', type=str, default=DATA_NAMES)
    parser.add_argument('--gt_root_dir', type=str, default=gt_root_dir)
    parser.add_argument('--pred_root_dir', type=str, default=pred_root_dir)
    parser.add_argument('--save_dir', type=str, default=score_dir)
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
