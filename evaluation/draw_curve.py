import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

'''
To run this script, pleasure run main.py first to produce the curve_cache.
'''
proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) )
# curve_cache_dir='/home/xxx/SOD_Evaluation_Metrics/score/curve_cache_2024/' #'./Your_Method_Name/score/curve_cache/'
curve_cache_dir=os.path.join(proj_path,'evaluation/result/score_contrast_ORSSD/curve_cache/')#'./Your_Method_Name/score/curve_cache/'
# curve_save_dir='/home/xxx/SOD_Evaluation_Metrics/curves/'
curve_save_dir=os.path.join(proj_path,'evaluation/result/score_contrast_ORSSD/curves/')
if not os.path.exists(curve_save_dir):
    os.makedirs(curve_save_dir)
datasets=os.listdir(curve_cache_dir)
for dataset in datasets:
    plot_pr_vals = {}
    plot_fm_vals = {}
    dataset_dir=os.path.join(curve_cache_dir,dataset)
    methods=os.listdir(dataset_dir)
    print(methods)
    # methods = ['HVPNet','DNTD', 'DAFNet', 'ACCoNet', 'UG2L', 'SDNet',  'ADSTNet', 'HFCNet', 'SFANet', 'Ours']
    

    for method in methods:
        method_dir=os.path.join(dataset_dir,method)
        pr_cache_path=os.path.join(method_dir,'pr.txt')
        fm_cache_path=os.path.join(method_dir,'fm.txt')
        prec=np.loadtxt(pr_cache_path)[:,0]
        recall=np.loadtxt(pr_cache_path)[:,1]
        fm=np.loadtxt(fm_cache_path)
        fm_x=np.array([i for i in range(1,256)])
        plot_pr_vals[method]=(recall,prec)
        plot_fm_vals[method]=(fm_x,fm)
        
    # method_dir=os.path.join('/home/xxx/SOD_Evaluation_Metrics/curve_cache_BGCANet/ORSSD/Ours')
    # pr_cache_path=os.path.join(method_dir,'pr.txt')
    # fm_cache_path=os.path.join(method_dir,'fm.txt')
    # prec=np.loadtxt(pr_cache_path)[:,0]
    # recall=np.loadtxt(pr_cache_path)[:,1]
    # fm=np.loadtxt(fm_cache_path)
    # fm_x=np.array([i for i in range(1,256)])
    # plot_pr_vals['Ours']=(recall,prec)
    # plot_fm_vals['Ours']=(fm_x,fm)    

    # methods = methods + ['Ours']
    # print(methods)
    plt.clf()
    colors = 'bmcgybmcgyr';  # 'rkbmc'
    ticks = ['--','-']
    # for i, m in enumerate(['HVPNet','DNTD', 'DAFNet', 'ACCoNet', 'UG2L', 'SDNet',  'ADSTNet', 'HFCNet', 'SFANet']):
    for i, m in enumerate(methods):
        x, y = plot_pr_vals[m]
        marker = colors[i % len(colors)] + ticks[i % 2]
        plt.plot(x, y, marker, linewidth=1.4, label=m)
    # x, y = plot_pr_vals['Ours']
    # marker = colors[-1] + ticks[-1]
    plt.plot(x, y, marker, linewidth=2, label=m)
    # ax = plt.gca()
    # from matplotlib.pyplot import MultipleLocator
    # ax.xaxis.set_major_locator(MultipleLocator(0.2))

    plt.grid(True)
    _font_size_ = 16
    plt.title(dataset, fontsize=_font_size_ + 2)
    # plt.xlim([0.55, 1.0]);  # plt.ylim([0.0, 1.0])
    plt.xlim([0, 1.0]);  # plt.ylim([0.0, 1.0])
    plt.xlabel("Recall", fontsize=_font_size_);
    plt.xticks(fontsize=_font_size_ - 4)
    plt.ylabel("Precision", fontsize=_font_size_);
    plt.yticks(fontsize=_font_size_ - 4)
    plt.legend(methods, loc='lower left', fontsize=_font_size_ - 2, framealpha=0.75)
    plt.savefig(os.path.join(curve_save_dir, '{}_pr.png'.format(dataset)), bbox_inches='tight')
    # plt.show()

    plt.clf()
    colors = 'bmcgybmcgyr';
    ticks = ['--','-']
    # for i, m in enumerate(['HVPNet','DNTD', 'DAFNet', 'ACCoNet', 'UG2L', 'SDNet',  'ADSTNet', 'HFCNet', 'SFANet']):
    for i, m in enumerate(methods):
        x, y = plot_fm_vals[m]
        marker = colors[i % len(colors)] + ticks[i % 2]
        plt.plot(x, y, marker, linewidth=1.4, label=m)
    # x, y = plot_fm_vals['Ours']
    # marker = colors[-1] + ticks[-1]
    plt.plot(x, y, marker, linewidth=2, label=m)

    plt.grid(True)
    _font_size_ = 16
    plt.title(dataset, fontsize=_font_size_ + 2)
    plt.xlim([0, 255]);  # plt.ylim([0.0, 1.0])
    plt.xlabel("Threshold", fontsize=_font_size_);
    plt.xticks(fontsize=_font_size_ - 4)
    plt.ylabel("F-measure", fontsize=_font_size_);
    plt.yticks(fontsize=_font_size_ - 4)
    plt.legend(methods, loc='lower left', fontsize=_font_size_ - 2, framealpha=0.75)
    plt.savefig(os.path.join(curve_save_dir, '{}_fm.png'.format(dataset)), bbox_inches='tight')

