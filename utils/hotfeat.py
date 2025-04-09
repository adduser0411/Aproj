import torch
import numpy as np
import cv2
import os
import os.path as osp
import torch.nn.functional as F

# 关于热力图==================================
# 模型调试：通过可视化特征图，开发者可以检查模型在不同层的输出，判断模型是否正常学习到有效的特征。
# 特征理解：帮助研究者理解模型在处理图像时关注的区域，从而更好地解释模型的决策过程。
# 模型优化：根据特征图的可视化结果，调整模型结构或超参数，以提高模型性能
output_shape = (256,256) # 输出形状
# ============================================


def feature_vis(feats, name, isMax=False, save_path=osp.dirname(osp.abspath(__file__))): # feaats形状: [b,c,h,w]
    '''
    feats: 输入的特征图
    name: 保存的文件名
    isMax: 是否使用最大值进行可视化
    save_path: 保存路径
    '''
    # print('进入函数')
    # print("feats.size()",feats.size())

    if isMax:
        channel_mean , _ = torch.max(feats,dim=1,keepdim=True) # 沿着通道维度求最大值，得到每个空间位置的【最大特征响应图】 channel_mean。
    else:
        channel_mean = torch.mean(feats,dim=1,keepdim=True)  #对输入的特征图 feats 沿着通道维度求均值，得到每个空间位置的【平均特征响应图】 channel_mean。
        
    # print("channel_mean1.size()",channel_mean.size())
    
    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False) #使用双线性插值方法将通道均值特征图上采样到指定的输出形状 (256, 256)。
    # print("channel_meanF.size()",channel_mean.size())
    # print(type(channel_mean))
    
    channel_mean = channel_mean[0].squeeze(0).squeeze(0).cpu().detach().numpy() # 处理后的特征图从 torch.Tensor 转换为 numpy 数组，并压缩维度至二维。
    # channel_mean = channel_mean[0][0]
    # print("channel_mean2.size()",channel_mean.size())
    
    #channel_mean = (((channel_mean - np.min(channel_mean))/(np.min(channel_mean)-np.max(channel_mean)))*255).astype(np.uint8) 
    channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8) #将特征图的值归一化到 0 - 255 范围，并转换为 uint8 类型，以便后续使用 OpenCV 处理。
    #print("channel_mean3.size()",channel_mean.size())
    save_path=osp.join(save_path,'feature_sem_vis_mean_min-max1')
    if not os.path.exists(save_path): os.makedirs(save_path) 
    save_path=save_path+'/'+ name+'.png'
    
    # channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET) # 应用 JET 颜色映射，将灰度特征图转换为彩色图像。
    # cv2.imwrite(save_path,channel_mean ) # 将彩色特征图保存为图像文件。

    # # 同时保存原本的feats:
    # feats_np = feats.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()    # 转换 feats 为 numpy 数组
    # if feats_np.max() > 0:# 归一化到 0-255 并转换为 uint8 类型
    #     feats_np = (feats_np / feats_np.max() * 255).astype(np.uint8)    
    # else:
    #     feats_np = feats_np.astype(np.uint8)
    # cv2.imwrite(save_path, feats_np)
    return save_path

    
if __name__ == '__main__':
    feats=torch.rand(16,32,88,88) # 四维，batchsize,channel,h,w
    name='123'
    feature_vis(feats,name)